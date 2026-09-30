import os
import gc
import json
import random
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.compute as pc
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, fbeta_score, accuracy_score, confusion_matrix
SEED = 42
MAX_SEQ_LEN = 120
EXPECTED_FEATURES = 137
TEMPORAL_COUNT = 54
STATIC_COUNT = 83
LSTM_INPUT_SIZE = 108
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.30
STATIC_HIDDEN = 64
FUSION_HIDDEN = 64
BATCH_SIZE = 128
ARROW_BATCH_SIZE = 64
LOCAL_DATA_DIR = "/home/mahith/BDA_PROJECT/Model/lstm_clean/rechunked_sequences"
LOCAL_VAL_PATH = os.path.join(LOCAL_DATA_DIR, "validation_sequences")
MODEL_DIR = "/home/mahith/BDA_PROJECT/Model/lstm_clean/results"
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_adaptive_best.pt")
NORMALIZATION_PATH = os.path.join(MODEL_DIR, "lstm_normalization_stats.json")
RESULTS_PATH = os.path.join(MODEL_DIR, "lstm_threshold_optimization_results.txt")
CSV_RESULTS_PATH = os.path.join(MODEL_DIR, "lstm_threshold_optimization_results.csv")
JSON_RESULTS_PATH = os.path.join(MODEL_DIR, "lstm_optimal_threshold.json")
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)
RECALL_FLOORS = [0.70, 0.75, 0.80, 0.85, 0.90]
PRIMARY_RECALL_TARGET = 0.80
os.makedirs(MODEL_DIR, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.set_num_threads(2)
if torch.backends.cudnn.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("LSTM THRESHOLD OPTIMIZATION")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
def count_rows(path):
    return ds.dataset(path, format="parquet").count_rows()
def validate_dataset(path):
    dataset = ds.dataset(path, format="parquet")
    required = {"temporal", "missing_mask", "padding_mask", "static", "sequence_length", "label"}
    missing = required - set(dataset.schema.names)
    if missing:
        raise RuntimeError(f"Missing columns: {sorted(missing)}")
    print(f"Dataset schema validation passed: {path}")
def load_normalization_statistics():
    if not os.path.exists(NORMALIZATION_PATH):
        raise FileNotFoundError(f"Normalization file not found: {NORMALIZATION_PATH}")
    with open(NORMALIZATION_PATH, "r") as f:
        stats = json.load(f)
    temporal_mean = np.asarray(stats["temporal_mean"], dtype=np.float32)
    temporal_std = np.asarray(stats["temporal_std"], dtype=np.float32)
    static_mean = np.asarray(stats["static_mean"], dtype=np.float32)
    static_std = np.asarray(stats["static_std"], dtype=np.float32)
    if len(temporal_mean) != TEMPORAL_COUNT or len(temporal_std) != TEMPORAL_COUNT:
        raise RuntimeError("Invalid temporal normalization dimension")
    if len(static_mean) != STATIC_COUNT or len(static_std) != STATIC_COUNT:
        raise RuntimeError("Invalid static normalization dimension")
    print("Train-only normalization statistics loaded")
    return temporal_mean, temporal_std, static_mean, static_std
class ValidationDataset(IterableDataset):
    def __init__(self, path, temporal_mean, temporal_std, static_mean, static_std):
        self.path = path
        self.temporal_mean = temporal_mean
        self.temporal_std = temporal_std
        self.static_mean = static_mean
        self.static_std = static_std
    def read_batch(self, record_batch):
        temporal_flat = pc.list_flatten(pc.list_flatten(record_batch.column("temporal"))).to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        missing_flat = pc.list_flatten(pc.list_flatten(record_batch.column("missing_mask"))).to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        padding_flat = pc.list_flatten(record_batch.column("padding_mask")).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        static_flat = pc.list_flatten(record_batch.column("static")).to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        labels = record_batch.column("label").to_numpy(zero_copy_only=False)
        row_count = len(labels)
        temporal = temporal_flat.reshape(row_count, MAX_SEQ_LEN, TEMPORAL_COUNT)
        missing = missing_flat.reshape(row_count, MAX_SEQ_LEN, TEMPORAL_COUNT)
        padding = padding_flat.reshape(row_count, MAX_SEQ_LEN)
        static = static_flat.reshape(row_count, STATIC_COUNT)
        for i in range(row_count):
            temporal_i = temporal[i].copy()
            missing_i = missing[i].copy()
            padding_i = padding[i].copy()
            static_i = static[i].copy()
            temporal_i = (temporal_i - self.temporal_mean) / self.temporal_std
            static_i = (static_i - self.static_mean) / self.static_std
            temporal_i = np.nan_to_num(temporal_i, nan=0.0, posinf=0.0, neginf=0.0)
            static_i = np.nan_to_num(static_i, nan=0.0, posinf=0.0, neginf=0.0)
            yield temporal_i.astype(np.float32, copy=False), missing_i.astype(np.float32, copy=False), padding_i, static_i.astype(np.float32, copy=False), int(labels[i])
    def __iter__(self):
        dataset = ds.dataset(self.path, format="parquet")
        scanner = dataset.scanner(columns=["temporal", "missing_mask", "padding_mask", "static", "sequence_length", "label"], batch_size=ARROW_BATCH_SIZE, use_threads=False)
        for record_batch in scanner.to_batches():
            yield from self.read_batch(record_batch)
            del record_batch
        gc.collect()
def collate_batch(batch):
    temporal = torch.from_numpy(np.stack([x[0] for x in batch]))
    missing_mask = torch.from_numpy(np.stack([x[1] for x in batch]))
    padding_mask = torch.from_numpy(np.stack([x[2] for x in batch]))
    static = torch.from_numpy(np.stack([x[3] for x in batch]))
    labels = torch.tensor([x[4] for x in batch], dtype=torch.float32)
    temporal_with_missing = torch.cat([temporal, missing_mask], dim=2)
    return temporal_with_missing, padding_mask, static, labels
class ICU_LSTM(nn.Module):
    def __init__(self, temporal_count, static_count, hidden_size, num_layers, dropout, static_hidden, fusion_hidden):
        super().__init__()
        self.lstm = nn.LSTM(input_size=temporal_count, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.static_network = nn.Sequential(nn.Linear(static_count, static_hidden), nn.ReLU(), nn.Dropout(dropout))
        self.fusion = nn.Sequential(nn.Linear(hidden_size + static_hidden, fusion_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(fusion_hidden, 1))
    def forward(self, temporal, padding_mask, static):
        lengths = (padding_mask == 0).sum(dim=1).long()
        lengths = torch.clamp(lengths, min=1, max=MAX_SEQ_LEN)
        packed_sequence = nn.utils.rnn.pack_padded_sequence(temporal, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed_sequence)
        temporal_output = hidden[-1]
        static_output = self.static_network(static)
        fused_output = torch.cat([temporal_output, static_output], dim=1)
        return self.fusion(fused_output).squeeze(1)
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Best LSTM model not found: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = ICU_LSTM(LSTM_INPUT_SIZE, STATIC_COUNT, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, STATIC_HIDDEN, FUSION_HIDDEN).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Best LSTM checkpoint loaded")
    print(f"Best epoch: {checkpoint.get('best_epoch')}")
    print(f"Best validation PR AUC: {checkpoint.get('best_validation_pr_auc'):.6f}")
    print(f"Best validation ROC AUC: {checkpoint.get('best_validation_roc_auc'):.6f}")
    return model
@torch.no_grad()
def generate_predictions(model, loader):
    all_probabilities = []
    all_labels = []
    total = 0
    print("\nGenerating validation predictions")
    for temporal, padding_mask, static, labels in loader:
        temporal = temporal.to(device, non_blocking=True)
        padding_mask = padding_mask.to(device, non_blocking=True)
        static = static.to(device, non_blocking=True)
        logits = model(temporal, padding_mask, static)
        probabilities = torch.sigmoid(logits)
        all_probabilities.append(probabilities.cpu().numpy())
        all_labels.append(labels.numpy())
        total += labels.size(0)
        if total % 50000 < labels.size(0):
            print(f"Validation samples processed: {total:,}")
        del temporal, padding_mask, static, labels, logits, probabilities
    y_prob = np.concatenate(all_probabilities)
    y_true = np.concatenate(all_labels)
    return y_true, y_prob
def evaluate_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    youden_j = recall + specificity - 1.0
    return {"threshold": float(threshold), "precision": float(precision), "recall": float(recall), "specificity": float(specificity), "f1": float(f1), "f2": float(f2), "youden_j": float(youden_j), "accuracy": float(accuracy), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
def main():
    print(f"\nValidation dataset: {LOCAL_VAL_PATH}")
    validate_dataset(LOCAL_VAL_PATH)
    validation_rows = count_rows(LOCAL_VAL_PATH)
    print(f"Validation rows: {validation_rows:,}")
    temporal_mean, temporal_std, static_mean, static_std = load_normalization_statistics()
    model = load_model()
    dataset = ValidationDataset(LOCAL_VAL_PATH, temporal_mean, temporal_std, static_mean, static_std)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_batch, num_workers=0, pin_memory=torch.cuda.is_available())
    y_true, y_prob = generate_predictions(model, loader)
    print("VALIDATION PREDICTIONS COMPLETED")
    print(f"Labels      : {len(y_true):,}")
    print(f"Predictions : {len(y_prob):,}")
    if len(y_true) != len(y_prob):
        raise RuntimeError("Prediction and label counts do not match")
    if len(np.unique(y_true)) < 2:
        raise RuntimeError("Validation data contains only one class")
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    print("\nVALIDATION METRICS")
    print(f"ROC-AUC : {roc_auc:.6f}")
    print(f"PR-AUC  : {pr_auc:.6f}")
    print("SEARCHING FOR OPTIMAL THRESHOLD")
    results = [evaluate_threshold(y_true, y_prob, threshold) for threshold in THRESHOLDS]
    results_df = pd.DataFrame(results)
    best_f1_row = results_df.loc[results_df["f1"].idxmax()]
    best_f2_row = results_df.loc[results_df["f2"].idxmax()]
    best_youden_row = results_df.loc[results_df["youden_j"].idxmax()]
    recall_candidates = {}
    for recall_floor in RECALL_FLOORS:
        candidates = results_df[results_df["recall"] >= recall_floor]
        if len(candidates) == 0:
            recall_candidates[str(recall_floor)] = None
        else:
            best_precision_row = candidates.loc[candidates["precision"].idxmax()]
            recall_candidates[str(recall_floor)] = {key: (float(best_precision_row[key]) if key not in ["tn", "fp", "fn", "tp"] else int(best_precision_row[key])) for key in ["threshold", "precision", "recall", "specificity", "f1", "f2", "youden_j", "accuracy", "tn", "fp", "fn", "tp"]}
    primary_key = str(PRIMARY_RECALL_TARGET)
    if recall_candidates[primary_key] is not None:
        recommended_threshold = float(recall_candidates[primary_key]["threshold"])
        recommendation_method = "best_precision_with_recall_at_least_80_percent"
    else:
        recommended_threshold = float(best_f2_row["threshold"])
        recommendation_method = "best_f2"
    print("BEST F1 THRESHOLD")
    print(f"Threshold   : {best_f1_row['threshold']:.2f}")
    print(f"Precision   : {best_f1_row['precision']:.6f}")
    print(f"Recall      : {best_f1_row['recall']:.6f}")
    print(f"Specificity : {best_f1_row['specificity']:.6f}")
    print(f"F1          : {best_f1_row['f1']:.6f}")
    print(f"F2          : {best_f1_row['f2']:.6f}")
    print(f"Accuracy    : {best_f1_row['accuracy']:.6f}")
    print("BEST F2 THRESHOLD")
    print(f"Threshold   : {best_f2_row['threshold']:.2f}")
    print(f"Precision   : {best_f2_row['precision']:.6f}")
    print(f"Recall      : {best_f2_row['recall']:.6f}")
    print(f"Specificity : {best_f2_row['specificity']:.6f}")
    print(f"F1          : {best_f2_row['f1']:.6f}")
    print(f"F2          : {best_f2_row['f2']:.6f}")
    print(f"Accuracy    : {best_f2_row['accuracy']:.6f}")
    print("BEST YOUDEN'S J THRESHOLD")
    print(f"Threshold   : {best_youden_row['threshold']:.2f}")
    print(f"Precision   : {best_youden_row['precision']:.6f}")
    print(f"Recall      : {best_youden_row['recall']:.6f}")
    print(f"Specificity : {best_youden_row['specificity']:.6f}")
    print(f"Youden's J  : {best_youden_row['youden_j']:.6f}")
    print(f"F1          : {best_youden_row['f1']:.6f}")
    print(f"F2          : {best_youden_row['f2']:.6f}")
    print(f"Accuracy    : {best_youden_row['accuracy']:.6f}")
    print("RECALL-CONSTRAINED THRESHOLDS")
    for recall_floor in RECALL_FLOORS:
        result = recall_candidates[str(recall_floor)]
        if result is None:
            print(f"Recall >= {recall_floor:.0%}: No threshold found")
        else:
            print(f"Recall >= {recall_floor:.0%}: Threshold={result['threshold']:.2f}, Precision={result['precision']:.6f}, Recall={result['recall']:.6f}, Specificity={result['specificity']:.6f}")
    print("RECOMMENDED VALIDATION THRESHOLD")
    print(f"Threshold : {recommended_threshold:.2f}")
    print(f"Method    : {recommendation_method}")
    recommended_row = results_df[results_df["threshold"] == recommended_threshold].iloc[0]
    print(f"Precision : {recommended_row['precision']:.6f}")
    print(f"Recall    : {recommended_row['recall']:.6f}")
    print(f"F1        : {recommended_row['f1']:.6f}")
    print(f"F2        : {recommended_row['f2']:.6f}")
    print(f"Specificity: {recommended_row['specificity']:.6f}")
    print("CONFUSION MATRIX AT RECOMMENDED THRESHOLD")
    print(np.array([[int(recommended_row["tn"]), int(recommended_row["fp"])], [int(recommended_row["fn"]), int(recommended_row["tp"])]]))
    comparison_thresholds = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    print("THRESHOLD COMPARISON")
    print(f"{'Threshold':<12}{'Precision':<12}{'Recall':<12}{'Specificity':<14}{'F1':<12}{'F2':<12}{'YoudenJ':<12}")
    for threshold in comparison_thresholds:
        row = results_df[results_df["threshold"] == threshold]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        print(f"{threshold:<12.2f}{row['precision']:<12.4f}{row['recall']:<12.4f}{row['specificity']:<14.4f}{row['f1']:<12.4f}{row['f2']:<12.4f}{row['youden_j']:<12.4f}")
    results_df.to_csv(CSV_RESULTS_PATH, index=False)
    with open(RESULTS_PATH, "w") as f:
        f.write("LSTM THRESHOLD OPTIMIZATION\n")
        f.write(f"Validation rows: {len(y_true)}\n")
        f.write(f"ROC-AUC: {roc_auc:.6f}\n")
        f.write(f"PR-AUC: {pr_auc:.6f}\n\n")
        f.write("BEST F1 THRESHOLD\n")
        for key in ["threshold", "precision", "recall", "specificity", "f1", "f2", "accuracy", "tn", "fp", "fn", "tp"]:
            f.write(f"{key}: {best_f1_row[key]}\n")
        f.write("\nBEST F2 THRESHOLD\n")
        for key in ["threshold", "precision", "recall", "specificity", "f1", "f2", "accuracy", "tn", "fp", "fn", "tp"]:
            f.write(f"{key}: {best_f2_row[key]}\n")
        f.write("\nBEST YOUDEN'S J THRESHOLD (reference only, not the recommendation)\n")
        for key in ["threshold", "precision", "recall", "specificity", "f1", "f2", "youden_j", "accuracy", "tn", "fp", "fn", "tp"]:
            f.write(f"{key}: {best_youden_row[key]}\n")
        f.write("\nRECALL-CONSTRAINED THRESHOLDS\n")
        for recall_floor in RECALL_FLOORS:
            result = recall_candidates[str(recall_floor)]
            if result is None:
                f.write(f"Recall >= {recall_floor:.0%}: No threshold found\n")
            else:
                f.write(f"Recall >= {recall_floor:.0%}: Threshold={result['threshold']:.2f}, Precision={result['precision']:.6f}, Recall={result['recall']:.6f}, Specificity={result['specificity']:.6f}\n")
        f.write("\nRECOMMENDED VALIDATION THRESHOLD\n")
        f.write(f"Threshold: {recommended_threshold:.2f}\n")
        f.write(f"Method: {recommendation_method}\n")
    threshold_data = {
        "model_path": MODEL_PATH,
        "validation_path": LOCAL_VAL_PATH,
        "validation_rows": int(len(y_true)),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "best_f1": {key: (float(best_f1_row[key]) if key not in ["tn", "fp", "fn", "tp"] else int(best_f1_row[key])) for key in ["threshold", "precision", "recall", "specificity", "f1", "f2", "accuracy", "tn", "fp", "fn", "tp"]},
        "best_f2": {key: (float(best_f2_row[key]) if key not in ["tn", "fp", "fn", "tp"] else int(best_f2_row[key])) for key in ["threshold", "precision", "recall", "specificity", "f1", "f2", "accuracy", "tn", "fp", "fn", "tp"]},
        "best_youden_j": {key: (float(best_youden_row[key]) if key not in ["tn", "fp", "fn", "tp"] else int(best_youden_row[key])) for key in ["threshold", "precision", "recall", "specificity", "f1", "f2", "youden_j", "accuracy", "tn", "fp", "fn", "tp"]},
        "recall_constrained_results": recall_candidates,
        "recommended_threshold": float(recommended_threshold),
        "recommendation_method": recommendation_method
    }
    with open(JSON_RESULTS_PATH, "w") as f:
        json.dump(threshold_data, f, indent=2)
    print("FILES SAVED")
    print(RESULTS_PATH)
    print(CSV_RESULTS_PATH)
    print(JSON_RESULTS_PATH)
    print("\nThreshold optimization completed successfully.")
    del model, loader, dataset, y_true, y_prob
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
if __name__ == "__main__":
    main()