import os
import gc
import json
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
THRESHOLD = 0.39
LOCAL_DATA_DIR = "/home/mahith/BDA_PROJECT/Model/lstm_clean/rechunked_sequences"
LOCAL_TEST_PATH = os.path.join(LOCAL_DATA_DIR, "test_sequences")
MODEL_DIR = "/home/mahith/BDA_PROJECT/Model/lstm_clean/results"
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_adaptive_best.pt")
NORMALIZATION_PATH = os.path.join(MODEL_DIR, "lstm_normalization_stats.json")
RESULTS_PATH = os.path.join(MODEL_DIR, "lstm_final_test_results.txt")
CSV_RESULTS_PATH = os.path.join(MODEL_DIR, "lstm_final_test_predictions.csv")
JSON_RESULTS_PATH = os.path.join(MODEL_DIR, "lstm_final_test_metrics.json")
os.makedirs(MODEL_DIR, exist_ok=True)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(2)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("LSTM FINAL TEST EVALUATION")
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
class TestDataset(IterableDataset):
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
            temporal_i = (temporal[i] - self.temporal_mean) / self.temporal_std
            static_i = (static[i] - self.static_mean) / self.static_std
            temporal_i = np.nan_to_num(temporal_i, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
            static_i = np.nan_to_num(static_i, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
            yield temporal_i, missing[i].astype(np.float32, copy=False), padding[i], static_i, int(labels[i])
    def __iter__(self):
        dataset = ds.dataset(self.path, format="parquet")
        scanner = dataset.scanner(
            columns=["temporal", "missing_mask", "padding_mask", "static", "label"],
            batch_size=ARROW_BATCH_SIZE,
            use_threads=False
        )
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
    return torch.cat([temporal, missing_mask], dim=2), padding_mask, static, labels
class ICU_LSTM(nn.Module):
    def __init__(self, temporal_count, static_count, hidden_size, num_layers, dropout, static_hidden, fusion_hidden):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=temporal_count,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.static_network = nn.Sequential(
            nn.Linear(static_count, static_hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size + static_hidden, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1)
        )
    def forward(self, temporal, padding_mask, static):
        lengths = torch.clamp((padding_mask == 0).sum(dim=1).long(), min=1, max=MAX_SEQ_LEN)
        packed_sequence = nn.utils.rnn.pack_padded_sequence(
            temporal,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed_sequence)
        temporal_output = hidden[-1]
        static_output = self.static_network(static)
        return self.fusion(torch.cat([temporal_output, static_output], dim=1)).squeeze(1)
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Best LSTM model not found: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = ICU_LSTM(
        LSTM_INPUT_SIZE,
        STATIC_COUNT,
        HIDDEN_SIZE,
        NUM_LAYERS,
        DROPOUT,
        STATIC_HIDDEN,
        FUSION_HIDDEN
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Best LSTM checkpoint loaded")
    print(f"Best epoch: {checkpoint.get('best_epoch')}")
    print(f"Validation PR AUC: {checkpoint.get('best_validation_pr_auc'):.6f}")
    print(f"Validation ROC AUC: {checkpoint.get('best_validation_roc_auc'):.6f}")
    return model
@torch.no_grad()
def generate_predictions(model, loader):
    all_probabilities = []
    all_labels = []
    total = 0
    print("Generating test predictions")
    for temporal, padding_mask, static, labels in loader:
        temporal = temporal.to(device, non_blocking=True)
        padding_mask = padding_mask.to(device, non_blocking=True)
        static = static.to(device, non_blocking=True)
        probabilities = torch.sigmoid(model(temporal, padding_mask, static))
        all_probabilities.append(probabilities.cpu().numpy())
        all_labels.append(labels.numpy())
        total += labels.size(0)
        if total % 50000 < labels.size(0):
            print(f"Test samples processed: {total:,}")
    return np.concatenate(all_labels), np.concatenate(all_probabilities)
def evaluate(y_true, y_prob):
    y_pred = (y_prob >= THRESHOLD).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if tn + fp > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    return {
        "threshold": THRESHOLD,
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "f2": f2,
        "accuracy": accuracy,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }
def main():
    print(f"\nTest dataset: {LOCAL_TEST_PATH}")
    print(f"Fixed threshold: {THRESHOLD:.2f}")
    validate_dataset(LOCAL_TEST_PATH)
    test_rows = count_rows(LOCAL_TEST_PATH)
    print(f"Test rows: {test_rows:,}")
    if test_rows != 1664239:
        raise RuntimeError(f"Test row count mismatch: expected 1664239, found {test_rows}")
    temporal_mean, temporal_std, static_mean, static_std = load_normalization_statistics()
    model = load_model()
    dataset = TestDataset(
        LOCAL_TEST_PATH,
        temporal_mean,
        temporal_std,
        static_mean,
        static_std
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate_batch,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    y_true, y_prob = generate_predictions(model, loader)
    print("FINAL TEST PREDICTIONS COMPLETED")
    print(f"Labels      : {len(y_true):,}")
    print(f"Predictions : {len(y_prob):,}")
    if len(y_true) != len(y_prob):
        raise RuntimeError("Prediction and label counts do not match")
    if len(np.unique(y_true)) < 2:
        raise RuntimeError("Test data contains only one class")
    metrics = evaluate(y_true, y_prob)
    print("FINAL TEST RESULTS")
    print(f"Threshold   : {metrics['threshold']:.2f}")
    print(f"ROC-AUC     : {metrics['roc_auc']:.6f}")
    print(f"PR-AUC      : {metrics['pr_auc']:.6f}")
    print(f"Precision   : {metrics['precision']:.6f}")
    print(f"Recall      : {metrics['recall']:.6f}")
    print(f"Specificity : {metrics['specificity']:.6f}")
    print(f"F1          : {metrics['f1']:.6f}")
    print(f"F2          : {metrics['f2']:.6f}")
    print(f"Accuracy    : {metrics['accuracy']:.6f}")
    print("CONFUSION MATRIX")
    print(np.array([
        [metrics["tn"], metrics["fp"]],
        [metrics["fn"], metrics["tp"]]
    ]))
    print(f"TN: {metrics['tn']:,}")
    print(f"FP: {metrics['fp']:,}")
    print(f"FN: {metrics['fn']:,}")
    print(f"TP: {metrics['tp']:,}")
    predictions_df = pd.DataFrame({
        "label": y_true.astype(np.int8),
        "probability": y_prob,
        "prediction": (y_prob >= THRESHOLD).astype(np.int8)
    })
    predictions_df.to_csv(CSV_RESULTS_PATH, index=False)
    with open(RESULTS_PATH, "w") as f:
        f.write("LSTM FINAL TEST EVALUATION\n")
        f.write(f"Test rows: {len(y_true)}\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    json_metrics = {
        key: int(value) if key in ["tn", "fp", "fn", "tp"] else float(value)
        for key, value in metrics.items()
    }
    with open(JSON_RESULTS_PATH, "w") as f:
        json.dump({
            "model_path": MODEL_PATH,
            "test_path": LOCAL_TEST_PATH,
            "test_rows": int(len(y_true)),
            "threshold": THRESHOLD,
            "metrics": json_metrics
        }, f, indent=2)
    print("FILES SAVED")
    print(RESULTS_PATH)
    print(CSV_RESULTS_PATH)
    print(JSON_RESULTS_PATH)
    print("Final test evaluation completed successfully.")

    del model, loader, dataset, y_true, y_prob
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
if __name__ == "__main__":
    main()