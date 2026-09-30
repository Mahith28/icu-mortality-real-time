import os
import gc
import json
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
import pyarrow.dataset as ds
import pyarrow.compute as pc
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
import matplotlib.pyplot as plt
import shap
from scipy.stats import spearmanr
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
ARROW_BATCH_SIZE = 64
BACKGROUND_SIZE = 50
EXPLAIN_SIZE = 200
STABILITY_SEEDS = [42, 123, 456, 789, 999]
STABILITY_BACKGROUND_SIZE = 20
STABILITY_EXPLAIN_SIZE = 50
SHAP_NSAMPLES = 200
LOCAL_DATA_DIR = "/home/mahith/BDA_PROJECT/Model/lstm_clean/rechunked_sequences"
LOCAL_TEST_PATH = os.path.join(LOCAL_DATA_DIR, "test_sequences")
MODEL_DIR = "/home/mahith/BDA_PROJECT/Model/lstm_clean/results"
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_adaptive_best.pt")
NORMALIZATION_PATH = os.path.join(MODEL_DIR, "lstm_normalization_stats.json")
SHAP_DIR = os.path.join(MODEL_DIR, "lstm_shap")
SHAP_VALUES_PATH = os.path.join(SHAP_DIR, "lstm_shap_values.npz")
FEATURE_IMPORTANCE_PATH = os.path.join(SHAP_DIR, "lstm_shap_feature_importance.csv")
SAMPLE_PATH = os.path.join(SHAP_DIR, "lstm_shap_sample_predictions.csv")
SUMMARY_PATH = os.path.join(SHAP_DIR, "lstm_shap_summary.json")
BAR_PLOT_PATH = os.path.join(SHAP_DIR, "lstm_shap_bar.png")
SUMMARY_PLOT_PATH = os.path.join(SHAP_DIR, "lstm_shap_summary.png")
TEMPORAL_PLOT_PATH = os.path.join(SHAP_DIR, "lstm_shap_temporal_importance.png")
STATIC_PLOT_PATH = os.path.join(SHAP_DIR, "lstm_shap_static_importance.png")
TEMPORAL_NAMES = [
    "Heart_Rate_mean","Heart_Rate_min","Heart_Rate_max","Heart_Rate_last",
    "SBP_mean","SBP_min","SBP_max","SBP_last",
    "DBP_mean","DBP_min","DBP_max","DBP_last",
    "MAP_mean","MAP_min","MAP_max","MAP_last",
    "Respiratory_Rate_mean","Respiratory_Rate_min","Respiratory_Rate_max","Respiratory_Rate_last",
    "Temperature_mean","Temperature_min","Temperature_max","Temperature_last",
    "SpO2_mean","SpO2_min","SpO2_max","SpO2_last",
    "GCS_Eye_mean","GCS_Eye_min","GCS_Eye_max","GCS_Eye_last",
    "GCS_Verbal_mean","GCS_Verbal_min","GCS_Verbal_max","GCS_Verbal_last",
    "GCS_Motor_mean","GCS_Motor_min","GCS_Motor_max","GCS_Motor_last",
    "Heart_Rate_missing","SBP_missing","DBP_missing","MAP_missing","Respiratory_Rate_missing",
    "Temperature_missing","SpO2_missing","GCS_Eye_missing","GCS_Verbal_missing","GCS_Motor_missing",
    "HR_trend","MAP_trend","RR_trend","GCS_Total_last"
]
CLEAN_TEST_FEATURE_PATH = "hdfs://localhost:9000/user/mahith/icu/clean_features/test_features"
if len(TEMPORAL_NAMES) != TEMPORAL_COUNT:
    raise RuntimeError("Temporal feature-name count mismatch")
os.makedirs(SHAP_DIR, exist_ok=True)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.set_num_threads(2)
if torch.backends.cudnn.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("LSTM SHAP EXPLAINABILITY ANALYSIS")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"SHAP version: {shap.__version__}")
def validate_dataset(path):
    dataset = ds.dataset(path, format="parquet")
    required = {"temporal","missing_mask","padding_mask","static","sequence_length","label"}
    missing = required - set(dataset.schema.names)
    if missing:
        raise RuntimeError(f"Missing columns: {sorted(missing)}")
    return dataset
def load_normalization_statistics():
    with open(NORMALIZATION_PATH, "r") as f:
        stats = json.load(f)
    temporal_mean = np.asarray(stats["temporal_mean"], dtype=np.float32)
    temporal_std = np.asarray(stats["temporal_std"], dtype=np.float32)
    static_mean = np.asarray(stats["static_mean"], dtype=np.float32)
    static_std = np.asarray(stats["static_std"], dtype=np.float32)
    if len(temporal_mean) != TEMPORAL_COUNT or len(temporal_std) != TEMPORAL_COUNT:
        raise RuntimeError("Invalid temporal normalization dimensions")
    if len(static_mean) != STATIC_COUNT or len(static_std) != STATIC_COUNT:
        raise RuntimeError("Invalid static normalization dimensions")
    return temporal_mean, temporal_std, static_mean, static_std
class SampleDataset(IterableDataset):
    def __init__(self, path, temporal_mean, temporal_std, static_mean, static_std):
        self.path = path
        self.temporal_mean = temporal_mean
        self.temporal_std = temporal_std
        self.static_mean = static_mean
        self.static_std = static_std
    def read_batch(self, record_batch):
        sequence_lengths = record_batch.column("sequence_length").to_numpy(zero_copy_only=False)
        temporal_flat = pc.list_flatten(pc.list_flatten(record_batch.column("temporal"))).to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        missing_flat = pc.list_flatten(pc.list_flatten(record_batch.column("missing_mask"))).to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        padding_flat = pc.list_flatten(record_batch.column("padding_mask")).to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
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
            yield np.concatenate([temporal_i, missing[i].astype(np.float32, copy=False)], axis=1), padding[i], static_i, int(labels[i]), int(sequence_lengths[i])
    def __iter__(self):
        dataset = ds.dataset(self.path, format="parquet")
        scanner = dataset.scanner(columns=["temporal","missing_mask","padding_mask","static","sequence_length","label"], batch_size=ARROW_BATCH_SIZE, use_threads=False)
        for record_batch in scanner.to_batches():
            yield from self.read_batch(record_batch)
            del record_batch
        gc.collect()
class ICU_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=LSTM_INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, batch_first=True, dropout=DROPOUT)
        self.static_network = nn.Sequential(nn.Linear(STATIC_COUNT, STATIC_HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT))
        self.fusion = nn.Sequential(nn.Linear(HIDDEN_SIZE + STATIC_HIDDEN, FUSION_HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(FUSION_HIDDEN, 1))

    def forward(self, temporal, padding_mask, static):
        lengths = torch.clamp((padding_mask.round() == 0).sum(dim=1).long(), min=1, max=MAX_SEQ_LEN)
        packed = nn.utils.rnn.pack_padded_sequence(temporal, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        temporal_output = hidden[-1]
        static_output = self.static_network(static)
        return self.fusion(torch.cat([temporal_output, static_output], dim=1)).squeeze(1)
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = ICU_LSTM().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    best_pr_auc = checkpoint.get("best_validation_pr_auc", float("nan"))
    best_roc_auc = checkpoint.get("best_validation_roc_auc", float("nan"))
    print(f"Best epoch: {checkpoint.get('best_epoch')}")
    print(f"Validation PR-AUC: {best_pr_auc:.6f}")
    print(f"Validation ROC-AUC: {best_roc_auc:.6f}")
    return model
def collect_samples(seed=SEED, background_size=BACKGROUND_SIZE, explain_size=EXPLAIN_SIZE):
    temporal_mean, temporal_std, static_mean, static_std = load_normalization_statistics()
    dataset = SampleDataset(LOCAL_TEST_PATH, temporal_mean, temporal_std, static_mean, static_std)
    loader = DataLoader(dataset, batch_size=ARROW_BATCH_SIZE, num_workers=0)
    rng = np.random.default_rng(seed)
    reservoir = []
    seen = 0
    for batch in loader:
        temporal_batch, padding_batch, static_batch, labels_batch, lengths_batch = batch
        for i in range(temporal_batch.shape[0]):
            if int(lengths_batch[i]) != MAX_SEQ_LEN:
                continue
            seen += 1
            item = (temporal_batch[i].numpy(), padding_batch[i].numpy(), static_batch[i].numpy(), int(labels_batch[i]))
            if len(reservoir) < background_size + explain_size:
                reservoir.append(item)
            else:
                j = int(rng.integers(0, seen))
                if j < background_size + explain_size:
                    reservoir[j] = item
    if len(reservoir) < background_size + explain_size:
        raise RuntimeError(f"Only {len(reservoir)} full-length sequences found; need {background_size + explain_size}")
    rng.shuffle(reservoir)
    temporal = np.stack([x[0] for x in reservoir])
    padding = np.stack([x[1] for x in reservoir])
    static = np.stack([x[2] for x in reservoir])
    labels = np.asarray([x[3] for x in reservoir], dtype=np.int8)
    if not np.all(padding == 0):
        raise RuntimeError("Full-length SHAP samples must have zero padding masks")
    return temporal, padding, static, labels
class SHAPWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, temporal, static):
        padding = torch.zeros((temporal.shape[0], MAX_SEQ_LEN), dtype=torch.float32, device=temporal.device)
        return self.model(temporal, padding, static).unsqueeze(1)
def normalize_shap_output(values, expected_shape):
    if isinstance(values, list) and len(values) == 1:
        values = values[0]
    values = np.asarray(values)
    if values.shape == expected_shape:
        return values
    if values.ndim == len(expected_shape) + 1 and values.shape[-1] == 1:
        values = np.squeeze(values, axis=-1)
    return values
def build_feature_importance(temporal_values, static_values):
    temporal_names_full = TEMPORAL_NAMES + [f"mask::{x}" for x in TEMPORAL_NAMES]
    temporal_patient_values = temporal_values.sum(axis=1)
    temporal_rows = pd.DataFrame({
        "feature": temporal_names_full,
        "type": "temporal",
        "mean_shap": np.mean(temporal_patient_values, axis=0),
        "mean_abs_shap": np.mean(np.abs(temporal_patient_values), axis=0)
    })
    static_rows = pd.DataFrame({
        "feature": STATIC_NAMES,
        "type": "static",
        "mean_shap": np.mean(static_values, axis=0),
        "mean_abs_shap": np.mean(np.abs(static_values), axis=0)
    })
    return pd.concat([temporal_rows, static_rows], ignore_index=True).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
def make_plots(temporal_values, temporal_inputs, static_values, static_inputs):
    temporal_names_full = TEMPORAL_NAMES + [f"mask::{x}" for x in TEMPORAL_NAMES]
    temporal_patient_values = temporal_values.sum(axis=1)
    temporal_feature_importance = np.mean(np.abs(temporal_patient_values), axis=0)
    temporal_signed_importance = np.mean(temporal_patient_values, axis=0)
    static_feature_importance = np.mean(np.abs(static_values), axis=0)
    static_signed_importance = np.mean(static_values, axis=0)
    temporal_rows = pd.DataFrame({"feature": temporal_names_full, "mean_shap": temporal_signed_importance, "mean_abs_shap": temporal_feature_importance}).sort_values("mean_abs_shap", ascending=False)
    static_rows = pd.DataFrame({"feature": STATIC_NAMES, "mean_shap": static_signed_importance, "mean_abs_shap": static_feature_importance}).sort_values("mean_abs_shap", ascending=False)
    combined = pd.concat([temporal_rows.assign(type="temporal"), static_rows.assign(type="static")], ignore_index=True).sort_values("mean_abs_shap", ascending=False)
    top = combined.head(20).sort_values("mean_abs_shap")
    plt.figure(figsize=(12, 9))
    plt.barh(top["feature"], top["mean_abs_shap"])
    plt.xlabel("Mean Absolute SHAP Value")
    plt.ylabel("Feature")
    plt.title("LSTM SHAP Feature Importance - Top 20 Features")
    plt.tight_layout()
    plt.savefig(BAR_PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Top-20 SHAP PNG: {BAR_PLOT_PATH}")
    top_t = temporal_rows.head(20).sort_values("mean_abs_shap")
    plt.figure(figsize=(12, 9))
    plt.barh(top_t["feature"], top_t["mean_abs_shap"])
    plt.xlabel("Mean Absolute SHAP Value")
    plt.ylabel("Temporal feature")
    plt.title("LSTM Temporal SHAP Importance")
    plt.tight_layout()
    plt.savefig(TEMPORAL_PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close()
    top_s = static_rows.head(20).sort_values("mean_abs_shap")
    plt.figure(figsize=(12, 9))
    plt.barh(top_s["feature"], top_s["mean_abs_shap"])
    plt.xlabel("Mean Absolute SHAP Value")
    plt.ylabel("Static feature")
    plt.title("LSTM Static SHAP Importance")
    plt.tight_layout()
    plt.savefig(STATIC_PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close()
    combined_values = np.concatenate([temporal_patient_values, static_values], axis=1)
    combined_inputs = np.concatenate([temporal_inputs.mean(axis=1), static_inputs], axis=1)
    combined_names = temporal_names_full + STATIC_NAMES
    shap.summary_plot(combined_values, combined_inputs, feature_names=combined_names, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(SUMMARY_PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close()
    timestep_importance = np.mean(np.abs(temporal_values), axis=(0, 2))
    pd.DataFrame({"timestep": np.arange(1, MAX_SEQ_LEN + 1), "mean_abs_shap": timestep_importance}).to_csv(os.path.join(SHAP_DIR, "lstm_shap_timestep_importance.csv"), index=False)
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(1, MAX_SEQ_LEN + 1), timestep_importance)
    plt.xlabel("Timestep")
    plt.ylabel("Mean Absolute SHAP Value")
    plt.title("LSTM Temporal SHAP Importance Across Timesteps")
    plt.tight_layout()
    plt.savefig(os.path.join(SHAP_DIR, "lstm_shap_timestep_importance.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"SHAP summary PNG: {SUMMARY_PLOT_PATH}")
def calculate_shap(wrapper, bg_temporal, bg_static, ex_temporal, ex_static, seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn_state = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    try:
        explainer = shap.GradientExplainer(wrapper, [bg_temporal, bg_static])
        values = explainer.shap_values([ex_temporal, ex_static], nsamples=SHAP_NSAMPLES, rseed=seed)
    finally:
        torch.backends.cudnn.enabled = cudnn_state
    if not isinstance(values, list) or len(values) != 2:
        raise RuntimeError(f"Expected 2 SHAP input outputs, received {len(values) if isinstance(values, list) else type(values).__name__}")
    temporal_values = normalize_shap_output(values[0], tuple(ex_temporal.shape))
    static_values = normalize_shap_output(values[1], tuple(ex_static.shape))
    if temporal_values.shape != tuple(ex_temporal.shape):
        raise RuntimeError(f"Temporal SHAP shape mismatch: {temporal_values.shape}")
    if static_values.shape != tuple(ex_static.shape):
        raise RuntimeError(f"Static SHAP shape mismatch: {static_values.shape}")
    return temporal_values, static_values
def rank_stability(wrapper):
    rows = []
    rankings = {}
    names = TEMPORAL_NAMES + [f"mask::{x}" for x in TEMPORAL_NAMES] + STATIC_NAMES
    for seed in STABILITY_SEEDS:
        temporal, _, static, _ = collect_samples(seed=seed, background_size=STABILITY_BACKGROUND_SIZE, explain_size=STABILITY_EXPLAIN_SIZE)
        bg_temporal = torch.from_numpy(temporal[:STABILITY_BACKGROUND_SIZE]).to(device)
        bg_static = torch.from_numpy(static[:STABILITY_BACKGROUND_SIZE]).to(device)
        ex_temporal = torch.from_numpy(temporal[STABILITY_BACKGROUND_SIZE:]).to(device)
        ex_static = torch.from_numpy(static[STABILITY_BACKGROUND_SIZE:]).to(device)
        temporal_values, static_values = calculate_shap(wrapper, bg_temporal, bg_static, ex_temporal, ex_static, seed=seed)
        patient_values = temporal_values.sum(axis=1)
        importance = np.concatenate([np.mean(np.abs(patient_values), axis=0), np.mean(np.abs(static_values), axis=0)])
        rankings[seed] = importance
        top10 = list(pd.Series(importance, index=names).nlargest(10).index)
        rows.append({"seed": seed, "background_size": STABILITY_BACKGROUND_SIZE, "explain_size": STABILITY_EXPLAIN_SIZE, "top_10_features": "|".join(top10)})
        del temporal, static, bg_temporal, bg_static, ex_temporal, ex_static, temporal_values, static_values
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    pairwise = []
    seeds = list(rankings)
    for i, seed_a in enumerate(seeds):
        for j in range(i + 1, len(seeds)):
            seed_b = seeds[j]
            result = spearmanr(rankings[seed_a], rankings[seed_b])
            correlation = getattr(result, "statistic", getattr(result, "correlation", np.nan))
            spearman = float(correlation) if correlation is not None else float("nan")
            top_a = set(rows[i]["top_10_features"].split("|"))
            top_b = set(rows[j]["top_10_features"].split("|"))
            jaccard = float(len(top_a & top_b) / len(top_a | top_b))
            pairwise.append({"seed_a": seed_a, "seed_b": seed_b, "spearman_rank_correlation": spearman, "top_10_jaccard": jaccard})
    pd.DataFrame(rows).to_csv(os.path.join(SHAP_DIR, "lstm_shap_stability_seed_results.csv"), index=False)
    pd.DataFrame(pairwise).to_csv(os.path.join(SHAP_DIR, "lstm_shap_stability_pairwise.csv"), index=False)
    return {
    "seeds": STABILITY_SEEDS,
    "background_size": STABILITY_BACKGROUND_SIZE,
    "explain_size": STABILITY_EXPLAIN_SIZE,
    "pair_count": len(pairwise),
    "mean_spearman_rank_correlation": float(
        np.mean([x["spearman_rank_correlation"] for x in pairwise])
    ),
    "median_spearman_rank_correlation": float(
        np.median([x["spearman_rank_correlation"] for x in pairwise])
    ),
    "mean_top_10_jaccard": float(
        np.mean([x["top_10_jaccard"] for x in pairwise])
    ),
    "median_top_10_jaccard": float(
        np.median([x["top_10_jaccard"] for x in pairwise])
    ),
    "seed_results": rows
}
def load_feature_names():
    spark = SparkSession.builder.master("local[2]").appName("LSTM_SHAP_FeatureMetadata").getOrCreate()
    df = spark.read.parquet(CLEAN_TEST_FEATURE_PATH)
    metadata = df.schema["features"].metadata
    if "ml_attr" not in metadata:
        spark.stop()
        raise RuntimeError("ml_attr metadata not found in clean test features")
    attrs = metadata["ml_attr"]["attrs"]
    names = []
    for group in attrs.values():
        for item in group:
            names.append((int(item["idx"]), item["name"]))
    spark.stop()
    names = [name for _, name in sorted(names)]
    if len(names) != EXPECTED_FEATURES:
        raise RuntimeError(f"Expected {EXPECTED_FEATURES} feature names, got {len(names)}")
    if names[:TEMPORAL_COUNT] != TEMPORAL_NAMES:
        raise RuntimeError("Temporal feature metadata does not match the LSTM sequence feature order")
    return names[TEMPORAL_COUNT:]
def main():
    global STATIC_NAMES
    STATIC_NAMES = load_feature_names()
    if len(STATIC_NAMES) != STATIC_COUNT:
        raise RuntimeError("Static feature-name count mismatch")
    dataset = validate_dataset(LOCAL_TEST_PATH)
    model = load_model()
    temporal, padding, static, labels = collect_samples()
    print(f"Background samples: {BACKGROUND_SIZE}")
    print(f"Explain samples: {EXPLAIN_SIZE}")
    print(f"Temporal input shape: {temporal.shape}")
    print(f"Static input shape: {static.shape}")
    bg_temporal = torch.from_numpy(temporal[:BACKGROUND_SIZE]).to(device)
    bg_static = torch.from_numpy(static[:BACKGROUND_SIZE]).to(device)
    ex_temporal = torch.from_numpy(temporal[BACKGROUND_SIZE:]).to(device)
    ex_static = torch.from_numpy(static[BACKGROUND_SIZE:]).to(device)
    wrapper = SHAPWrapper(model).to(device)
    wrapper.eval()
    with torch.no_grad():
        logits = wrapper(ex_temporal, ex_static).squeeze(1).cpu().numpy()
        probabilities = 1.0 / (1.0 + np.exp(-logits))
    print("\nCreating SHAP GradientExplainer")
    print(f"Calculating SHAP values with nsamples={SHAP_NSAMPLES} and rseed={SEED}")
    temporal_values, static_values = calculate_shap(wrapper, bg_temporal, bg_static, ex_temporal, ex_static, seed=SEED)
    np.savez_compressed(SHAP_VALUES_PATH, temporal_shap=temporal_values, static_shap=static_values)
    importance = build_feature_importance(temporal_values, static_values)
    importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    pd.DataFrame({"sample": np.arange(EXPLAIN_SIZE), "label": labels[BACKGROUND_SIZE:], "logit": logits, "probability": probabilities}).to_csv(SAMPLE_PATH, index=False)
    top_features = importance.head(20).to_dict("records")
    with torch.no_grad():
        background_logits = wrapper(bg_temporal, bg_static).cpu().numpy().reshape(-1)
    baseline = float(np.mean(background_logits))
    reconstructed = baseline + temporal_values.reshape(len(temporal_values), -1).sum(axis=1) + static_values.sum(axis=1)
    efficiency_error = np.abs(reconstructed - logits)
    efficiency = {
        "baseline_mean_background_logit": baseline,
        "mean_absolute_reconstruction_error": float(np.mean(efficiency_error)),
        "median_absolute_reconstruction_error": float(np.median(efficiency_error)),
        "max_absolute_reconstruction_error": float(np.max(efficiency_error)),
        "explained_logit_std": float(np.std(logits)),
        "normalized_mean_error": float(np.mean(efficiency_error) / (np.std(logits) + 1e-12)),
        "nsamples": SHAP_NSAMPLES
    }
    pd.DataFrame([efficiency]).to_csv(os.path.join(SHAP_DIR, "lstm_shap_efficiency_check.csv"), index=False)
    overall_n = 0
    overall_positive = 0
    full_n = 0
    full_positive = 0
    pop_scanner = dataset.scanner(columns=["sequence_length", "label"], batch_size=ARROW_BATCH_SIZE, use_threads=False)
    for rb in pop_scanner.to_batches():
        lengths = rb.column("sequence_length").to_numpy(zero_copy_only=False)
        labs = rb.column("label").to_numpy(zero_copy_only=False)
        overall_n += len(labs)
        overall_positive += int(np.sum(labs))
        full = lengths == MAX_SEQ_LEN
        full_n += int(np.sum(full))
        full_positive += int(np.sum(labs[full]))
    coverage = 100.0 * full_n / overall_n if overall_n else 0.0
    overall_rate = float(overall_positive / overall_n) if overall_n else 0.0
    full_rate = float(full_positive / full_n) if full_n else 0.0
    population = pd.DataFrame([
        {
            "cohort": "overall_test",
            "n": overall_n,
            "mortality_rate": overall_rate
        },
        {
            "cohort": "full_length_test",
            "n": full_n,
            "mortality_rate": full_rate
        }
    ])
    population.to_csv(os.path.join(SHAP_DIR, "lstm_shap_population_characterization.csv"), index=False)
    print(f"Full-length coverage: {coverage:.2f}% ({full_n}/{overall_n})")
    print(f"Overall test mortality rate: {overall_rate:.4f}")
    print(f"Full-length mortality rate: {full_rate:.4f}")
    print("\nRunning SHAP stability analysis")
    stability = rank_stability(wrapper)
    print(f"Mean Spearman rank correlation: {stability['mean_spearman_rank_correlation']:.4f}")
    print(f"Mean top-10 Jaccard: {stability['mean_top_10_jaccard']:.4f}")
    summary = {
        "model_path": MODEL_PATH,
        "test_path": LOCAL_TEST_PATH,
        "feature_metadata_source": CLEAN_TEST_FEATURE_PATH,
        "background_samples": BACKGROUND_SIZE,
        "explained_samples": EXPLAIN_SIZE,
        "shap_nsamples": SHAP_NSAMPLES,
        "only_full_length_sequences": True,
        "total_test_sequences": int(overall_n),
        "full_length_test_sequences": int(full_n),
        "full_length_coverage_percent": coverage,
        "population_characterization": population.to_dict("records"),
        "temporal_shape": list(temporal_values.shape),
        "static_shape": list(static_values.shape),
        "shap_output": "raw_logit",
        "shap_efficiency_check": efficiency,
        "stability_analysis": stability,
        "top_20_features": top_features
    }
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    make_plots(temporal_values, ex_temporal.cpu().numpy(), static_values, ex_static.cpu().numpy())
    print("LSTM SHAP ANALYSIS COMPLETED")
    print(f"SHAP values       : {SHAP_VALUES_PATH}")
    print(f"Feature importance: {FEATURE_IMPORTANCE_PATH}")
    print(f"Sample predictions: {SAMPLE_PATH}")
    print(f"Summary            : {SUMMARY_PATH}")
    print(f"Top-20 plot        : {BAR_PLOT_PATH}")
    print(f"Temporal plot      : {TEMPORAL_PLOT_PATH}")
    print(f"Static plot        : {STATIC_PLOT_PATH}")
    print(f"SHAP summary plot  : {SUMMARY_PLOT_PATH}")
    print("\nTop 20 features:")
    print(importance.head(20).to_string(index=False))
    del model, wrapper, temporal, padding, static, labels, temporal_values, static_values, bg_temporal, bg_static, ex_temporal, ex_static, logits, probabilities, dataset, background_logits
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
if __name__ == "__main__":
    main()