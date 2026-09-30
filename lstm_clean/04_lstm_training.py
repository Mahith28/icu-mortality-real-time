import os
import gc
import json
import random
import numpy as np
import pyarrow.dataset as ds
import pyarrow.compute as pc
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
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
SHUFFLE_BUFFER = 512
EPOCHS = 30
PATIENCE = 6
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
GRADIENT_CLIP = 1.0
LOCAL_DATA_DIR = "/home/mahith/BDA_PROJECT/Model/lstm_clean/rechunked_sequences"
LOCAL_TRAIN_PATH = os.path.join(LOCAL_DATA_DIR, "train_sequences")
LOCAL_VAL_PATH = os.path.join(LOCAL_DATA_DIR, "validation_sequences")
MODEL_DIR = "/home/mahith/BDA_PROJECT/Model/lstm_clean/results"
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_adaptive_best.pt")
HISTORY_PATH = os.path.join(MODEL_DIR, "lstm_adaptive_training_history.json")
NORMALIZATION_PATH = os.path.join(MODEL_DIR, "lstm_normalization_stats.json")
IMPUTATION_ARTIFACT_PATH = "hdfs://localhost:9000/user/mahith/icu/lstm/artifacts/lstm_imputation_values.json"
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
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.1f} GB")
def count_parquet_rows(path):
    return ds.dataset(path, format="parquet").count_rows()
def validate_dataset_schema(path):
    dataset = ds.dataset(path, format="parquet")
    schema_names = set(dataset.schema.names)
    required_columns = {"temporal", "missing_mask", "padding_mask", "static", "sequence_length", "label"}
    missing_columns = required_columns - schema_names
    if missing_columns:
        raise RuntimeError(f"Missing required columns: {sorted(missing_columns)}")
    print(f"Dataset schema validation passed: {path}")
def validate_dataset_samples(path):
    dataset = ds.dataset(path, format="parquet")
    scanner = dataset.scanner(
        columns=["temporal", "missing_mask", "padding_mask", "static", "sequence_length", "label"],
        batch_size=ARROW_BATCH_SIZE,
        use_threads=False
    )
    record_batch = next(scanner.to_batches(), None)
    if record_batch is None:
        raise RuntimeError(f"Dataset contains no samples: {path}")
    temporal_flat = pc.list_flatten(pc.list_flatten(record_batch.column("temporal"))).to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    missing_flat = pc.list_flatten(pc.list_flatten(record_batch.column("missing_mask"))).to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    padding_flat = pc.list_flatten(record_batch.column("padding_mask")).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    static_flat = pc.list_flatten(record_batch.column("static")).to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    sequence_rows = record_batch.column("sequence_length").to_numpy(zero_copy_only=False)
    label_rows = record_batch.column("label").to_numpy(zero_copy_only=False)
    row_count = len(label_rows)
    temporal_rows = temporal_flat.reshape(row_count, MAX_SEQ_LEN, TEMPORAL_COUNT)
    missing_rows = missing_flat.reshape(row_count, MAX_SEQ_LEN, TEMPORAL_COUNT)
    padding_rows = padding_flat.reshape(row_count, MAX_SEQ_LEN)
    static_rows = static_flat.reshape(row_count, STATIC_COUNT)
    for i in range(row_count):
        if temporal_rows[i].shape != (MAX_SEQ_LEN, TEMPORAL_COUNT):
            raise ValueError(f"Invalid temporal shape: {temporal_rows[i].shape}")
        if missing_rows[i].shape != (MAX_SEQ_LEN, TEMPORAL_COUNT):
            raise ValueError(f"Invalid missing mask shape: {missing_rows[i].shape}")
        if padding_rows[i].shape != (MAX_SEQ_LEN,):
            raise ValueError(f"Invalid padding mask shape: {padding_rows[i].shape}")
        if static_rows[i].shape != (STATIC_COUNT,):
            raise ValueError(f"Invalid static shape: {static_rows[i].shape}")
        if int(sequence_rows[i]) < 1 or int(sequence_rows[i]) > MAX_SEQ_LEN:
            raise ValueError(f"Invalid sequence length: {int(sequence_rows[i])}")
        if int(np.sum(padding_rows[i] == 0)) != int(sequence_rows[i]):
            raise ValueError("Sequence length does not match padding mask")
        if not np.all(np.isin(padding_rows[i], [0, 1])):
            raise ValueError("Padding mask contains invalid values")
        if not np.all(padding_rows[i][:int(sequence_rows[i])] == 0):
            raise ValueError("Real timesteps are not packed at the beginning")
        if not np.all(padding_rows[i][int(sequence_rows[i]):] == 1):
            raise ValueError("Padding timesteps are not packed at the end")
        if int(label_rows[i]) not in (0, 1):
            raise ValueError(f"Invalid label: {int(label_rows[i])}")
    print(f"Dataset sample validation passed: {path}")
def fit_normalization_statistics(path):
    print("Fitting train-only normalization statistics")
    dataset = ds.dataset(path, format="parquet")
    scanner = dataset.scanner(columns=["temporal", "static", "sequence_length"], batch_size=ARROW_BATCH_SIZE, use_threads=False)
    temporal_sum = np.zeros(TEMPORAL_COUNT, dtype=np.float64)
    temporal_sum_sq = np.zeros(TEMPORAL_COUNT, dtype=np.float64)
    temporal_count = np.zeros(TEMPORAL_COUNT, dtype=np.int64)
    static_sum = np.zeros(STATIC_COUNT, dtype=np.float64)
    static_sum_sq = np.zeros(STATIC_COUNT, dtype=np.float64)
    static_count = np.zeros(STATIC_COUNT, dtype=np.int64)
    processed_rows = 0
    for record_batch in scanner.to_batches():
        temporal_flat = pc.list_flatten(pc.list_flatten(record_batch.column("temporal"))).to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        static_flat = pc.list_flatten(record_batch.column("static")).to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        sequence_rows = record_batch.column("sequence_length").to_numpy(zero_copy_only=False)
        row_count = len(sequence_rows)
        temporal_rows = temporal_flat.reshape(row_count, MAX_SEQ_LEN, TEMPORAL_COUNT)
        static_rows = static_flat.reshape(row_count, STATIC_COUNT)
        valid_mask = np.arange(MAX_SEQ_LEN)[None, :] < sequence_rows[:, None]
        valid_temporal = temporal_rows[valid_mask]
        finite_temporal = np.isfinite(valid_temporal)
        for feature_index in range(TEMPORAL_COUNT):
            values = valid_temporal[:, feature_index]
            valid_values = values[finite_temporal[:, feature_index]]
            if len(valid_values) > 0:
                temporal_sum[feature_index] += np.sum(valid_values, dtype=np.float64)
                temporal_sum_sq[feature_index] += np.sum(np.square(valid_values), dtype=np.float64)
                temporal_count[feature_index] += len(valid_values)
        finite_static = np.isfinite(static_rows)
        for feature_index in range(STATIC_COUNT):
            values = static_rows[:, feature_index]
            valid_values = values[finite_static[:, feature_index]]
            if len(valid_values) > 0:
                static_sum[feature_index] += np.sum(valid_values, dtype=np.float64)
                static_sum_sq[feature_index] += np.sum(np.square(valid_values), dtype=np.float64)
                static_count[feature_index] += len(valid_values)
        processed_rows += row_count
        del temporal_flat
        del static_flat
        del sequence_rows
        del temporal_rows
        del static_rows
        del valid_temporal
        del finite_temporal
        del finite_static
        del record_batch
    temporal_mean = temporal_sum / np.maximum(temporal_count, 1)
    static_mean = static_sum / np.maximum(static_count, 1)
    temporal_variance = temporal_sum_sq / np.maximum(temporal_count, 1) - np.square(temporal_mean)
    static_variance = static_sum_sq / np.maximum(static_count, 1) - np.square(static_mean)
    temporal_std = np.sqrt(np.maximum(temporal_variance, 0.0))
    static_std = np.sqrt(np.maximum(static_variance, 0.0))
    temporal_std[temporal_std < 1e-8] = 1.0
    static_std[static_std < 1e-8] = 1.0
    stats = {
        "method": "train_only_standardization",
        "temporal_mean": temporal_mean.tolist(),
        "temporal_std": temporal_std.tolist(),
        "static_mean": static_mean.tolist(),
        "static_std": static_std.tolist(),
        "temporal_count": temporal_count.tolist(),
        "static_count": static_count.tolist(),
        "training_rows_used": int(processed_rows),
        "temporal_features": TEMPORAL_COUNT,
    "missing_mask_features": TEMPORAL_COUNT,
    "lstm_input_features": LSTM_INPUT_SIZE,
        "static_features": STATIC_COUNT,
        "max_sequence_length": MAX_SEQ_LEN
    }
    with open(NORMALIZATION_PATH, "w") as normalization_file:
        json.dump(stats, normalization_file, indent=2)
    print(f"Normalization statistics saved: {NORMALIZATION_PATH}")
    print(f"Normalization rows processed: {processed_rows:,}")
    return stats
def load_normalization_statistics():
    with open(NORMALIZATION_PATH, "r") as normalization_file:
        stats = json.load(normalization_file)
    temporal_mean = np.asarray(stats["temporal_mean"], dtype=np.float32)
    temporal_std = np.asarray(stats["temporal_std"], dtype=np.float32)
    static_mean = np.asarray(stats["static_mean"], dtype=np.float32)
    static_std = np.asarray(stats["static_std"], dtype=np.float32)
    if len(temporal_mean) != TEMPORAL_COUNT or len(temporal_std) != TEMPORAL_COUNT:
        raise RuntimeError("Invalid temporal normalization dimension")
    if len(static_mean) != STATIC_COUNT or len(static_std) != STATIC_COUNT:
        raise RuntimeError("Invalid static normalization dimension")
    print("Normalization statistics validation passed")
    return temporal_mean, temporal_std, static_mean, static_std
class ParquetSequenceDataset(IterableDataset):
    def __init__(self, path, temporal_mean, temporal_std, static_mean, static_std, shuffle=False, seed=42):
        self.path = path
        self.temporal_mean = temporal_mean
        self.temporal_std = temporal_std
        self.static_mean = static_mean
        self.static_std = static_std
        self.shuffle = shuffle
        self.seed = seed
    def _read_arrow_batch(self, record_batch):
        temporal_flat = pc.list_flatten(pc.list_flatten(record_batch.column("temporal"))).to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        missing_flat = pc.list_flatten(pc.list_flatten(record_batch.column("missing_mask"))).to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        padding_flat = pc.list_flatten(record_batch.column("padding_mask")).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        static_flat = pc.list_flatten(record_batch.column("static")).to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        sequence_rows = record_batch.column("sequence_length").to_numpy(zero_copy_only=False)
        label_rows = record_batch.column("label").to_numpy(zero_copy_only=False)
        row_count = len(label_rows)
        temporal_rows = temporal_flat.reshape(row_count, MAX_SEQ_LEN, TEMPORAL_COUNT)
        missing_rows = missing_flat.reshape(row_count, MAX_SEQ_LEN, TEMPORAL_COUNT)
        padding_rows = padding_flat.reshape(row_count, MAX_SEQ_LEN)
        static_rows = static_flat.reshape(row_count, STATIC_COUNT)
        samples = []
        for i in range(row_count):
            temporal = temporal_rows[i].copy()
            missing_mask = missing_rows[i].copy()
            padding_mask = padding_rows[i].copy()
            static = static_rows[i].copy()
            temporal = ((temporal - self.temporal_mean) / self.temporal_std).astype(np.float32, copy=False)
            static = ((static - self.static_mean) / self.static_std).astype(np.float32, copy=False)
            temporal = np.nan_to_num(temporal, nan=0.0, posinf=0.0, neginf=0.0)
            static = np.nan_to_num(static, nan=0.0, posinf=0.0, neginf=0.0)
            sequence_length = int(sequence_rows[i])
            label = int(label_rows[i])
            samples.append((temporal, missing_mask.copy(), padding_mask, static, label))
        del temporal_flat
        del missing_flat
        del padding_flat
        del static_flat
        del sequence_rows
        del label_rows
        del temporal_rows
        del missing_rows
        del padding_rows
        del static_rows
        return samples
    def __iter__(self):
        dataset = ds.dataset(self.path, format="parquet")
        scanner = dataset.scanner(
            columns=["temporal", "missing_mask", "padding_mask", "static", "sequence_length", "label"],
            batch_size=ARROW_BATCH_SIZE,
            use_threads=False
        )
        rng = random.Random(self.seed)
        if self.shuffle:
            buffer = []
            for record_batch in scanner.to_batches():
                samples = self._read_arrow_batch(record_batch)
                for sample in samples:
                    if len(buffer) < SHUFFLE_BUFFER:
                        buffer.append(sample)
                    else:
                        index = rng.randrange(SHUFFLE_BUFFER)
                        output_sample = buffer[index]
                        buffer[index] = sample
                        yield output_sample
                del samples
                del record_batch
            rng.shuffle(buffer)
            for sample in buffer:
                yield sample
            del buffer
        else:
            for record_batch in scanner.to_batches():
                samples = self._read_arrow_batch(record_batch)
                for sample in samples:
                    yield sample
                del samples
                del record_batch
        gc.collect()
def collate_batch(batch):
    temporal = torch.from_numpy(np.stack([sample[0] for sample in batch]))
    missing_mask = torch.from_numpy(np.stack([sample[1] for sample in batch]))
    padding_mask = torch.from_numpy(np.stack([sample[2] for sample in batch]))
    static = torch.from_numpy(np.stack([sample[3] for sample in batch]))
    labels = torch.tensor([sample[4] for sample in batch], dtype=torch.float32)
    temporal_with_missing = torch.cat([temporal, missing_mask], dim=2)
    return temporal_with_missing, padding_mask, static, labels
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
        lengths = (padding_mask == 0).sum(dim=1).long()
        lengths = torch.clamp(lengths, min=1, max=MAX_SEQ_LEN)
        packed_sequence = nn.utils.rnn.pack_padded_sequence(
            temporal,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed_sequence)
        temporal_output = hidden[-1]
        static_output = self.static_network(static)
        fused_output = torch.cat([temporal_output, static_output], dim=1)
        logits = self.fusion(fused_output).squeeze(1)
        return logits
def create_loader(path, temporal_mean, temporal_std, static_mean, static_std, shuffle=False, seed=42):
    dataset = ParquetSequenceDataset(path, temporal_mean, temporal_std, static_mean, static_std, shuffle, seed)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate_batch,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    total_samples = 0
    processed_samples = 0
    for temporal, padding_mask, static, labels in loader:
        temporal = temporal.to(device, non_blocking=True)
        padding_mask = padding_mask.to(device, non_blocking=True)
        static = static.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(temporal, padding_mask, static)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()
        current_batch_size = labels.size(0)
        total_loss += loss.item() * current_batch_size
        total_samples += current_batch_size
        processed_samples += current_batch_size
        if processed_samples % 100000 < current_batch_size:
            print(f"Training samples processed: {processed_samples:,}")
        del temporal
        del padding_mask
        del static
        del labels
        del logits
        del loss
    if total_samples == 0:
        raise RuntimeError("Training dataset produced zero samples")
    return total_loss / total_samples
@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_probabilities = []
    all_labels = []
    for temporal, padding_mask, static, labels in loader:
        temporal = temporal.to(device, non_blocking=True)
        padding_mask = padding_mask.to(device, non_blocking=True)
        static = static.to(device, non_blocking=True)
        labels_device = labels.to(device, non_blocking=True)
        logits = model(temporal, padding_mask, static)
        loss = criterion(logits, labels_device)
        probabilities = torch.sigmoid(logits)
        current_batch_size = labels_device.size(0)
        total_loss += loss.item() * current_batch_size
        total_samples += current_batch_size
        all_probabilities.append(probabilities.cpu().numpy())
        all_labels.append(labels.numpy())
        del temporal
        del padding_mask
        del static
        del labels_device
        del logits
        del loss
        del probabilities
    if total_samples == 0:
        raise RuntimeError("Validation dataset produced zero samples")
    probabilities_array = np.concatenate(all_probabilities)
    labels_array = np.concatenate(all_labels)
    if len(np.unique(labels_array)) < 2:
        raise RuntimeError("Validation data contains only one class")
    average_loss = total_loss / total_samples
    roc_auc = roc_auc_score(labels_array, probabilities_array)
    pr_auc = average_precision_score(labels_array, probabilities_array)
    del all_probabilities
    del all_labels
    del probabilities_array
    del labels_array
    return average_loss, roc_auc, pr_auc
print("Checking local datasets")
if not os.path.exists(LOCAL_TRAIN_PATH):
    raise FileNotFoundError(f"Training dataset not found: {LOCAL_TRAIN_PATH}")
if not os.path.exists(LOCAL_VAL_PATH):
    raise FileNotFoundError(f"Validation dataset not found: {LOCAL_VAL_PATH}")
print("Local training and validation datasets found")
validate_dataset_schema(LOCAL_TRAIN_PATH)
validate_dataset_schema(LOCAL_VAL_PATH)
train_rows = count_parquet_rows(LOCAL_TRAIN_PATH)
val_rows = count_parquet_rows(LOCAL_VAL_PATH)
print(f"Training rows: {train_rows:,}")
print(f"Validation rows: {val_rows:,}")
if train_rows != 841619 or val_rows != 184926:
    raise RuntimeError("Local dataset row count verification failed")
print("Local dataset row count verification passed")
validate_dataset_samples(LOCAL_TRAIN_PATH)
validate_dataset_samples(LOCAL_VAL_PATH)

if not os.path.exists(NORMALIZATION_PATH):
    fit_normalization_statistics(LOCAL_TRAIN_PATH)
else:
    print(f"Using existing normalization statistics: {NORMALIZATION_PATH}")

temporal_mean, temporal_std, static_mean, static_std = load_normalization_statistics()

print("Checking training class distribution")
train_dataset = ds.dataset(LOCAL_TRAIN_PATH, format="parquet")
train_scanner = train_dataset.scanner(
    columns=["label"],
    batch_size=ARROW_BATCH_SIZE,
    use_threads=False
)
negative_count = 0
positive_count = 0
for record_batch in train_scanner.to_batches():
    labels = record_batch.column("label").to_numpy(zero_copy_only=False)
    negative_count += int(np.sum(labels == 0))
    positive_count += int(np.sum(labels == 1))
    del labels
    del record_batch
if negative_count == 0 or positive_count == 0:
    raise RuntimeError("Training data must contain both classes")
pos_weight_value = negative_count / positive_count
print(f"Training negatives: {negative_count}")
print(f"Training positives: {positive_count}")
print(f"Positive class weight: {pos_weight_value:.6f}")
print("Initializing LSTM model")
model = ICU_LSTM(
    LSTM_INPUT_SIZE,
    STATIC_COUNT,
    HIDDEN_SIZE,
    NUM_LAYERS,
    DROPOUT,
    STATIC_HIDDEN,
    FUSION_HIDDEN
).to(device)
pos_weight_tensor = torch.tensor(
    pos_weight_value,
    dtype=torch.float32,
    device=device
)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=1,
    min_lr=1e-6
)
parameter_count = sum(parameter.numel() for parameter in model.parameters())
trainable_parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
print(f"Total parameters: {parameter_count:,}")
print(f"Trainable parameters: {trainable_parameter_count:,}")
print(f"Maximum epochs: {EPOCHS}")
print(f"Early stopping patience: {PATIENCE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Arrow batch size: {ARROW_BATCH_SIZE}")
print(f"Shuffle buffer: {SHUFFLE_BUFFER}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Weight decay: {WEIGHT_DECAY}")
print(f"Dropout: {DROPOUT}")
print(f"Gradient clipping: {GRADIENT_CLIP}")
print(f"LSTM input features: {LSTM_INPUT_SIZE}")
print("Missing mask enabled")
print("Train-only normalization enabled")
best_val_pr_auc = -np.inf
best_val_roc_auc = -np.inf
best_epoch = 0
epochs_without_improvement = 0
history = []
print("Starting GPU LSTM training")
for epoch in range(1, EPOCHS + 1):
    print(f"Starting epoch {epoch}/{EPOCHS}")
    train_loader = create_loader(
        LOCAL_TRAIN_PATH,
        temporal_mean,
        temporal_std,
        static_mean,
        static_std,
        shuffle=True,
        seed=SEED + epoch
    )
    train_loss = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer
    )
    del train_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Training epoch {epoch} completed")
    val_loader = create_loader(
        LOCAL_VAL_PATH,
        temporal_mean,
        temporal_std,
        static_mean,
        static_std,
        shuffle=False,
        seed=SEED
    )
    val_loss, val_roc_auc, val_pr_auc = evaluate(
        model,
        val_loader,
        criterion
    )
    del val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    scheduler.step(val_pr_auc)
    current_lr = optimizer.param_groups[0]["lr"]
    epoch_result = {
        "epoch": epoch,
        "train_loss": float(train_loss),
        "validation_loss": float(val_loss),
        "validation_roc_auc": float(val_roc_auc),
        "validation_pr_auc": float(val_pr_auc),
        "learning_rate": float(current_lr)
    }
    history.append(epoch_result)
    print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val ROC AUC: {val_roc_auc:.6f} | Val PR AUC: {val_pr_auc:.6f} | LR: {current_lr:.8f}")
    if val_pr_auc > best_val_pr_auc:
        best_val_pr_auc = val_pr_auc
        best_val_roc_auc = val_roc_auc
        best_epoch = epoch
        epochs_without_improvement = 0
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "temporal_count": TEMPORAL_COUNT,
            "missing_mask_count": TEMPORAL_COUNT,
            "lstm_input_size": LSTM_INPUT_SIZE,
            "static_count": STATIC_COUNT,
            "expected_features": EXPECTED_FEATURES,
            "max_seq_len": MAX_SEQ_LEN,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "static_hidden": STATIC_HIDDEN,
            "fusion_hidden": FUSION_HIDDEN,
            "batch_size": BATCH_SIZE,
            "arrow_batch_size": ARROW_BATCH_SIZE,
            "shuffle_buffer": SHUFFLE_BUFFER,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip": GRADIENT_CLIP,
            "pos_weight": float(pos_weight_value),
            "train_negative_count": negative_count,
            "train_positive_count": positive_count,
            "best_epoch": best_epoch,
            "best_validation_pr_auc": float(best_val_pr_auc),
            "best_validation_roc_auc": float(best_val_roc_auc),
            "normalization_path": NORMALIZATION_PATH,
            "imputation_artifact_path": IMPUTATION_ARTIFACT_PATH,
            "normalization_method": "train_only_standardization",
            "missing_mask_enabled": True,
            "seed": SEED
        }
        torch.save(checkpoint, MODEL_PATH)
        del checkpoint
        print(f"Validation PR AUC improved; best model saved: {MODEL_PATH}")
    else:
        epochs_without_improvement += 1
        print(f"No validation PR AUC improvement: {epochs_without_improvement}/{PATIENCE}")
        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}")
            break
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
training_summary = {
    "status": "completed",
    "training_framework": "PyTorch",
    "training_device": str(device),
    "best_epoch": int(best_epoch),
    "best_validation_pr_auc": float(best_val_pr_auc),
    "best_validation_roc_auc": float(best_val_roc_auc),
    "maximum_epochs": EPOCHS,
    "epochs_completed": len(history),
    "early_stopping_patience": PATIENCE,
    "batch_size": BATCH_SIZE,
    "arrow_batch_size": ARROW_BATCH_SIZE,
    "shuffle_buffer": SHUFFLE_BUFFER,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "dropout": DROPOUT,
    "gradient_clip": GRADIENT_CLIP,
    "hidden_size": HIDDEN_SIZE,
    "num_layers": NUM_LAYERS,
    "static_hidden": STATIC_HIDDEN,
    "fusion_hidden": FUSION_HIDDEN,
    "temporal_features": TEMPORAL_COUNT,
    "missing_mask_features": TEMPORAL_COUNT,
    "lstm_input_features": LSTM_INPUT_SIZE,
    "static_features": STATIC_COUNT,
    "total_features": EXPECTED_FEATURES,
    "maximum_sequence_length": MAX_SEQ_LEN,
    "positive_class_weight": float(pos_weight_value),
    "train_negative_count": negative_count,
    "train_positive_count": positive_count,
    "training_rows": train_rows,
    "validation_rows": val_rows,
    "parameter_count": parameter_count,
    "trainable_parameter_count": trainable_parameter_count,
    "model_path": MODEL_PATH,
    "normalization_path": NORMALIZATION_PATH,
    "imputation_artifact_path": IMPUTATION_ARTIFACT_PATH,
    "missing_mask_enabled": True,
    "normalization_method": "train_only_standardization",
    "history": history
}
with open(HISTORY_PATH, "w") as history_file:
    json.dump(training_summary, history_file, indent=2)
print(f"Training history saved: {HISTORY_PATH}")
print(f"Training completed after {len(history)} epoch(s)")
print(f"Best epoch: {best_epoch}")
print(f"Best validation ROC AUC: {best_val_roc_auc:.6f}")
print(f"Best validation PR AUC: {best_val_pr_auc:.6f}")
print(f"Best model: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    raise RuntimeError("Best LSTM checkpoint was not created")
print("LSTM training checkpoint verification passed")