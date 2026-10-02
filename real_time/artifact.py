import json
import os
import numpy as np
import torch
import xgboost as xgb
import config as C
ARTIFACT_DIR = C.ARTIFACT_DIR
def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return float(out) if out.ndim == 0 else out
def logit(p):
    p = np.clip(float(p), 1e-7, 1.0 - 1e-7)
    return np.log(p / (1.0 - p))
def platt_probability(raw_probability, intercept, coefficient):
    return float(sigmoid(intercept + coefficient * logit(raw_probability)))
class ICU_LSTM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        dropout = C.DROPOUT if C.LSTM_NUM_LAYERS > 1 else 0.0
        self.lstm = torch.nn.LSTM(input_size=C.LSTM_INPUT_SIZE, hidden_size=C.LSTM_HIDDEN_SIZE, num_layers=C.LSTM_NUM_LAYERS, batch_first=True, dropout=dropout)
        self.static_network = torch.nn.Sequential(torch.nn.Linear(C.STATIC_FEATURES, C.LSTM_STATIC_HIDDEN), torch.nn.ReLU(), torch.nn.Dropout(C.DROPOUT))
        self.fusion = torch.nn.Sequential(torch.nn.Linear(C.LSTM_HIDDEN_SIZE + C.LSTM_STATIC_HIDDEN, C.LSTM_FUSION_HIDDEN), torch.nn.ReLU(), torch.nn.Dropout(C.DROPOUT), torch.nn.Linear(C.LSTM_FUSION_HIDDEN, 1))
    def forward(self, x, padding_mask, static):
        lengths = (padding_mask.round() == 0).sum(dim=1).to(torch.long).cpu().clamp(min=1, max=C.MAX_SEQ_LEN)
        packed = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        temporal_output = hidden[-1]
        static_output = self.static_network(static)
        return self.fusion(torch.cat([temporal_output, static_output], dim=1))
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)
def validate_dimensions():
    if C.TEMPORAL_FEATURES + C.STATIC_FEATURES != C.FEATURE_COUNT:
        raise RuntimeError("Temporal and static feature counts do not equal 137")
    if C.LSTM_INPUT_SIZE != C.TEMPORAL_FEATURES * 2:
        raise RuntimeError("LSTM input size does not equal 108")
    if C.STATIC_FEATURE_START != C.TEMPORAL_FEATURES or C.STATIC_FEATURE_END != C.FEATURE_COUNT:
        raise RuntimeError("Static feature range does not match configuration")
def load_feature_names():
    data = load_json(os.path.join(ARTIFACT_DIR, "feature_names.json"))
    names = data.get("feature_names")
    if not isinstance(names, list) or len(names) != C.FEATURE_COUNT:
        raise RuntimeError("Feature names artifact does not contain 137 features")
    if len(set(names)) != len(names):
        raise RuntimeError("Feature names contain duplicates")
    if names[:C.TEMPORAL_FEATURES] != list(C.TEMPORAL_FEATURE_NAMES):
        raise RuntimeError("Feature name order does not match configuration")
    return names
def load_xgb(feature_names):
    if not os.path.isfile(C.XGB_MODEL_PATH):
        raise RuntimeError(f"XGBoost model not found: {C.XGB_MODEL_PATH}")
    booster = xgb.Booster()
    booster.load_model(C.XGB_MODEL_PATH)
    if booster.num_features() != len(feature_names):
        raise RuntimeError("XGBoost feature count does not match feature_names.json")
    if len(feature_names) != C.FEATURE_COUNT:
        raise RuntimeError("feature_names.json does not contain the expected feature count")
    if len(set(feature_names)) != len(feature_names):
        raise RuntimeError("feature_names.json contains duplicate feature names")
    return booster
def load_lstm():
    if not os.path.isfile(C.LSTM_MODEL_PATH):
        raise RuntimeError(f"LSTM checkpoint not found: {C.LSTM_MODEL_PATH}")
    checkpoint = torch.load(C.LSTM_MODEL_PATH, map_location=C.DEVICE, weights_only=True)
    if "model_state_dict" not in checkpoint:
        raise RuntimeError("LSTM checkpoint does not contain model_state_dict")
    expected = {
        "temporal_count": C.TEMPORAL_FEATURES,
        "missing_mask_count": C.TEMPORAL_FEATURES,
        "lstm_input_size": C.LSTM_INPUT_SIZE,
        "static_count": C.STATIC_FEATURES,
        "expected_features": C.FEATURE_COUNT,
        "max_seq_len": C.MAX_SEQ_LEN,
        "hidden_size": C.LSTM_HIDDEN_SIZE,
        "num_layers": C.LSTM_NUM_LAYERS,
        "dropout": C.DROPOUT,
        "static_hidden": C.LSTM_STATIC_HIDDEN,
        "fusion_hidden": C.LSTM_FUSION_HIDDEN,
        "missing_mask_enabled": True
    }
    for key, value in expected.items():
        if key not in checkpoint or checkpoint[key] != value:
            raise RuntimeError(f"LSTM checkpoint metadata mismatch: {key}")
    model = ICU_LSTM()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(C.DEVICE)
    model.eval()
    return model
def load_imputation():
    data = load_json(os.path.join(ARTIFACT_DIR, "imputation_values.json"))
    values = data.get("values")
    if not isinstance(values, dict) or set(values) != set(C.TEMPORAL_FEATURE_NAMES):
        raise RuntimeError("Imputation artifact does not match temporal features")
    return {name: float(values[name]) for name in C.TEMPORAL_FEATURE_NAMES}
def load_normalization():
    data = load_json(os.path.join(ARTIFACT_DIR, "normalization.json"))
    output = {
        "temporal_mean": np.asarray(data["temporal_mean"], dtype=np.float32),
        "temporal_std": np.asarray(data["temporal_std"], dtype=np.float32),
        "static_mean": np.asarray(data["static_mean"], dtype=np.float32),
        "static_std": np.asarray(data["static_std"], dtype=np.float32)
    }
    if output["temporal_mean"].shape != (C.TEMPORAL_FEATURES,) or output["temporal_std"].shape != (C.TEMPORAL_FEATURES,):
        raise RuntimeError("Temporal normalization shape mismatch")
    if output["static_mean"].shape != (C.STATIC_FEATURES,) or output["static_std"].shape != (C.STATIC_FEATURES,):
        raise RuntimeError("Static normalization shape mismatch")
    if np.any(output["temporal_std"] <= 0) or np.any(output["static_std"] <= 0):
        raise RuntimeError("Normalization contains non-positive standard deviation")
    return output
def load_calibration():
    data = load_json(os.path.join(ARTIFACT_DIR, "calibration.json"))
    output = {key: float(data[key]) for key in ["xgb_intercept", "xgb_coefficient", "lstm_intercept", "lstm_coefficient"]}
    expected = {
        "xgb_intercept": C.XGB_PLATT_INTERCEPT,
        "xgb_coefficient": C.XGB_PLATT_COEFFICIENT,
        "lstm_intercept": C.LSTM_PLATT_INTERCEPT,
        "lstm_coefficient": C.LSTM_PLATT_COEFFICIENT
    }
    for key, value in expected.items():
        if not np.isclose(output[key], value, rtol=0.0, atol=1e-12):
            raise RuntimeError(f"Calibration parameter does not match configuration: {key}")
    return output
def load_ensemble():
    data = load_json(os.path.join(ARTIFACT_DIR, "ensemble_config.json"))
    output = {
        "threshold": float(data["threshold"]),
        "xgb_weight": float(data["xgb_weight"]),
        "lstm_weight": float(data["lstm_weight"]),
        "test_used": data.get("test_used")
    }
    if output["test_used"] is not False:
        raise RuntimeError("Locked ensemble artifact must have test_used=False")
    if abs(output["xgb_weight"] + output["lstm_weight"] - 1.0) > 1e-12:
        raise RuntimeError("Ensemble weights do not sum to 1")
    if not np.isclose(output["xgb_weight"], C.XGB_WEIGHT, rtol=0.0, atol=1e-12):
        raise RuntimeError("XGBoost ensemble weight does not match configuration")
    if not np.isclose(output["lstm_weight"], C.LSTM_WEIGHT, rtol=0.0, atol=1e-12):
        raise RuntimeError("LSTM ensemble weight does not match configuration")
    if not np.isclose(output["threshold"], C.ENSEMBLE_THRESHOLD, rtol=0.0, atol=1e-9):
        raise RuntimeError("Ensemble threshold does not match configuration")
    return output
def load_all():
    validate_dimensions()
    feature_names = load_feature_names()
    normalization = load_normalization()
    calibration = load_calibration()
    ensemble = load_ensemble()
    imputation = load_imputation()
    xgb_model = load_xgb(feature_names)
    lstm_model = load_lstm()
    return {
        "feature_names": feature_names,
        "imputation": imputation,
        "normalization": normalization,
        "calibration": calibration,
        "ensemble": ensemble,
        "xgb_model": xgb_model,
        "lstm_model": lstm_model
    }
def extract_vector(value):
    if hasattr(value, "toArray"):
        return np.asarray(value.toArray(), dtype=np.float32)
    return np.asarray(value, dtype=np.float32)
def validate_sequence_structure(temporal, missing_mask, padding_mask):
    if temporal.ndim != 2 or temporal.shape[1] != C.TEMPORAL_FEATURES:
        raise RuntimeError("Temporal input must contain 54 features")
    if missing_mask.shape != temporal.shape:
        raise RuntimeError("Missing mask shape mismatch")
    if len(padding_mask) != len(temporal):
        raise RuntimeError("Padding mask length mismatch")
    sequence_length = int((padding_mask.round() == 0).sum())
    if sequence_length <= 0 or sequence_length > C.MAX_SEQ_LEN:
        raise RuntimeError("Invalid sequence length")
    if np.any(padding_mask[:sequence_length].round() != 0) or np.any(padding_mask[sequence_length:].round() != 1):
        raise RuntimeError("Padding mask structure is invalid")
    return sequence_length
def validate_input_parity(xgb_features, temporal, static, sequence_length, feature_names):
    if xgb_features.shape != (C.FEATURE_COUNT,) or static.shape != (C.STATIC_FEATURES,):
        raise RuntimeError("Feature dimensions do not match configuration")
    if not np.allclose(static, xgb_features[C.STATIC_FEATURE_START:C.STATIC_FEATURE_END], rtol=0.0, atol=1e-6, equal_nan=True):
        raise RuntimeError("LSTM static features do not match XGBoost static features")
    for name in [name for name in C.TEMPORAL_FEATURE_NAMES if name.endswith("_last")]:
        temporal_idx = C.TEMPORAL_FEATURE_NAMES.index(name)
        xgb_idx = feature_names.index(name)
        if np.isnan(xgb_features[xgb_idx]):
            continue
        if not np.isclose(temporal[sequence_length - 1, temporal_idx], xgb_features[xgb_idx], rtol=0.0, atol=1e-6, equal_nan=True):
            raise RuntimeError(f"Temporal/XGBoost parity failed for {name}")
def score_xgb(booster, features, feature_names):
    matrix = xgb.DMatrix(np.asarray(features, dtype=np.float32).reshape(1, -1), missing=np.nan)
    probability = booster.predict(matrix, iteration_range=(0, C.XGB_BEST_ITERATION + 1))
    return float(probability[0])
def score_xgb_with_margin(booster, features, feature_names):
    matrix = xgb.DMatrix(np.asarray(features, dtype=np.float32).reshape(1, -1), missing=np.nan)
    margin = booster.predict(matrix, output_margin=True, iteration_range=(0, C.XGB_BEST_ITERATION + 1))
    probability = booster.predict(matrix, output_margin=False, iteration_range=(0, C.XGB_BEST_ITERATION + 1))
    return float(probability[0]), float(margin[0])
def score_lstm(model, temporal, missing_mask, padding_mask, static, normalization):
    temporal = np.array(temporal, dtype=np.float32)
    missing_mask = np.array(missing_mask, dtype=np.float32)
    padding_mask = np.array(padding_mask, dtype=np.float32)
    static = np.array(static, dtype=np.float32)
    sequence_length = int((padding_mask.round() == 0).sum())
    temporal = (temporal - normalization["temporal_mean"]) / normalization["temporal_std"]
    static = (static - normalization["static_mean"]) / normalization["static_std"]
    if not np.isfinite(temporal[:sequence_length]).all() or not np.isfinite(static).all():
        raise RuntimeError("LSTM input contains invalid values after normalization")
    temporal[sequence_length:] = 0.0
    missing_mask[sequence_length:] = 0.0
    x = np.concatenate([temporal, missing_mask], axis=1).astype(np.float32)
    x_tensor = torch.from_numpy(x).unsqueeze(0).to(C.DEVICE)
    static_tensor = torch.from_numpy(static).unsqueeze(0).to(C.DEVICE)
    with torch.inference_mode():
        lstm_output, _ = model.lstm(x_tensor)
        temporal_output = lstm_output[:, sequence_length - 1, :]
        static_output = model.static_network(static_tensor)
        fused_output = torch.cat([temporal_output, static_output], dim=1)
        logit = float(model.fusion(fused_output).squeeze(1).item())
    return logit, float(sigmoid(logit))