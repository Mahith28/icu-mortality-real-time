import logging
import math
import threading
import time
from collections import OrderedDict
from datetime import datetime, date
import numpy as np
import shap
import xgboost as xgb
import config as C
import artifact
logger = logging.getLogger(__name__)
class RealtimeSHAP:
    def __init__(self, model, feature_names=None, top_k=20, max_gap_windows=6, state_ttl_seconds=21600, max_stays=10000):
        self.model = model
        self.feature_names = list(feature_names if feature_names is not None else C.FEATURE_NAMES)
        self.expected_features = int(C.FEATURE_COUNT)
        self.top_k = int(top_k)
        self.max_gap_windows = int(max_gap_windows)
        self.state_ttl_seconds = float(state_ttl_seconds)
        self.max_stays = int(max_stays)
        self.tree_limit = int(C.XGB_BEST_ITERATION) + 1
        self.previous_state = OrderedDict()
        self.lock = threading.RLock()
        self.last_cleanup = time.monotonic()
        self.missing_margin_count = 0
        if self.expected_features <= 0:
            raise ValueError("FEATURE_COUNT must be greater than 0")
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if self.max_gap_windows < 0:
            raise ValueError("max_gap_windows must be >= 0")
        if self.state_ttl_seconds <= 0:
            raise ValueError("state_ttl_seconds must be greater than 0")
        if self.max_stays <= 0:
            raise ValueError("max_stays must be greater than 0")
        if len(self.feature_names) != self.expected_features:
            raise ValueError(f"Feature count mismatch: expected {self.expected_features}, got {len(self.feature_names)}")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("Feature names must be unique")
        self._validate_model()
        rounds = int(self.model.num_boosted_rounds())
        shap_model = self.model if self.tree_limit == rounds else self.model[:self.tree_limit]
        if int(shap_model.num_boosted_rounds()) != self.tree_limit:
            raise RuntimeError("Limited model has unexpected round count")
        self.shap_model = shap_model
        self.explainer = shap.TreeExplainer(self.shap_model, feature_perturbation="tree_path_dependent", model_output="raw")
        self.expected_value = self._get_expected_value()
        if not math.isfinite(self.expected_value):
            raise ValueError(f"SHAP expected value is not finite: {self.expected_value}")
    def _validate_model(self):
        if not isinstance(self.model, xgb.Booster):
            raise TypeError(f"Expected xgboost.Booster, got {type(self.model).__name__}")
        model_feature_count = int(self.model.num_features())
        if model_feature_count != self.expected_features:
            raise ValueError(f"Model feature count mismatch: expected {self.expected_features}, got {model_feature_count}")
        model_names = self.model.feature_names
        if model_names is not None:
            if len(model_names) != self.expected_features:
                raise ValueError(f"Model feature-name count mismatch: expected {self.expected_features}, got {len(model_names)}")
            if list(model_names) != self.feature_names:
                raise ValueError("Model feature names/order do not match the realtime feature contract")
        boosted_rounds = int(self.model.num_boosted_rounds())
        if self.tree_limit <= 0:
            raise ValueError(f"Invalid tree_limit: {self.tree_limit}")
        if self.tree_limit > boosted_rounds:
            raise ValueError(f"Configured tree_limit {self.tree_limit} exceeds model boosted rounds {boosted_rounds}")
        model_best_iteration = getattr(self.model, "best_iteration", None)
        if model_best_iteration is not None:
            if int(model_best_iteration) != int(C.XGB_BEST_ITERATION):
                raise ValueError(f"XGB best iteration mismatch: config={C.XGB_BEST_ITERATION}, model={model_best_iteration}")
    def _get_expected_value(self):
        value = self.explainer.expected_value
        if isinstance(value, np.ndarray):
            value = value.reshape(-1)[0]
        return float(value)
    def _validate_features(self, features):
        features = np.asarray(features, dtype=np.float32)
        if features.ndim == 2:
            if features.shape[0] != 1:
                raise ValueError(f"Expected one feature row, got shape {features.shape}")
            features = features[0]
        if features.ndim != 1:
            raise ValueError(f"Expected 1D feature vector, got shape {features.shape}")
        if len(features) != self.expected_features:
            raise ValueError(f"Feature count mismatch: expected {self.expected_features}, got {len(features)}")
        if np.any(np.isinf(features)):
            raise ValueError("Feature vector contains infinite values")
        return features
    def _validate_probability(self, value, name):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
        return value
    def _normalize_stay_id(self, stay_id):
        if stay_id is None:
            raise ValueError("stay_id cannot be None")
        return str(stay_id)
    def _normalize_window_id(self, window_id):
        if window_id is None:
            raise ValueError("window_id cannot be None")
        if isinstance(window_id, (bool, np.bool_)):
            raise ValueError("window_id cannot be boolean")
        if isinstance(window_id, (float, np.floating)):
            if not np.isfinite(window_id) or not float(window_id).is_integer():
                raise ValueError(f"window_id must be an integer, got {window_id!r}")
            return int(window_id)
        try:
            value = int(window_id)
        except (TypeError, ValueError):
            raise ValueError(f"window_id must be numeric, got {window_id!r}")
        return value
    def _normalize_timestamp(self, timestamp):
        if timestamp is None:
            return None
        if isinstance(timestamp, np.datetime64):
            return str(timestamp)
        if isinstance(timestamp, (datetime, date)):
            return timestamp.isoformat()
        if isinstance(timestamp, np.integer):
            return int(timestamp)
        if isinstance(timestamp, np.floating):
            return float(timestamp)
        if isinstance(timestamp, (int, float, str)):
            return timestamp
        if hasattr(timestamp, "isoformat"):
            return timestamp.isoformat()
        return str(timestamp)
    def _sigmoid(self, value):
        value = float(value)
        if value >= 0:
            z = math.exp(-value)
            return 1.0 / (1.0 + z)
        z = math.exp(value)
        return z / (1.0 + z)
    def _predict_probability(self, features):
        matrix = xgb.DMatrix(features.reshape(1, -1), missing=np.nan, feature_names=self.feature_names)
        prediction = self.shap_model.predict(matrix, output_margin=False)
        return float(np.asarray(prediction).reshape(-1)[0])
    def _predict_margin(self, features):
        matrix = xgb.DMatrix(features.reshape(1, -1), missing=np.nan, feature_names=self.feature_names)
        prediction = self.shap_model.predict(matrix, output_margin=True)
        return float(np.asarray(prediction).reshape(-1)[0])
    def _predict_reference_margin(self, features):
        matrix = xgb.DMatrix(features.reshape(1, -1), missing=np.nan, feature_names=self.feature_names)
        prediction = self.model.predict(matrix, output_margin=True, iteration_range=(0, self.tree_limit))
        return float(np.asarray(prediction).reshape(-1)[0])
    def _is_newer_window(self, previous_window_id, current_window_id):
        if previous_window_id is None:
            return True
        previous_window_id = self._normalize_window_id(previous_window_id)
        current_window_id = self._normalize_window_id(current_window_id)
        return current_window_id > previous_window_id
    def _values_differ(self, previous_value, current_value, tolerance=1e-6):
        previous_nan = previous_value is None or np.isnan(previous_value)
        current_nan = current_value is None or np.isnan(current_value)
        if previous_nan and current_nan:
            return False
        if previous_nan or current_nan:
            return True
        return abs(float(current_value) - float(previous_value)) > tolerance
    def _cleanup_state(self):
        current_time = time.monotonic()
        if current_time - self.last_cleanup < 60 and len(self.previous_state) <= self.max_stays:
            return
        expired = [
            stay_id
            for stay_id, state in self.previous_state.items()
            if current_time - state["updated_at"] > self.state_ttl_seconds
        ]
        for stay_id in expired:
            del self.previous_state[stay_id]
        while len(self.previous_state) > self.max_stays:
            self.previous_state.popitem(last=False)
        self.last_cleanup = current_time
    def _top_features(self, features, shap_values):
        absolute_values = np.abs(shap_values)
        order = np.argsort(-absolute_values, kind="stable")[:self.top_k]
        return [
            {
                "feature": self.feature_names[int(index)],
                "value": None if np.isnan(features[int(index)]) else float(features[int(index)]),
                "shap_value": float(shap_values[int(index)]),
                "abs_shap_value": float(absolute_values[int(index)])
            }
            for index in order
        ]
    def _changed_features(self, previous_features, previous_shap_values, current_features, current_shap_values):
        if previous_features is None or previous_shap_values is None:
            return []
        shap_delta = current_shap_values - previous_shap_values
        changed_indices = np.where(np.abs(shap_delta) > 1e-6)[0]
        if len(changed_indices) == 0:
            return []
        order = changed_indices[np.argsort(-np.abs(shap_delta[changed_indices]), kind="stable")[:self.top_k]]
        changed = []
        for index in order:
            index = int(index)
            previous_nan = previous_features[index] is None or np.isnan(previous_features[index])
            current_nan = current_features[index] is None or np.isnan(current_features[index])
            value_changed = self._values_differ(previous_features[index], current_features[index])
            changed.append(
                {
                    "feature": self.feature_names[index],
                    "previous_value": None if previous_nan else float(previous_features[index]),
                    "current_value": None if current_nan else float(current_features[index]),
                    "previous_shap_value": float(previous_shap_values[index]),
                    "current_shap_value": float(current_shap_values[index]),
                    "value_delta": None if previous_nan or current_nan else float(current_features[index] - previous_features[index]),
                    "shap_delta": float(shap_delta[index]),
                    "value_changed": bool(value_changed)
                }
            )
        return changed
    def explain(self, features, xgb_raw_probability=None, xgb_raw_margin=None, verify=True):
        features = self._validate_features(features)
        shap_values = self.explainer.shap_values(features.reshape(1, -1), check_additivity=False)
        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 2:
            shap_values = shap_values[0]
        if shap_values.shape != (self.expected_features,):
            raise RuntimeError(f"Unexpected SHAP shape: expected {(self.expected_features,)}, got {shap_values.shape}")
        if not np.all(np.isfinite(shap_values)):
            raise RuntimeError("SHAP values contain NaN or infinite values")
        reconstructed_margin = self.expected_value + float(np.sum(shap_values))
        reconstructed_probability = self._sigmoid(reconstructed_margin)
        margin_error = None
        probability_error = None
        if verify:
            if xgb_raw_margin is None:
                self.missing_margin_count += 1
                logger.warning("xgb_raw_margin missing; SHAP margin verification skipped")
            if xgb_raw_margin is not None:
                xgb_raw_margin = float(xgb_raw_margin)
                if not math.isfinite(xgb_raw_margin):
                    raise ValueError("xgb_raw_margin must be finite")
                margin_error = abs(reconstructed_margin - xgb_raw_margin)
                if margin_error > 1e-3:
                    raise RuntimeError(f"SHAP margin additivity check failed: error={margin_error:.8g}")
            if xgb_raw_probability is not None:
                xgb_raw_probability = self._validate_probability(xgb_raw_probability, "xgb_raw_probability")
                probability_error = abs(reconstructed_probability - xgb_raw_probability)
                if probability_error > 1e-4:
                    raise RuntimeError(f"SHAP probability check failed: error={probability_error:.8g}")
            if xgb_raw_margin is None and xgb_raw_probability is None:
                logger.warning("SHAP verification skipped because no inference XGBoost output was supplied")
        return {
            "expected_value": float(self.expected_value),
            "shap_values": shap_values.astype(float).tolist(),
            "top_features": self._top_features(features, shap_values),
            "reconstructed_margin": float(reconstructed_margin),
            "reconstructed_probability": float(reconstructed_probability),
            "margin_error": None if margin_error is None else float(margin_error),
            "probability_error": None if probability_error is None else float(probability_error)
        }
    def update(self, stay_id, window_id, timestamp, features, inference_result, window_status="LIVE"):
        """
        Ordering violations and invalid input raise ValueError/TypeError/KeyError.
        SHAP runtime/additivity failures return status="error" with the risk result preserved.
        After a SHAP failure, state is not advanced, so the next successful window compares against the last successful window.
        """
        if not isinstance(inference_result, dict):
            raise TypeError("inference_result must be a dictionary")
        required_fields = ["xgb_raw_probability", "xgb_probability", "lstm_probability", "ensemble_probability"]
        missing_fields = [field for field in required_fields if field not in inference_result]
        if missing_fields:
            raise KeyError(f"Missing inference fields: {missing_fields}")
        stay_key = self._normalize_stay_id(stay_id)
        original_window_id = window_id
        current_window_id = self._normalize_window_id(window_id)
        timestamp = self._normalize_timestamp(timestamp)
        features = self._validate_features(features)
        xgb_raw_probability = self._validate_probability(inference_result["xgb_raw_probability"], "xgb_raw_probability")
        xgb_probability = self._validate_probability(inference_result["xgb_probability"], "xgb_probability")
        lstm_probability = self._validate_probability(inference_result["lstm_probability"], "lstm_probability")
        ensemble_probability = self._validate_probability(inference_result["ensemble_probability"], "ensemble_probability")
        xgb_raw_margin = inference_result.get("xgb_raw_margin")
        if xgb_raw_margin is not None:
            xgb_raw_margin = float(xgb_raw_margin)
            if not math.isfinite(xgb_raw_margin):
                raise ValueError("xgb_raw_margin must be finite")
        with self.lock:
            self._cleanup_state()
            previous = self.previous_state.get(stay_key)
            previous_window_id = None if previous is None else previous["window_id"]
            previous_window_id_original = None if previous is None else previous["window_id_original"]
            if previous_window_id is not None:
                if not self._is_newer_window(previous_window_id, current_window_id):
                    raise ValueError(f"Window ID is not newer: previous={previous_window_id}, current={current_window_id}")
            window_gap = None
            previous_discarded = False
            if previous_window_id is not None:
                window_gap = current_window_id - previous_window_id
                if window_gap > self.max_gap_windows:
                    previous_discarded = True
            try:
                explanation = self.explain(features, xgb_raw_probability=xgb_raw_probability, xgb_raw_margin=xgb_raw_margin, verify=True)
            except RuntimeError as exc:
                logger.exception("Realtime SHAP failed for stay_id=%s window_id=%s", stay_key, original_window_id)
                result = {
                    "status": "error",
                    "stay_id": stay_key,
                    "window_id": original_window_id,
                    "timestamp": timestamp,
                    "window_status": window_status,
                    "previous_window_id": previous_window_id_original,
                    "window_gap": window_gap,
                    "previous_discarded": previous_discarded,
                    "xgb_raw_probability": xgb_raw_probability,
                    "xgb_raw_margin": xgb_raw_margin,
                    "xgb_probability": xgb_probability,
                    "lstm_probability": lstm_probability,
                    "ensemble_probability": ensemble_probability,
                    "error": str(exc)
                }
                return self.validate_json_safe(result)
            current_shap_values = np.asarray(explanation["shap_values"], dtype=np.float64)
            previous_features = None if previous is None or previous_discarded else previous["features"]
            previous_shap_values = None if previous is None or previous_discarded else previous["shap_values"]
            changed_features = self._changed_features(previous_features, previous_shap_values, features, current_shap_values)
            previous_xgb_probability = None if previous is None or previous_discarded else previous["xgb_probability"]
            previous_lstm_probability = None if previous is None or previous_discarded else previous["lstm_probability"]
            previous_ensemble_probability = None if previous is None or previous_discarded else previous["ensemble_probability"]
            result = {
                "status": "ok",
                "stay_id": stay_key,
                "window_id": original_window_id,
                "timestamp": timestamp,
                "window_status": window_status,
                "previous_window_id": previous_window_id_original,
                "window_gap": window_gap,
                "previous_discarded": previous_discarded,
                "xgb_raw_probability": xgb_raw_probability,
                "xgb_raw_margin": xgb_raw_margin,
                "xgb_probability": xgb_probability,
                "lstm_probability": lstm_probability,
                "ensemble_probability": ensemble_probability,
                "xgb_probability_delta": None if previous_xgb_probability is None else float(xgb_probability - previous_xgb_probability),
                "lstm_probability_delta": None if previous_lstm_probability is None else float(lstm_probability - previous_lstm_probability),
                "ensemble_probability_delta": None if previous_ensemble_probability is None else float(ensemble_probability - previous_ensemble_probability),
                "shap": explanation,
                "changed_features": changed_features
            }
            self.previous_state[stay_key] = {
                "window_id": current_window_id,
                "window_id_original": original_window_id,
                "timestamp": timestamp,
                "window_status": window_status,
                "features": features.copy(),
                "shap_values": current_shap_values.copy(),
                "xgb_probability": xgb_probability,
                "lstm_probability": lstm_probability,
                "ensemble_probability": ensemble_probability,
                "updated_at": time.monotonic()
            }
            self.previous_state.move_to_end(stay_key)
            while len(self.previous_state) > self.max_stays:
                self.previous_state.popitem(last=False)
            return self.validate_json_safe(result)
    def reset(self, stay_id):
        stay_key = self._normalize_stay_id(stay_id)
        with self.lock:
            self.previous_state.pop(stay_key, None)
    def clear(self):
        with self.lock:
            self.previous_state.clear()
            self.last_cleanup = time.monotonic()
    def get_previous(self, stay_id):
        stay_key = self._normalize_stay_id(stay_id)
        with self.lock:
            state = self.previous_state.get(stay_key)
            if state is None:
                return None
            return {
                "window_id": state["window_id"],
                "window_id_original": state["window_id_original"],
                "timestamp": state["timestamp"],
                "window_status": state["window_status"],
                "features": state["features"].copy(),
                "shap_values": state["shap_values"].copy(),
                "xgb_probability": state["xgb_probability"],
                "lstm_probability": state["lstm_probability"],
                "ensemble_probability": state["ensemble_probability"],
                "updated_at": state["updated_at"]
            }
    def __len__(self):
        with self.lock:
            return len(self.previous_state)
    def validate_json_safe(self, value):
        if isinstance(value, dict):
            return {str(key): self.validate_json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.validate_json_safe(item) for item in value]
        if isinstance(value, tuple):
            return [self.validate_json_safe(item) for item in value]
        if isinstance(value, np.ndarray):
            return self.validate_json_safe(value.tolist())
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            value = float(value)
            if math.isfinite(value):
                return value
            logger.warning("Replacing non-finite numpy value with None")
            return None
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, np.datetime64):
            return str(value)
        if isinstance(value, float):
            if math.isfinite(value):
                return value
            logger.warning("Replacing non-finite float value with None")
            return None
        return value
    def startup_self_test(self, test_rows=None):
        if test_rows is None:
            zero_row = np.zeros(self.expected_features, dtype=np.float32)
            nan_row = np.full(self.expected_features, np.nan, dtype=np.float32)
            test_rows = [zero_row, nan_row]
        for index, features in enumerate(test_rows):
            features = self._validate_features(features)
            probability = self._predict_probability(features)
            margin = self._predict_margin(features)
            reference_margin = self._predict_reference_margin(features)
            if abs(reference_margin - margin) > 1e-4:
                raise RuntimeError(f"Sliced model disagrees with iteration_range inference on row {index}")
            explanation = self.explain(features, xgb_raw_probability=probability, xgb_raw_margin=margin, verify=True)
            if len(explanation["shap_values"]) != self.expected_features:
                raise RuntimeError(f"SHAP self-test failed for row {index}: expected {self.expected_features} values")
            if not np.all(np.isfinite(np.asarray(explanation["shap_values"]))):
                raise RuntimeError(f"SHAP self-test failed for row {index}: non-finite SHAP values")
            if not math.isfinite(explanation["reconstructed_probability"]):
                raise RuntimeError(f"SHAP self-test failed for row {index}: non-finite probability")
        return {
            "status": "ok",
            "rows_tested": len(test_rows),
            "feature_count": self.expected_features,
            "tree_limit": self.tree_limit,
            "model_boosted_rounds": int(self.model.num_boosted_rounds()),
            "expected_value": self.expected_value,
            "missing_margin_count": self.missing_margin_count
        }
class RealtimeInference:
    def __init__(self):
        self.artifacts = None
        self.xgb_model = None
        self.lstm_model = None
        self.feature_names = None
        self.normalization = None
        self.calibration = None
        self.ensemble = None
        self.imputation = None
        self.started = False
    def start(self):
        self.artifacts = artifact.load_all()
        self.xgb_model = self.artifacts["xgb_model"]
        self.lstm_model = self.artifacts["lstm_model"]
        self.feature_names = self.artifacts["feature_names"]
        self.normalization = self.artifacts["normalization"]
        self.calibration = self.artifacts["calibration"]
        self.ensemble = self.artifacts["ensemble"]
        self.imputation = self.artifacts["imputation"]
        self.lstm_model.eval()
        self.started = True
        return {
            "status": "ok",
            "feature_count": len(self.feature_names),
            "xgb_weight": float(self.ensemble["xgb_weight"]),
            "lstm_weight": float(self.ensemble["lstm_weight"]),
            "threshold": float(self.ensemble["threshold"])
        }
    def _require_started(self):
        if not self.started:
            raise RuntimeError("RealtimeInference.start() must be called before predict()")
    def predict(
        self,
        features,
        temporal,
        missing_mask,
        padding_mask,
        static
    ):
        self._require_started()
        xgb_features = np.asarray(features, dtype=np.float32).reshape(-1)
        temporal = np.asarray(temporal, dtype=np.float32)
        missing_mask = np.asarray(missing_mask, dtype=np.float32)
        padding_mask = np.asarray(padding_mask, dtype=np.float32).reshape(-1)
        static = np.asarray(static, dtype=np.float32).reshape(-1)
        sequence_length = artifact.validate_sequence_structure(
            temporal,
            missing_mask,
            padding_mask
        )
        artifact.validate_input_parity(
            xgb_features,
            temporal,
            static,
            sequence_length,
            self.feature_names
        )
        xgb_raw_probability, xgb_raw_margin = artifact.score_xgb_with_margin(
            self.xgb_model,
            xgb_features,
            self.feature_names
        )
        lstm_logit, lstm_raw_probability = artifact.score_lstm(
            self.lstm_model,
            temporal,
            missing_mask,
            padding_mask,
            static,
            self.normalization
        )
        xgb_probability = artifact.platt_probability(
            xgb_raw_probability,
            self.calibration["xgb_intercept"],
            self.calibration["xgb_coefficient"]
        )
        lstm_probability = artifact.platt_probability(
            lstm_raw_probability,
            self.calibration["lstm_intercept"],
            self.calibration["lstm_coefficient"]
        )
        ensemble_probability = (
            float(self.ensemble["xgb_weight"]) * xgb_probability
            + float(self.ensemble["lstm_weight"]) * lstm_probability
        )
        threshold = float(self.ensemble["threshold"])
        prediction = int(ensemble_probability >= threshold)
        return {
            "xgb_raw_probability": float(xgb_raw_probability),
            "xgb_raw_margin": float(xgb_raw_margin),
            "xgb_probability": float(xgb_probability),
            "lstm_logit": float(lstm_logit),
            "lstm_raw_probability": float(lstm_raw_probability),
            "lstm_probability": float(lstm_probability),
            "ensemble_probability": float(ensemble_probability),
            "threshold": threshold,
            "prediction": prediction,
            "sequence_length": int(sequence_length),
            "model_version": getattr(C, "MODEL_VERSION", "frozen_ensemble")
        }
def main():
    import artifact
    artifacts = artifact.load_all()
    model = artifacts["xgb_model"]
    feature_names = artifacts["feature_names"]
    engine = RealtimeSHAP(model, feature_names=feature_names, top_k=20, max_gap_windows=6, state_ttl_seconds=21600, max_stays=10000)
    print(f"Realtime SHAP: features={engine.expected_features}, tree_limit={engine.tree_limit}, expected_value={engine.expected_value:.8f}")
    result = engine.startup_self_test()
    print(result)
if __name__ == "__main__":
    main()
