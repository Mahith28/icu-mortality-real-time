import logging
import math
import threading
import time
from datetime import datetime, date
from collections import OrderedDict
import numpy as np
import shap
import xgboost as xgb
import config as C
logger = logging.getLogger("realtime_shap")
class RealtimeSHAP:
    def __init__(self, xgb_model, feature_names, top_k=10, additivity_tolerance=1e-3, max_gap_windows=6, state_ttl_seconds=21600, max_stays=10000):
        self.model = xgb_model
        self.feature_names = list(feature_names)
        self.expected_features = int(C.FEATURE_COUNT)
        self.top_k = int(top_k)
        self.additivity_tolerance = float(additivity_tolerance)
        self.max_gap_windows = None if max_gap_windows is None else int(max_gap_windows)
        self.state_ttl_seconds = float(state_ttl_seconds)
        self.max_stays = int(max_stays)
        self.tree_limit = int(C.XGB_BEST_ITERATION) + 1
        self.previous_state = OrderedDict()
        self.lock = threading.RLock()
        self.last_cleanup = time.monotonic()
        self.missing_margin_count = 0
        if self.expected_features <= 0 or self.top_k <= 0 or self.state_ttl_seconds <= 0 or self.max_stays <= 0:
            raise ValueError("Invalid SHAP configuration")
        if self.max_gap_windows is not None and self.max_gap_windows < 0:
            raise ValueError("max_gap_windows must be >= 0")
        if len(self.feature_names) != self.expected_features or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("Feature-name contract is invalid")
        self._validate_model()
        rounds = int(self.model.num_boosted_rounds())
        self.shap_model = self.model if self.tree_limit == rounds else self.model[:self.tree_limit]
        if int(self.shap_model.num_boosted_rounds()) != self.tree_limit:
            raise RuntimeError("Limited XGBoost model has unexpected round count")
        self.explainer = shap.TreeExplainer(self.shap_model, feature_perturbation="tree_path_dependent", model_output="raw")
        self.expected_value = self._get_expected_value()
        if not math.isfinite(self.expected_value):
            raise ValueError(f"SHAP expected value is not finite: {self.expected_value}")
    def _validate_model(self):
        if not isinstance(self.model, xgb.Booster):
            raise TypeError(f"Expected xgboost.Booster, got {type(self.model).__name__}")
        if int(self.model.num_features()) != self.expected_features:
            raise ValueError(f"Model feature count mismatch: expected {self.expected_features}, got {self.model.num_features()}")
        model_names = self.model.feature_names
        if model_names is not None and list(model_names) != self.feature_names:
            raise ValueError("Model feature names/order do not match the realtime feature contract")
        rounds = int(self.model.num_boosted_rounds())
        if self.tree_limit <= 0 or self.tree_limit > rounds:
            raise ValueError(f"Invalid tree_limit {self.tree_limit} for {rounds} boosted rounds")
        best_iteration = getattr(self.model, "best_iteration", None)
        if best_iteration is not None and int(best_iteration) != int(C.XGB_BEST_ITERATION):
            raise ValueError(f"XGB best iteration mismatch: config={C.XGB_BEST_ITERATION}, model={best_iteration}")
    def _get_expected_value(self):
        value = self.explainer.expected_value
        if isinstance(value, np.ndarray):
            value = value.reshape(-1)[0]
        return float(value)
    def _validate_features(self, features):
        values = np.asarray(features, dtype=np.float32).reshape(-1)
        if values.shape != (self.expected_features,):
            raise RuntimeError(f"Expected {self.expected_features} features but got {values.shape}")
        if np.isinf(values).any():
            raise RuntimeError("XGBoost features contain infinite values")
        return values
    def _validate_probability(self, value, name):
        value = float(value)
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
        return value
    def _normalize_stay_id(self, stay_id):
        if stay_id is None:
            raise ValueError("stay_id cannot be None")
        return str(stay_id)
    def _normalize_window_id(self, window_id):
        if window_id is None or isinstance(window_id, (bool, np.bool_)):
            raise ValueError("window_id must be an integer")
        value = int(window_id)
        if isinstance(window_id, (float, np.floating)) and not float(window_id).is_integer():
            raise ValueError(f"window_id must be an integer, got {window_id!r}")
        return value
    def _normalize_timestamp(self, timestamp):
        if timestamp is None:
            return None
        if isinstance(timestamp, np.datetime64):
            return str(timestamp)
        if isinstance(timestamp, (datetime, date)):
            return timestamp.isoformat()
        if hasattr(timestamp, "isoformat"):
            return timestamp.isoformat()
        return str(timestamp)
    def _get_margin(self, features, xgb_raw_probability=None, xgb_raw_margin=None):
        if xgb_raw_margin is not None:
            margin = float(xgb_raw_margin)
            if not math.isfinite(margin):
                raise ValueError("xgb_raw_margin must be finite")
            return margin
        if xgb_raw_probability is None:
            raise ValueError("Either xgb_raw_margin or xgb_raw_probability is required")
        probability = self._validate_probability(xgb_raw_probability, "xgb_raw_probability")
        if probability <= 0.0 or probability >= 1.0:
            margin = self._predict_margin(features)
            logger.warning("xgb_raw_probability=%s is saturated; using model output margin", probability)
            return margin
        return float(np.log(probability / (1.0 - probability)))
    def _predict_margin(self, features):
        matrix = xgb.DMatrix(features.reshape(1, -1), missing=np.nan, feature_names=self.feature_names)
        prediction = self.shap_model.predict(matrix, output_margin=True)
        return float(np.asarray(prediction).reshape(-1)[0])
    def _predict_probability(self, features):
        matrix = xgb.DMatrix(features.reshape(1, -1), missing=np.nan, feature_names=self.feature_names)
        prediction = self.shap_model.predict(matrix, output_margin=False)
        return float(np.asarray(prediction).reshape(-1)[0])
    def _predict_reference_margin(self, features):
        matrix = xgb.DMatrix(features.reshape(1, -1), missing=np.nan, feature_names=self.feature_names)
        prediction = self.model.predict(matrix, output_margin=True, iteration_range=(0, self.tree_limit))
        return float(np.asarray(prediction).reshape(-1)[0])
    def _sigmoid(self, value):
        value = float(value)
        if value >= 0:
            z = math.exp(-value)
            return 1.0 / (1.0 + z)
        z = math.exp(value)
        return z / (1.0 + z)
    def explain(self, features, xgb_raw_probability=None, xgb_raw_margin=None, verify=True):
        values = self._validate_features(features)
        shap_values = np.asarray(self.explainer.shap_values(values.reshape(1, -1), check_additivity=False)).reshape(-1)
        if shap_values.shape != (self.expected_features,) or not np.all(np.isfinite(shap_values)):
            raise RuntimeError("Invalid SHAP values")
        reconstructed_margin = self.expected_value + float(np.sum(shap_values))
        model_margin = self._get_margin(values, xgb_raw_probability, xgb_raw_margin)
        reconstructed_probability = self._sigmoid(reconstructed_margin)
        margin_error = abs(reconstructed_margin - model_margin)
        probability_error = None if xgb_raw_probability is None else abs(reconstructed_probability - self._validate_probability(xgb_raw_probability, "xgb_raw_probability"))
        if verify and margin_error > self.additivity_tolerance:
            raise RuntimeError(f"SHAP margin additivity check failed: error={margin_error:.8g}")
        if verify and probability_error is not None and probability_error > 5e-4:
            raise RuntimeError(f"SHAP probability check failed: error={probability_error:.8g}")
        return values, shap_values, model_margin, margin_error
    def _top_features(self, values, shap_values):
        order = np.argsort(-np.abs(shap_values), kind="stable")[:self.top_k]
        return [{"feature": self.feature_names[int(i)], "value": None if np.isnan(values[int(i)]) else float(values[int(i)]), "shap_value": float(shap_values[int(i)]), "abs_shap_value": float(abs(shap_values[int(i)]))} for i in order]
    def _changed_features(self, previous_values, previous_shap, current_values, current_shap):
        if previous_values is None or previous_shap is None:
            return []
        shap_delta = current_shap - previous_shap
        order = np.argsort(-np.abs(shap_delta), kind="stable")[:self.top_k]
        output = []
        for i in order:
            i = int(i)
            pv = previous_values[i]
            cv = current_values[i]
            pv_nan = np.isnan(pv)
            cv_nan = np.isnan(cv)
            output.append({"feature": self.feature_names[i], "previous_value": None if pv_nan else float(pv), "current_value": None if cv_nan else float(cv), "previous_shap_value": float(previous_shap[i]), "current_shap_value": float(current_shap[i]), "value_delta": None if pv_nan or cv_nan else float(cv - pv), "shap_delta": float(shap_delta[i]), "value_changed": bool(pv_nan != cv_nan or (not pv_nan and not cv_nan and abs(float(cv - pv)) > 1e-6))})
        return output
    def explain_live(self, features, xgb_raw_probability=None, xgb_raw_margin=None, verify=True):
        values, shap_values, model_margin, margin_error = self.explain(features, xgb_raw_probability, xgb_raw_margin, verify)
        return {"component": "xgboost", "output": "raw_log_odds", "expected_value": self.expected_value, "model_margin": model_margin, "additivity_error": margin_error, "tree_limit": self.tree_limit, "tree_count": int(self.model.num_boosted_rounds()), "top_features": self._top_features(values, shap_values), "shap_values": shap_values.astype(float).tolist()}
    def update(self, stay_id, window_id, timestamp, features, inference_result, window_status="FINAL"):
        start = time.perf_counter()
        stay_key = self._normalize_stay_id(stay_id)
        original_window_id = int(window_id)
        current_window_id = self._normalize_window_id(window_id)
        timestamp = self._normalize_timestamp(timestamp)
        required = ["xgb_raw_probability", "xgb_probability", "lstm_probability", "ensemble_probability"]
        if not isinstance(inference_result, dict):
            raise TypeError("inference_result must be a dictionary")
        missing = [name for name in required if name not in inference_result]
        if missing:
            raise KeyError(f"Missing inference fields: {missing}")
        xgb_raw_probability = self._validate_probability(inference_result["xgb_raw_probability"], "xgb_raw_probability")
        xgb_probability = self._validate_probability(inference_result["xgb_probability"], "xgb_probability")
        lstm_probability = self._validate_probability(inference_result["lstm_probability"], "lstm_probability")
        ensemble_probability = self._validate_probability(inference_result["ensemble_probability"], "ensemble_probability")
        xgb_raw_margin = inference_result.get("xgb_raw_margin")
        if xgb_raw_margin is not None and not math.isfinite(float(xgb_raw_margin)):
            raise ValueError("xgb_raw_margin must be finite")
        with self.lock:
            self._cleanup_state()
            previous = self.previous_state.get(stay_key)
            if previous is not None and current_window_id <= previous["window_id"]:
                if current_window_id == previous["window_id"] and previous.get("last_result") is not None:
                    return self.validate_json_safe(previous["last_result"])
                raise ValueError(f"SHAP window order violation: previous={previous['window_id']}, current={current_window_id}")
            previous_window_id = None if previous is None else int(previous["window_id_original"])
            window_gap = None if previous is None else current_window_id - int(previous["window_id"])
            previous_discarded = previous is not None and self.max_gap_windows is not None and window_gap > self.max_gap_windows
            values, shap_values, model_margin, margin_error = self.explain(features, xgb_raw_probability, xgb_raw_margin, verify=True)
            previous_values = None if previous is None or previous_discarded else previous["features"]
            previous_shap = None if previous is None or previous_discarded else previous["shap_values"]
            previous_xgb = None if previous is None or previous_discarded else previous["xgb_probability"]
            previous_lstm = None if previous is None or previous_discarded else previous["lstm_probability"]
            previous_ensemble = None if previous is None or previous_discarded else previous["ensemble_probability"]
            explanation = {"component": "xgboost", "output": "raw_log_odds", "expected_value": self.expected_value, "model_margin": model_margin, "additivity_error": margin_error, "tree_limit": self.tree_limit, "tree_count": int(self.model.num_boosted_rounds()), "top_features": self._top_features(values, shap_values), "shap_values": shap_values.astype(float).tolist()}
            result = {"status": "ok", "stay_id": int(stay_key), "window_id": original_window_id, "timestamp": timestamp, "window_status": str(window_status), "previous_window_id": previous_window_id, "window_gap": None if previous_discarded else window_gap, "previous_discarded": bool(previous_discarded), "xgb_raw_probability": xgb_raw_probability, "xgb_raw_margin": None if xgb_raw_margin is None else float(xgb_raw_margin), "xgb_probability": xgb_probability, "lstm_probability": lstm_probability, "ensemble_probability": ensemble_probability, "xgb_probability_delta": None if previous_xgb is None else float(xgb_probability - previous_xgb), "lstm_probability_delta": None if previous_lstm is None else float(lstm_probability - previous_lstm), "ensemble_probability_delta": None if previous_ensemble is None else float(ensemble_probability - previous_ensemble), "risk": {"ensemble_probability": ensemble_probability, "ensemble_probability_previous": previous_ensemble, "ensemble_probability_delta": None if previous_ensemble is None else float(ensemble_probability - previous_ensemble)}, "xgb_component": {"raw_probability": xgb_raw_probability, "raw_margin": None if xgb_raw_margin is None else float(xgb_raw_margin), "probability": xgb_probability, "probability_previous": previous_xgb, "probability_delta": None if previous_xgb is None else float(xgb_probability - previous_xgb), "weight": float(C.XGB_WEIGHT)}, "lstm_component": {"probability": lstm_probability, "probability_previous": previous_lstm, "probability_delta": None if previous_lstm is None else float(lstm_probability - previous_lstm), "weight": float(C.LSTM_WEIGHT)}, "shap": explanation, "changed_features": self._changed_features(previous_values, previous_shap, values, shap_values), "latency_ms": (time.perf_counter() - start) * 1000.0}
            self.previous_state[stay_key] = {"window_id": current_window_id, "window_id_original": original_window_id, "timestamp": timestamp, "window_status": str(window_status), "features": values.copy(), "shap_values": shap_values.copy(), "xgb_probability": xgb_probability, "lstm_probability": lstm_probability, "ensemble_probability": ensemble_probability, "updated_at": time.monotonic(), "last_result": result}
            self.previous_state.move_to_end(stay_key)
            while len(self.previous_state) > self.max_stays:
                self.previous_state.popitem(last=False)
            return self.validate_json_safe(result)
    def _cleanup_state(self):
        now = time.monotonic()
        if now - self.last_cleanup < 60 and len(self.previous_state) <= self.max_stays:
            return
        expired = [key for key, state in self.previous_state.items() if now - state["updated_at"] > self.state_ttl_seconds]
        for key in expired:
            self.previous_state.pop(key, None)
        while len(self.previous_state) > self.max_stays:
            self.previous_state.popitem(last=False)
        self.last_cleanup = now
    def reset(self, stay_id):
        with self.lock:
            self.previous_state.pop(self._normalize_stay_id(stay_id), None)
    def clear(self):
        with self.lock:
            self.previous_state.clear()
            self.last_cleanup = time.monotonic()
    def get_previous(self, stay_id):
        with self.lock:
            state = self.previous_state.get(self._normalize_stay_id(stay_id))
            if state is None:
                return None
            return {"window_id": state["window_id"], "window_id_original": state["window_id_original"], "timestamp": state["timestamp"], "window_status": state["window_status"], "features": state["features"].copy(), "shap_values": state["shap_values"].copy(), "xgb_probability": state["xgb_probability"], "lstm_probability": state["lstm_probability"], "ensemble_probability": state["ensemble_probability"], "updated_at": state["updated_at"]}
    def __len__(self):
        with self.lock:
            return len(self.previous_state)
    def validate_json_safe(self, value):
        if isinstance(value, dict):
            return {str(k): self.validate_json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.validate_json_safe(v) for v in value]
        if isinstance(value, tuple):
            return [self.validate_json_safe(v) for v in value]
        if isinstance(value, np.ndarray):
            return self.validate_json_safe(value.tolist())
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            value = float(value)
            return value if math.isfinite(value) else None
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, np.datetime64):
            return str(value)
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        return value
    def startup_self_test(self):
        rows = [np.zeros(self.expected_features, dtype=np.float32), np.full(self.expected_features, np.nan, dtype=np.float32)]
        for index, values in enumerate(rows):
            values = self._validate_features(values)
            margin = self._predict_margin(values)
            reference_margin = self._predict_reference_margin(values)
            if abs(margin - reference_margin) > 1e-4:
                raise RuntimeError(f"Sliced model disagrees with iteration_range inference on row {index}")
            probability = self._predict_probability(values)
            _, shap_values, _, error = self.explain(values, xgb_raw_probability=probability, xgb_raw_margin=margin, verify=True)
            if len(shap_values) != self.expected_features or not np.all(np.isfinite(shap_values)) or not math.isfinite(error):
                raise RuntimeError(f"SHAP startup self-test failed on row {index}")
        return {"status": "ok", "rows_tested": len(rows), "feature_count": self.expected_features, "tree_limit": self.tree_limit, "model_boosted_rounds": int(self.model.num_boosted_rounds()), "expected_value": self.expected_value}
if __name__ == "__main__":
    import artifact
    artifacts = artifact.load_all()
    engine = RealtimeSHAP(artifacts["xgb_model"], artifacts["feature_names"])
    print(engine.startup_self_test())