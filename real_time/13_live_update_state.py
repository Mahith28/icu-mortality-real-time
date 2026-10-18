from copy import deepcopy
from datetime import datetime
from math import isfinite
from numbers import Integral, Number
import numpy as np
import pandas as pd
class LiveUpdateStateError(Exception):
    pass
class LiveUpdateStateValidationError(LiveUpdateStateError):
    pass
class OutOfOrderUpdateError(LiveUpdateStateError):
    pass
class StaleWindowError(LiveUpdateStateError):
    pass
class ClosedWindowError(LiveUpdateStateError):
    pass
class LiveUpdateState:
    PROBABILITY_TOLERANCE = 1e-6
    def __init__(
        self,
        stay_id,
        window_id,
        charttime,
        ensemble_probability,
        features,
        shap_values,
        xgb_probability=None,
        lstm_probability=None,
        observations=None,
    ):
        self.stay_id = self._validate_stay_id(stay_id)
        self.window_id = self._validate_window_id(window_id)
        self.charttime = self._validate_charttime(charttime)
        self.ensemble_probability = self._validate_probability(
            ensemble_probability,
            "ensemble_probability",
            required=True,
        )
        self.xgb_probability = self._validate_probability(xgb_probability, "xgb_probability")
        self.lstm_probability = self._validate_probability(lstm_probability, "lstm_probability")
        self.features = self._validate_features(features)
        self.shap_values = self._validate_shap_values(shap_values)
        self._validate_lengths(self.features, self.shap_values)
        self.observations = self._validate_observations(observations)
    @staticmethod
    def _validate_stay_id(stay_id):
        if stay_id is None:
            raise LiveUpdateStateValidationError("stay_id is required")
        if isinstance(stay_id, bool):
            raise LiveUpdateStateValidationError("stay_id must not be boolean")
        if isinstance(stay_id, Integral):
            return str(int(stay_id))
        if isinstance(stay_id, (float, np.floating)):
            numeric = float(stay_id)
            if not isfinite(numeric):
                raise LiveUpdateStateValidationError("stay_id must be finite")
            if numeric.is_integer():
                return str(int(numeric))
        value = str(stay_id).strip()
        if not value:
            raise LiveUpdateStateValidationError("stay_id must not be empty")
        return value
    @staticmethod
    def _validate_window_id(window_id):
        if isinstance(window_id, bool):
            raise LiveUpdateStateValidationError("window_id must be an integer")
        if isinstance(window_id, Integral):
            value = int(window_id)
        elif isinstance(window_id, float) and window_id.is_integer():
            value = int(window_id)
        elif isinstance(window_id, str):
            try:
                value = int(window_id.strip())
            except ValueError as exc:
                raise LiveUpdateStateValidationError("window_id must be an integer") from exc
        else:
            raise LiveUpdateStateValidationError("window_id must be an integer")
        if value < 0:
            raise LiveUpdateStateValidationError("window_id must be non-negative")
        return value
    @staticmethod
    def _validate_charttime(charttime):
        if isinstance(charttime, Number):
            raise LiveUpdateStateValidationError("charttime must be datetime-like")
        if isinstance(charttime, (pd.Timestamp, datetime, np.datetime64)):
            try:
                timestamp = pd.Timestamp(charttime)
            except Exception as exc:
                raise LiveUpdateStateValidationError("charttime must be a valid datetime") from exc
        elif isinstance(charttime, str):
            try:
                timestamp = pd.Timestamp(charttime)
            except Exception as exc:
                raise LiveUpdateStateValidationError("charttime must be a valid datetime") from exc
        else:
            raise LiveUpdateStateValidationError("charttime must be datetime-like")
        if pd.isna(timestamp):
            raise LiveUpdateStateValidationError("charttime must not be NaT")
        if timestamp.tzinfo is None:
            return timestamp.tz_localize("UTC")
        return timestamp.tz_convert("UTC")
    @classmethod
    def _validate_probability(cls, value, name, required=False):
        if value is None:
            if required:
                raise LiveUpdateStateValidationError(f"{name} is required")
            return None
        if isinstance(value, bool):
            raise LiveUpdateStateValidationError(f"{name} must be numeric")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise LiveUpdateStateValidationError(f"{name} must be numeric") from exc
        if not isfinite(numeric):
            raise LiveUpdateStateValidationError(f"{name} must be finite")
        if numeric < -cls.PROBABILITY_TOLERANCE or numeric > 1.0 + cls.PROBABILITY_TOLERANCE:
            raise LiveUpdateStateValidationError(f"{name} must be between 0 and 1")
        return float(min(1.0, max(0.0, numeric)))
    @staticmethod
    def _validate_features(features):
        if features is None:
            raise LiveUpdateStateValidationError("features are required")
        try:
            values = list(features)
        except TypeError as exc:
            raise LiveUpdateStateValidationError("features must be iterable") from exc
        if not values:
            raise LiveUpdateStateValidationError("features must not be empty")
        validated = []
        for index, value in enumerate(values):
            if value is None:
                validated.append(np.nan)
                continue
            if isinstance(value, bool):
                raise LiveUpdateStateValidationError(f"feature {index} must be numeric")
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise LiveUpdateStateValidationError(f"feature {index} must be numeric") from exc
            if np.isinf(numeric):
                raise LiveUpdateStateValidationError(f"feature {index} must be finite or NaN")
            validated.append(numeric)
        return validated
    @staticmethod
    def _validate_shap_values(shap_values):
        if shap_values is None:
            raise LiveUpdateStateValidationError("shap_values are required")
        try:
            values = list(shap_values)
        except TypeError as exc:
            raise LiveUpdateStateValidationError("shap_values must be iterable") from exc
        if not values:
            raise LiveUpdateStateValidationError("shap_values must not be empty")
        validated = []
        for index, value in enumerate(values):
            if value is None:
                raise LiveUpdateStateValidationError(f"shap value {index} must be finite")
            if isinstance(value, bool):
                raise LiveUpdateStateValidationError(f"shap value {index} must be numeric")
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise LiveUpdateStateValidationError(f"shap value {index} must be numeric") from exc
            if not isfinite(numeric):
                raise LiveUpdateStateValidationError(f"shap value {index} must be finite")
            validated.append(numeric)
        return validated
    @staticmethod
    def _validate_lengths(features, shap_values):
        if len(features) != len(shap_values):
            raise LiveUpdateStateValidationError(f"features and shap_values length mismatch: {len(features)} != {len(shap_values)}")
    @staticmethod
    def _validate_observations(observations):
        if observations is None:
            return {}
        if not isinstance(observations, dict):
            raise LiveUpdateStateValidationError("observations must be a dictionary")
        validated = {}
        for key, value in observations.items():
            normalized_key = str(key)
            if value is None:
                validated[normalized_key] = None
                continue
            if isinstance(value, bool):
                raise LiveUpdateStateValidationError(f"observation {normalized_key} must be numeric or None")
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise LiveUpdateStateValidationError(f"observation {normalized_key} must be numeric or None") from exc
            if not isfinite(numeric):
                raise LiveUpdateStateValidationError(f"observation {normalized_key} must be finite or None")
            validated[normalized_key] = numeric
        return validated
    def to_dict(self):
        return {
            "stay_id": self.stay_id,
            "window_id": self.window_id,
            "charttime": self.charttime.isoformat(),
            "ensemble_probability": self.ensemble_probability,
            "xgb_probability": self.xgb_probability,
            "lstm_probability": self.lstm_probability,
            "features": list(self.features),
            "shap_values": list(self.shap_values),
            "observations": dict(self.observations),
        }
class LiveUpdateStateManager:
    def __init__(self, expected_n_features=137):
        if expected_n_features is not None:
            if isinstance(expected_n_features, bool) or not isinstance(expected_n_features, Integral):
                raise LiveUpdateStateValidationError("expected_n_features must be an integer or None")
            expected_n_features = int(expected_n_features)
            if expected_n_features <= 0:
                raise LiveUpdateStateValidationError("expected_n_features must be positive")
        self.expected_n_features = expected_n_features
        self._states = {}
        self._last_closed_window = {}
        self._feature_lengths = {}
    def _validate_state(self, state):
        if not isinstance(state, LiveUpdateState):
            raise LiveUpdateStateValidationError("state must be a LiveUpdateState")
        if self.expected_n_features is not None and len(state.features) != self.expected_n_features:
            raise LiveUpdateStateValidationError(f"expected {self.expected_n_features} features, got {len(state.features)}")
        expected_length = self._feature_lengths.get(state.stay_id)
        if expected_length is not None and len(state.features) != expected_length:
            raise LiveUpdateStateValidationError(f"feature length changed for stay {state.stay_id}: {expected_length} != {len(state.features)}")
        return state
    def get(self, stay_id):
        stay_id = LiveUpdateState._validate_stay_id(stay_id)
        state = self._states.get(stay_id)
        if state is None:
            return None
        return deepcopy(state)
    def contains(self, stay_id):
        stay_id = LiveUpdateState._validate_stay_id(stay_id)
        return stay_id in self._states
    def check(self, state, allow_same_charttime_refresh=False):
        state = self._validate_state(state)
        stay_id = state.stay_id
        window_id = state.window_id
        previous = self._states.get(stay_id)
        last_closed = self._last_closed_window.get(stay_id)
        if last_closed is not None and window_id <= last_closed:
            raise ClosedWindowError(f"window {window_id} for stay {stay_id} is already closed at window {last_closed}")
        if previous is None or window_id > previous.window_id:
            return None, "updated"
        if window_id < previous.window_id:
            raise StaleWindowError(f"window {window_id} is older than active window {previous.window_id} for stay {stay_id}")
        if state.charttime == previous.charttime:
            if allow_same_charttime_refresh:
                return deepcopy(previous), "updated"
            return deepcopy(previous), "duplicate"
        if state.charttime < previous.charttime:
            raise OutOfOrderUpdateError(f"charttime {state.charttime} is older than previous charttime {previous.charttime}")
        return deepcopy(previous), "updated"
    def update(self, state, allow_same_charttime_refresh=False):
        state = self._validate_state(state)
        stay_id = state.stay_id
        window_id = state.window_id
        previous = self._states.get(stay_id)
        last_closed = self._last_closed_window.get(stay_id)
        if last_closed is not None and window_id <= last_closed:
            raise ClosedWindowError(f"window {window_id} for stay {stay_id} is already closed at window {last_closed}")
        if previous is None:
            self._states[stay_id] = deepcopy(state)
            self._feature_lengths[stay_id] = len(state.features)
            return None, deepcopy(state), "updated"
        if window_id < previous.window_id:
            raise StaleWindowError(f"window {window_id} is older than active window {previous.window_id} for stay {stay_id}")
        if window_id > previous.window_id:
            self._states[stay_id] = deepcopy(state)
            return None, deepcopy(state), "updated"
        if state.charttime == previous.charttime and not allow_same_charttime_refresh:
            return None, deepcopy(previous), "duplicate"
        if state.charttime < previous.charttime:
            raise OutOfOrderUpdateError(f"charttime {state.charttime} is older than previous charttime {previous.charttime}")
        self._states[stay_id] = deepcopy(state)
        return deepcopy(previous), deepcopy(state), "updated"
    def clear_window(self, stay_id, window_id):
        stay_id = LiveUpdateState._validate_stay_id(stay_id)
        window_id = LiveUpdateState._validate_window_id(window_id)
        state = self._states.get(stay_id)
        if state is not None:
            if window_id < state.window_id:
                return False
            del self._states[stay_id]
            self._last_closed_window[stay_id] = max(window_id, self._last_closed_window.get(stay_id, -1))
            return True
        last_closed = self._last_closed_window.get(stay_id)
        if last_closed is not None and window_id <= last_closed:
            return False
        self._last_closed_window[stay_id] = window_id
        return False
    def to_state_dict(self):
        """Return a JSON-safe snapshot of all manager state.

        NaN feature values are encoded as None because the SQLite state store
        uses strict JSON (allow_nan=False). LiveUpdateState restores None
        feature entries to NaN through its normal validation path.
        """
        states = {}
        for stay_id, state in self._states.items():
            item = state.to_dict()
            item["features"] = [
                None if isinstance(value, (float, np.floating)) and np.isnan(value)
                else float(value)
                for value in state.features
            ]
            states[stay_id] = item
        return {
            "schema_version": 1,
            "expected_n_features": self.expected_n_features,
            "states": states,
            "last_closed_window": dict(self._last_closed_window),
            "feature_lengths": dict(self._feature_lengths),
        }

    def restore_state_dict(self, payload):
        """Validate a manager snapshot fully, then atomically replace its state."""
        if not isinstance(payload, dict):
            raise LiveUpdateStateValidationError("manager snapshot must be a dictionary")
        required = {
            "schema_version", "expected_n_features", "states",
            "last_closed_window", "feature_lengths",
        }
        if set(payload) != required:
            missing = sorted(required - set(payload))
            extra = sorted(set(payload) - required)
            raise LiveUpdateStateValidationError(
                f"invalid manager snapshot keys; missing={missing}, extra={extra}"
            )
        if (
            isinstance(payload["schema_version"], bool)
            or not isinstance(payload["schema_version"], Integral)
            or int(payload["schema_version"]) != 1
        ):
            raise LiveUpdateStateValidationError("unsupported manager snapshot schema_version")
        if payload["expected_n_features"] != self.expected_n_features:
            raise LiveUpdateStateValidationError(
                "snapshot expected_n_features does not match this manager"
            )

        raw_states = payload["states"]
        raw_closed = payload["last_closed_window"]
        raw_lengths = payload["feature_lengths"]
        if not isinstance(raw_states, dict):
            raise LiveUpdateStateValidationError("states must be a dictionary")
        if not isinstance(raw_closed, dict):
            raise LiveUpdateStateValidationError("last_closed_window must be a dictionary")
        if not isinstance(raw_lengths, dict):
            raise LiveUpdateStateValidationError("feature_lengths must be a dictionary")

        new_states, new_closed, new_lengths = {}, {}, {}
        for raw_stay_id, item in raw_states.items():
            stay_id = LiveUpdateState._validate_stay_id(raw_stay_id)
            if stay_id in new_states:
                raise LiveUpdateStateValidationError("duplicate normalized stay_id in states")
            if not isinstance(item, dict):
                raise LiveUpdateStateValidationError(f"state for stay {stay_id} must be a dictionary")
            required_state_keys = {
                "stay_id", "window_id", "charttime", "ensemble_probability",
                "xgb_probability", "lstm_probability", "features", "shap_values",
                "observations",
            }
            if set(item) != required_state_keys:
                missing = sorted(required_state_keys - set(item))
                extra = sorted(set(item) - required_state_keys)
                raise LiveUpdateStateValidationError(
                    f"invalid state keys for stay {stay_id}; missing={missing}, extra={extra}"
                )
            try:
                state = LiveUpdateState(
                    stay_id=item["stay_id"],
                    window_id=item["window_id"],
                    charttime=item["charttime"],
                    ensemble_probability=item["ensemble_probability"],
                    xgb_probability=item.get("xgb_probability"),
                    lstm_probability=item.get("lstm_probability"),
                    features=item["features"],
                    shap_values=item["shap_values"],
                    observations=item.get("observations"),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise LiveUpdateStateValidationError(
                    f"invalid serialized state for stay {stay_id}"
                ) from exc
            if state.stay_id != stay_id:
                raise LiveUpdateStateValidationError(
                    f"state key does not match stay_id for stay {stay_id}"
                )
            if (
                self.expected_n_features is not None
                and len(state.features) != self.expected_n_features
            ):
                raise LiveUpdateStateValidationError(
                    f"expected {self.expected_n_features} features, got {len(state.features)}"
                )
            new_states[stay_id] = deepcopy(state)

        for raw_stay_id, raw_window_id in raw_closed.items():
            stay_id = LiveUpdateState._validate_stay_id(raw_stay_id)
            if stay_id in new_closed:
                raise LiveUpdateStateValidationError("duplicate normalized stay_id in last_closed_window")
            new_closed[stay_id] = LiveUpdateState._validate_window_id(raw_window_id)

        for raw_stay_id, raw_length in raw_lengths.items():
            stay_id = LiveUpdateState._validate_stay_id(raw_stay_id)
            if stay_id in new_lengths:
                raise LiveUpdateStateValidationError("duplicate normalized stay_id in feature_lengths")
            if (
                isinstance(raw_length, bool)
                or not isinstance(raw_length, Integral)
                or int(raw_length) <= 0
            ):
                raise LiveUpdateStateValidationError(
                    f"feature length for stay {stay_id} must be a positive integer"
                )
            length = int(raw_length)
            if self.expected_n_features is not None and length != self.expected_n_features:
                raise LiveUpdateStateValidationError(
                    f"feature length for stay {stay_id} does not match expected_n_features"
                )
            new_lengths[stay_id] = length

        for stay_id, state in new_states.items():
            if stay_id not in new_lengths or len(state.features) != new_lengths[stay_id]:
                raise LiveUpdateStateValidationError(
                    f"missing or inconsistent feature length for active stay {stay_id}"
                )
            if stay_id in new_closed and state.window_id <= new_closed[stay_id]:
                raise LiveUpdateStateValidationError(
                    f"active state for stay {stay_id} is at or before its closed window"
                )

        # Swap only after every section has passed validation.
        self._states = new_states
        self._last_closed_window = new_closed
        self._feature_lengths = new_lengths

    def reset(self, stay_id):
        stay_id = LiveUpdateState._validate_stay_id(stay_id)
        self._states.pop(stay_id, None)
        self._last_closed_window.pop(stay_id, None)
        self._feature_lengths.pop(stay_id, None)
    def clear_all(self):
        self._states.clear()
        self._last_closed_window.clear()
        self._feature_lengths.clear()
    def active_stays(self):
        return tuple(self._states.keys())
    def size(self):
        return len(self._states)
    def evict(self, stay_id):
        """Remove all manager history for a stay.

        Only call after discharge/final cleanup when no replayable observations
        or pending Kafka work can refer to this stay. Eviction removes closed-
        window protection as well as active state and feature-length metadata.
        """
        stay_id = LiveUpdateState._validate_stay_id(stay_id)
        removed = stay_id in self._states or stay_id in self._last_closed_window
        self._states.pop(stay_id, None)
        self._last_closed_window.pop(stay_id, None)
        self._feature_lengths.pop(stay_id, None)
        return removed
