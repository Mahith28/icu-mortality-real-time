from collections import deque
import hashlib
import numpy as np
import config as C
class PatientStateError(ValueError):
    pass
class ConflictingWindowError(PatientStateError):
    pass
class StaleWindowError(PatientStateError):
    pass
class DischargedStayError(PatientStateError):
    pass
class TimestampOrderError(PatientStateError):
    pass
class StaticChangedError(PatientStateError):
    pass
class PatientState:
    def __init__(self, stay_id):
        self.stay_id = stay_id
        self.temporal = deque(maxlen=C.MAX_SEQ_LEN)
        self.missing_mask = deque(maxlen=C.MAX_SEQ_LEN)
        self.timestamps = deque(maxlen=C.MAX_SEQ_LEN)
        self.window_records = deque(maxlen=C.MAX_SEQ_LEN)
        self.static = None
        self.latest_xgb_features = None
        self.last_window_id = None
        self.last_timestamp = None
        self.windows_accepted = 0
        self.discharged = False
    def _window_hash(self, window_id, timestamp, temporal, missing_mask, static, xgb_features):
        hasher = hashlib.sha256()
        hasher.update(str(int(window_id)).encode())
        hasher.update(str(float(timestamp)).encode())
        hasher.update(temporal.tobytes())
        hasher.update(missing_mask.tobytes())
        hasher.update(static.tobytes())
        if xgb_features is None:
            hasher.update(b"NONE")
        else:
            hasher.update(xgb_features.tobytes())
        return hasher.hexdigest()
    def add_window(self, window_id, timestamp, temporal, missing_mask, static, xgb_features=None):
        if isinstance(window_id, bool) or not isinstance(window_id, (int, np.integer)):
            raise TypeError("window_id must be a non-negative integer")
        if window_id < 0:
            raise ValueError("window_id must be non-negative")
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float, np.integer, np.floating)):
            raise TypeError("timestamp must be a numeric Unix timestamp in seconds")
        timestamp = float(timestamp)
        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        temporal = np.asarray(temporal, dtype=np.float32)
        missing_mask = np.asarray(missing_mask, dtype=np.float32)
        static = np.asarray(static, dtype=np.float32)
        if temporal.shape != (C.TEMPORAL_FEATURES,):
            raise ValueError(f"Expected temporal shape {(C.TEMPORAL_FEATURES,)}, got {temporal.shape}")
        if missing_mask.shape != (C.TEMPORAL_FEATURES,):
            raise ValueError(f"Expected missing mask shape {(C.TEMPORAL_FEATURES,)}, got {missing_mask.shape}")
        if static.shape != (C.STATIC_FEATURES,):
            raise ValueError(f"Expected static shape {(C.STATIC_FEATURES,)}, got {static.shape}")
        if not np.isfinite(temporal).all():
            raise ValueError("Temporal features contain non-finite values")
        if not np.isfinite(missing_mask).all():
            raise ValueError("Missing mask contains non-finite values")
        if not np.isin(missing_mask, (0.0, 1.0)).all():
            raise ValueError("Missing mask must contain only 0.0 or 1.0")
        if not np.isfinite(static).all():
            raise ValueError("Static features contain non-finite values")
        if xgb_features is not None:
            xgb_features = np.asarray(xgb_features, dtype=np.float32)
            if xgb_features.shape != (C.FEATURE_COUNT,):
                raise ValueError(f"Expected XGBoost feature shape {(C.FEATURE_COUNT,)}, got {xgb_features.shape}")
            if np.isinf(xgb_features).any():
                raise ValueError("XGBoost features contain infinite values")
        incoming_hash = self._window_hash(window_id, timestamp, temporal, missing_mask, static, xgb_features)
        for stored_window_id, stored_hash in self.window_records:
            if window_id == stored_window_id:
                if incoming_hash == stored_hash:
                    return False
                raise ConflictingWindowError(f"Conflicting duplicate window {window_id} for stay {self.stay_id}")
        if self.discharged:
            raise DischargedStayError(f"Stay {self.stay_id} is already discharged")
        if self.last_window_id is not None and window_id < self.last_window_id:
            raise StaleWindowError(f"Out-of-order window {window_id} after {self.last_window_id}")
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            raise TimestampOrderError(f"Timestamp {timestamp} is not after previous timestamp {self.last_timestamp}")
        if self.static is None:
            self.static = static.copy()
        elif not np.array_equal(static, self.static):
            raise StaticChangedError(f"Static features changed for stay {self.stay_id}")
        self.temporal.append(temporal.copy())
        self.missing_mask.append(missing_mask.copy())
        self.timestamps.append(timestamp)
        self.window_records.append((int(window_id), incoming_hash))
        self.latest_xgb_features = None if xgb_features is None else xgb_features.copy()
        self.last_window_id = int(window_id)
        self.last_timestamp = timestamp
        self.windows_accepted += 1
        return True
    def get_sequence(self):
        sequence_length = len(self.temporal)
        if sequence_length == 0:
            return None
        temporal = np.zeros((C.MAX_SEQ_LEN, C.TEMPORAL_FEATURES), dtype=np.float32)
        missing_mask = np.zeros((C.MAX_SEQ_LEN, C.TEMPORAL_FEATURES), dtype=np.float32)
        padding_mask = np.ones(C.MAX_SEQ_LEN, dtype=np.float32)
        temporal[:sequence_length] = np.asarray(self.temporal, dtype=np.float32)
        missing_mask[:sequence_length] = np.asarray(self.missing_mask, dtype=np.float32)
        padding_mask[:sequence_length] = 0.0
        xgb_features = None if self.latest_xgb_features is None else self.latest_xgb_features.copy()
        return temporal, missing_mask, padding_mask, self.static.copy(), sequence_length, xgb_features
    def get_identity(self):
        return self.last_window_id, self.last_timestamp
    def get_windows_accepted(self):
        return self.windows_accepted
    def discharge(self):
        self.discharged = True
class PatientStateManager:
    def __init__(self, timeout_hours=C.STATE_TIMEOUT_HOURS):
        self.states = {}
        self.timeout_seconds = float(timeout_hours) * 3600.0
    def get(self, stay_id):
        return self.states.get(stay_id)
    def add_window(self, stay_id, window_id, timestamp, temporal, missing_mask, static, xgb_features=None):
        state = self.states.get(stay_id)
        is_new = state is None
        if is_new:
            state = PatientState(stay_id)
        accepted = state.add_window(window_id, timestamp, temporal, missing_mask, static, xgb_features)
        if is_new:
            self.states[stay_id] = state
        return state, accepted
    def discharge(self, stay_id):
        state = self.get(stay_id)
        if state is not None:
            state.discharge()
    def remove(self, stay_id):
        self.states.pop(stay_id, None)
    def cleanup(self, watermark_timestamp, evict_active=True):
        if isinstance(watermark_timestamp, bool) or not isinstance(watermark_timestamp, (int, float, np.integer, np.floating)):
            raise TypeError("watermark_timestamp must be a numeric Unix timestamp in seconds")
        watermark_timestamp = float(watermark_timestamp)
        if not np.isfinite(watermark_timestamp):
            raise ValueError("watermark_timestamp must be finite")
        expired = []
        for stay_id, state in self.states.items():
            if state.last_timestamp is None:
                continue
            if not state.discharged and not evict_active:
                continue
            if watermark_timestamp - state.last_timestamp > self.timeout_seconds:
                expired.append(stay_id)
        for stay_id in expired:
            self.remove(stay_id)
        return expired
    def __len__(self):
        return len(self.states)