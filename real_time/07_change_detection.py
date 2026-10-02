import copy
from collections import deque
import numpy as np
import config as C
class ChangeDetector:
    def __init__(self, threshold=None, cooldown_windows=None, escalation_delta=0.10, min_rise=0.01):
        self.threshold = float(C.ENSEMBLE_THRESHOLD if threshold is None else threshold)
        if isinstance(cooldown_windows, bool) or (cooldown_windows is not None and not isinstance(cooldown_windows, (int, np.integer))):
            raise TypeError("cooldown_windows must be an integer")
        self.cooldown_windows = int(C.ALERT_COOLDOWN_WINDOWS if cooldown_windows is None else cooldown_windows)
        if isinstance(escalation_delta, bool) or not isinstance(escalation_delta, (int, float, np.integer, np.floating)):
            raise TypeError("escalation_delta must be numeric")
        if isinstance(min_rise, bool) or not isinstance(min_rise, (int, float, np.integer, np.floating)):
            raise TypeError("min_rise must be numeric")
        self.escalation_delta = float(escalation_delta)
        self.min_rise = float(min_rise)
        if not np.isfinite(self.threshold):
            raise ValueError("threshold must be finite")
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if self.cooldown_windows < 0:
            raise ValueError("cooldown_windows must be non-negative")
        if not np.isfinite(self.escalation_delta):
            raise ValueError("escalation_delta must be finite")
        if self.escalation_delta <= 0.0:
            raise ValueError("escalation_delta must be positive")
        if not np.isfinite(self.min_rise):
            raise ValueError("min_rise must be finite")
        if self.min_rise < 0.0:
            raise ValueError("min_rise must be non-negative")
        self.states = {}
    def _validate(self, stay_id, window_id, probability, prediction):
        if stay_id is None:
            raise ValueError("stay_id is required")
        if isinstance(window_id, bool) or not isinstance(window_id, (int, np.integer)):
            raise TypeError("window_id must be an integer")
        if window_id < 0:
            raise ValueError("window_id must be non-negative")
        if isinstance(probability, bool) or not isinstance(probability, (int, float, np.integer, np.floating)):
            raise TypeError("ensemble_probability must be numeric")
        probability = float(probability)
        if not np.isfinite(probability):
            raise ValueError("ensemble_probability must be finite")
        if not 0.0 <= probability <= 1.0:
            raise ValueError("ensemble_probability must be between 0 and 1")
        if prediction is None:
            raise TypeError("prediction is required")
        if isinstance(prediction, bool) or not isinstance(prediction, (int, np.integer)):
            raise TypeError("prediction must be an integer")
        if int(prediction) not in (0, 1):
            raise ValueError("prediction must be 0 or 1")
        return probability, int(prediction)
    def _cooldown_remaining(self, state, window_id):
        if state["last_alert_window_id"] is None:
            return 0
        elapsed = window_id - state["last_alert_window_id"]
        return max(0, self.cooldown_windows - elapsed)
    def _direction(self, previous_probability, probability):
        if previous_probability is None:
            return "initial"
        delta = probability - previous_probability
        if delta > self.min_rise:
            return "rising"
        if delta < -self.min_rise:
            return "falling"
        return "stable"
    def update(self, stay_id, window_id, ensemble_probability, prediction):
        probability, supplied_prediction = self._validate(stay_id, window_id, ensemble_probability, prediction)
        state = self.states.get(stay_id)
        if state is not None and state["last_window_id"] is not None and window_id <= state["last_window_id"]:
            stale = window_id < state["last_window_id"]
            return {
                "stay_id": stay_id,
                "window_id": int(window_id),
                "ensemble_probability": None,
                "threshold": self.threshold,
                "prediction": None,
                "supplied_prediction": None,
                "risk_direction": "stale",
                "alert": False,
                "alert_reason": "stale_suppressed" if stale else "duplicate_suppressed",
                "pending_alert": state["pending_alert"],
                "cooldown_remaining": self._cooldown_remaining(state, state["last_window_id"]),
                "prediction_mismatch": False,
                "duplicate": not stale,
                "stale": stale,
                "fresh_evaluation": False
            }
        if state is None:
            state = {
                "last_window_id": None,
                "last_probability": None,
                "last_alert_window_id": None,
                "last_alert_probability": None,
                "pending_alert": False
            }
            self.states[stay_id] = state
        previous_probability = state["last_probability"]
        direction = self._direction(previous_probability, probability)
        current_above = probability >= self.threshold
        previous_above = previous_probability is not None and previous_probability >= self.threshold
        crossing = previous_probability is not None and not previous_above and current_above
        cooldown_remaining = self._cooldown_remaining(state, window_id)
        prediction_mismatch = supplied_prediction != int(current_above)
        alert = False
        alert_reason = None
        if previous_probability is None:
            if current_above:
                alert = True
                alert_reason = "initial_high_risk"
        elif crossing:
            if cooldown_remaining == 0:
                alert = True
                alert_reason = "threshold_crossing"
            else:
                state["pending_alert"] = True
                alert_reason = "threshold_crossing_suppressed"
        elif state["pending_alert"] and current_above and cooldown_remaining == 0:
            alert = True
            alert_reason = "pending_threshold_crossing"
        if current_above and state["last_alert_probability"] is not None and cooldown_remaining == 0 and probability - state["last_alert_probability"] >= self.escalation_delta:
            alert = True
            alert_reason = "risk_escalation"
        if alert:
            state["last_alert_window_id"] = int(window_id)
            state["last_alert_probability"] = probability
            state["pending_alert"] = False
        if not current_above and state["pending_alert"]:
            state["pending_alert"] = False
        state["last_window_id"] = int(window_id)
        state["last_probability"] = probability
        return {
            "stay_id": stay_id,
            "window_id": int(window_id),
            "ensemble_probability": probability,
            "threshold": self.threshold,
            "prediction": int(current_above),
            "supplied_prediction": supplied_prediction,
            "risk_direction": direction,
            "alert": alert,
            "alert_reason": alert_reason,
            "pending_alert": state["pending_alert"],
            "cooldown_remaining": self._cooldown_remaining(state, window_id),
            "prediction_mismatch": prediction_mismatch,
            "duplicate": False,
            "stale": False,
            "fresh_evaluation": True
        }
    def reset(self, stay_id):
        self.states.pop(stay_id, None)
    def get_state(self, stay_id):
        state = self.states.get(stay_id)
        if state is None:
            return None
        return copy.deepcopy(state)
class AlertLogger:
    def __init__(self, max_events=10000):
        if isinstance(max_events, bool) or not isinstance(max_events, (int, np.integer)):
            raise TypeError("max_events must be an integer")
        if max_events <= 0:
            raise ValueError("max_events must be positive")
        self.events = deque(maxlen=int(max_events))
    def record(self, result):
        if result.get("alert") or result.get("prediction_mismatch") or result.get("alert_reason") == "threshold_crossing_suppressed":
            self.events.append(copy.deepcopy(result))
    def get_events(self):
        return copy.deepcopy(list(self.events))
    def clear(self):
        self.events.clear()