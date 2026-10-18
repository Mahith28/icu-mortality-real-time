import importlib.util
from pathlib import Path
import numpy as np

PATH = Path(__file__).with_name("09_kafka_realtime.py")
SPEC = importlib.util.spec_from_file_location("kafka_realtime_live_policy_test", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
STATE_PATH = Path(__file__).with_name("13_live_update_state.py")
STATE_SPEC = importlib.util.spec_from_file_location("live_update_state_policy_test", STATE_PATH)
STATE_MODULE = importlib.util.module_from_spec(STATE_SPEC)
STATE_SPEC.loader.exec_module(STATE_MODULE)

class Future:
    def get(self, timeout=None):
        return None

class Producer:
    def __init__(self):
        self.messages = []
    def send(self, topic, key=None, value=None):
        self.messages.append((topic, key, value))
        return Future()

class CompareEngine:
    def compare_live(self, previous, current):
        changed_features = []
        changed_vitals = []
        if previous is not None:
            old_features = np.asarray(previous["features"], dtype=np.float64)
            new_features = np.asarray(current["features"], dtype=np.float64)
            changed_features = [i for i, (old, new) in enumerate(zip(old_features, new_features)) if not (np.isnan(old) and np.isnan(new)) and (np.isnan(old) != np.isnan(new) or (not np.isnan(old) and abs(float(new - old)) > 1e-6))]
            old_obs = previous.get("observations", {})
            new_obs = current.get("observations", {})
            changed_vitals = [name for name, value in new_obs.items() if old_obs.get(name) != value]
        return {"changed_features": changed_features, "changed_vitals": changed_vitals}

def fake_scoring(stay_id, window_id, charttime, result, sequence, inference, shap_engine, status):
    probability = 0.2 + float(np.asarray(result["features"])[0]) * 0.01
    return ({"ensemble_probability": probability, "xgb_probability": probability, "lstm_probability": probability, "prediction": 0, "sequence_length": 1, "threshold": 0.108, "xgb_raw_probability": probability, "lstm_raw_probability": probability, "lstm_raw_margin": 0.0, "model_version": "test"}, {"status": "ok", "shap": {"shap_values": [0.0] * 137, "top_features": []}})

def publish(producer, manager, features, observations, refresh=False, charttime="2026-01-01T00:01:00Z"):
    result = {"features": np.asarray(features, dtype=np.float32), "missing_mask": np.zeros(54, dtype=np.float32)}
    return MODULE.publish_live_update(producer, 1001, 1, charttime, result, {}, object(), CompareEngine(), STATE_MODULE, manager, observations, {}, "realtime", "test", "2026-01-01T00:01:01Z", "event-test", "2026-01-01T00:00:00Z", "2026-01-01T00:05:00Z", refresh=refresh)

def main():
    MODULE.run_scoring = fake_scoring
    MODULE.make_risk_message = lambda *args, **kwargs: {}
    producer = Producer()
    manager = STATE_MODULE.LiveUpdateStateManager(expected_n_features=137)
    features = np.zeros(137, dtype=np.float32)
    assert publish(producer, manager, features, {"hr": 80.0}) is True
    assert len([m for m in producer.messages if m[0] == MODULE.RISK_TOPIC]) == 1
    assert publish(producer, manager, features, {"hr": 80.0}) is False
    assert len([m for m in producer.messages if m[0] == MODULE.RISK_TOPIC]) == 1
    changed = features.copy()
    changed[0] = 1.0
    assert publish(
        producer, manager, changed, {"hr": 81.0},
        charttime="2026-01-01T00:02:00Z",
    ) is True
    assert publish(
        producer, manager, changed, {"hr": 81.0},
        refresh=True, charttime="2026-01-01T00:02:00Z",
    ) is True
    risk_messages = [m[2] for m in producer.messages if m[0] == MODULE.RISK_TOPIC]
    assert [message["update_reason"] for message in risk_messages] == ["initial_update", "meaningful_change", "periodic_refresh"]
    print("LIVE UPDATE MEANINGFUL-CHANGE AND REFRESH TEST PASSED")

if __name__ == "__main__":
    main()
