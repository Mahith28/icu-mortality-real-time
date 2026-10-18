import importlib.util
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PATH = Path(__file__).with_name("09_kafka_realtime.py")
SPEC = importlib.util.spec_from_file_location("kafka_realtime", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Future:
    def get(self, timeout=None):
        return None


class Producer:
    def __init__(self):
        self.messages = []

    def send(self, topic, key=None, value=None):
        self.messages.append((topic, key, value))
        return Future()


class Preprocessor:
    def __init__(self):
        self.active_windows = {1001: {1: [{"charttime": datetime(2026, 1, 1, 0, 4, 0), "vital_sign": "hr", "value": 80.0}]}}
        self.get_final_window_calls = 0

    def get_final_window(self, stay_id, window_id, static_data):
        self.get_final_window_calls += 1
        if stay_id not in self.active_windows or window_id not in self.active_windows[stay_id]:
            raise RuntimeError("FINAL window was requested after destructive preprocessing")
        del self.active_windows[stay_id][window_id]
        if not self.active_windows[stay_id]:
            del self.active_windows[stay_id]
        return {
            "features": np.zeros(137, dtype=np.float32),
            "missing_mask": np.zeros(54, dtype=np.float32),
            "temporal": np.zeros(54, dtype=np.float32),
            "static": np.zeros(83, dtype=np.float32)
        }

    def build_sequence(self, windows):
        return {
            "temporal": np.zeros((120, 54), dtype=np.float32),
            "missing_mask": np.zeros((120, 54), dtype=np.float32),
            "padding_mask": np.zeros(120, dtype=np.float32),
            "static": np.zeros(83, dtype=np.float32),
            "sequence_length": 120
        }


class Inference:
    def __init__(self):
        self.fail = True

    def predict(self, features, temporal, missing_mask, padding_mask, static):
        if self.fail:
            raise RuntimeError("intentional FINAL scoring failure")
        return {
            "xgb_raw_probability": 0.10,
            "xgb_raw_margin": -2.0,
            "xgb_probability": 0.10,
            "lstm_raw_probability": 0.11,
            "lstm_probability": 0.11,
            "ensemble_probability": 0.105,
            "prediction": 0,
            "threshold": 0.108,
            "sequence_length": 120,
            "model_version": "test"
        }


class SHAP:
    def update(self, stay_id, window_id, timestamp, features, inference_result, window_status="FINAL"):
        def reset(self, stay_id):
            return None
        return {
            "status": "ok",
            "stay_id": int(stay_id),
            "window_id": int(window_id),
            "window_status": window_status,
            "shap": {"top_features": []}
        }


class LiveState:
    def clear_window(self, stay_id, window_id):
        return None


def test_destructive_final_preprocessing_is_not_repeated(monkeypatch):
    monkeypatch.setattr(MODULE, "persist_risk_event_sql", lambda message: True)
    stay_id = 1001
    window_id = 1
    preprocessor = Preprocessor()
    producer = Producer()
    inference = Inference()
    shap_engine = SHAP()
    static_data = {stay_id: {"intime": datetime(2026, 1, 1, 0, 0, 0)}}
    window_cache = defaultdict(dict)
    finalized = set()
    final_attempts = defaultdict(int)
    final_retry_after = {}
    window_last_charttime = {stay_id: {window_id: datetime(2026, 1, 1, 0, 4, 0)}}
    source_modes = {stay_id: "realtime"}
    window_metadata = {stay_id: {window_id: {"source_mode": "realtime", "source_type": "test", "ingestion_time": datetime(2026, 1, 1, 0, 4, 1)}}}
    live_state_manager = LiveState()
    live_chart_batches = defaultdict(dict)
    finalized_risk = {}
    final_payloads = {}
    final_risk_sent = {}
    final_explanation_sent = {}
    last_finalized_window = {}
    error_counts = defaultdict(int)

    first = MODULE.attempt_final(
        stay_id,
        window_id,
        preprocessor,
        producer,
        inference,
        shap_engine,
        static_data,
        window_cache,
        finalized,
        final_attempts,
        final_retry_after,
        window_last_charttime,
        source_modes,
        window_metadata,
        live_state_manager,
        live_chart_batches,
        finalized_risk,
        final_payloads,
        final_risk_sent,
        final_explanation_sent,
        last_finalized_window,
        error_counts
    )

    assert first == "retry"
    assert preprocessor.get_final_window_calls == 1
    assert stay_id not in preprocessor.active_windows
    assert (stay_id, window_id) in final_payloads
    assert (stay_id, window_id) in final_attempts

    inference.fail = False
    final_retry_after[(stay_id, window_id)] = 0.0

    second = MODULE.attempt_final(
        stay_id,
        window_id,
        preprocessor,
        producer,
        inference,
        shap_engine,
        static_data,
        window_cache,
        finalized,
        final_attempts,
        final_retry_after,
        window_last_charttime,
        source_modes,
        window_metadata,
        live_state_manager,
        live_chart_batches,
        finalized_risk,
        final_payloads,
        final_risk_sent,
        final_explanation_sent,
        last_finalized_window,
        error_counts
    )

    assert second == "ok"
    assert preprocessor.get_final_window_calls == 1
    assert (stay_id, window_id) in finalized
    assert (stay_id, window_id) not in final_payloads
    assert final_risk_sent == {}
    assert final_explanation_sent == {}
    print("DESTRUCTIVE FINAL RETRY TEST PASSED")


if __name__ == "__main__":
    test_destructive_final_preprocessing_is_not_repeated()


def test_final_explanation_failure_remains_retryable(monkeypatch):
    monkeypatch.setattr(MODULE, "persist_risk_event_sql", lambda message: True)
    stay_id, window_id = 1001, 1
    preprocessor = Preprocessor()
    producer = Producer()
    inference = Inference()
    inference.fail = False
    static_data = {stay_id: {"intime": datetime(2026, 1, 1, 0, 0, 0)}}
    window_cache = defaultdict(dict)
    finalized = set()
    final_attempts = defaultdict(int)
    final_retry_after = {}
    window_last_charttime = {stay_id: {window_id: datetime(2026, 1, 1, 0, 4, 0)}}
    source_modes = {stay_id: "realtime"}
    window_metadata = {
        stay_id: {
            window_id: {
                "source_mode": "realtime",
                "source_type": "test",
                "ingestion_time": datetime(2026, 1, 1, 0, 4, 1),
            }
        }
    }
    live_state_manager = LiveState()
    live_chart_batches = defaultdict(dict)
    finalized_risk = {}
    final_payloads = {}
    final_risk_sent = {}
    final_explanation_sent = {}
    last_finalized_window = {}
    error_counts = defaultdict(int)

    original_send = producer.send
    explanation_failures = {"remaining": MODULE.MAX_FINAL_ATTEMPTS + 1}

    def flaky_send(topic, key=None, value=None):
        if topic == MODULE.EXPLANATION_TOPIC and explanation_failures["remaining"] > 0:
            explanation_failures["remaining"] -= 1
            raise RuntimeError("intentional SHAP publication failure")
        return original_send(topic, key=key, value=value)

    monkeypatch.setattr(producer, "send", flaky_send)

    args = (
        stay_id, window_id, preprocessor, producer, inference, SHAP(),
        static_data, window_cache, finalized, final_attempts, final_retry_after,
        window_last_charttime, source_modes, window_metadata, live_state_manager,
        live_chart_batches, finalized_risk, final_payloads, final_risk_sent,
        final_explanation_sent, last_finalized_window, error_counts,
    )

        # Exhaust the current retry budget and verify the FINAL stays pending.
    for _ in range(MODULE.MAX_FINAL_ATTEMPTS + 1):
        final_retry_after[(stay_id, window_id)] = 0.0
        result = MODULE.attempt_final(*args)

    assert result == "retry", "failed SHAP delivery must remain retryable"
    assert (stay_id, window_id) not in finalized
    assert (stay_id, window_id) in final_payloads
    assert final_risk_sent[(stay_id, window_id)] is True
    assert final_explanation_sent.get((stay_id, window_id), False) is False
    assert sum(topic == MODULE.RISK_TOPIC for topic, _, _ in producer.messages) == 1

    # A later successful attempt must deliver SHAP without resending risk.
    calls_before_retry = preprocessor.get_final_window_calls
    final_retry_after[(stay_id, window_id)] = 0.0
    result = MODULE.attempt_final(*args)

    assert result == "ok"
    assert preprocessor.get_final_window_calls == calls_before_retry
    assert (stay_id, window_id) in finalized
    assert sum(topic == MODULE.RISK_TOPIC for topic, _, _ in producer.messages) == 1
    assert sum(topic == MODULE.EXPLANATION_TOPIC for topic, _, _ in producer.messages) == 1
def test_idle_sweep_does_not_evict_pending_final_explanation(monkeypatch):
    from collections import defaultdict

    stay_id = 1001
    window_id = 1
    key = (stay_id, window_id)

    preprocessor = Preprocessor()
    producer = Producer()
    inference = Inference()
    shap_engine = SHAP()

    window_cache = defaultdict(dict)
    window_cache[stay_id][window_id] = {"features": np.zeros(137)}
    finalized = set()
    final_attempts = defaultdict(int)
    final_retry_after = {key: float("inf")}
    window_last_charttime = {stay_id: {}}
    max_event_time = {}
    last_seen = {stay_id: (0.0, None)}
    static_data = {stay_id: {"intime": datetime(2026, 1, 1, tzinfo=timezone.utc)}}

    finalized_risk = {}
    final_payloads = {key: {"final_id": f"{stay_id}:{window_id}:FINAL"}}
    final_risk_sent = {key: True}
    final_explanation_sent = {key: False}
    last_finalized_window = {}
    source_modes = {stay_id: "realtime"}
    live_observations = {}
    live_observation_times = {}
    live_chart_batches = defaultdict(dict)
    processed_event_ids = {}
    processed_fallback_keys = {}
    processed_event_order = {}
    processed_fallback_order = {}
    window_metadata = defaultdict(dict)
    live_state_manager = LiveState()
    error_counts = defaultdict(int)
    idle_sweep_cursor = [0]

    monkeypatch.setattr(MODULE, "EVICT_AFTER_SECONDS", 6 * 60 * 60)
    monkeypatch.setattr(MODULE, "IDLE_FINALIZE_SECONDS", 0)
    monkeypatch.setattr(MODULE, "IDLE_SWEEP_BUDGET_SECONDS", 20)
    monkeypatch.setattr(MODULE.time, "monotonic", lambda: 13 * 60 * 60)

    original_attempt_final = MODULE.attempt_final
    monkeypatch.setattr(MODULE, "attempt_final", lambda *args, **kwargs: "retry")

    MODULE.finalize_idle_stays(
        preprocessor, producer, inference, shap_engine, static_data,
        window_cache, last_seen, finalized, final_attempts,
        final_retry_after, idle_sweep_cursor, error_counts,
        window_last_charttime, max_event_time, live_state_manager,
        live_observations, live_observation_times, live_chart_batches,
        finalized_risk, final_payloads, final_risk_sent,
        final_explanation_sent, last_finalized_window, source_modes,
        processed_event_ids, processed_event_order,
        processed_fallback_keys, processed_fallback_order, window_metadata,
    )

    assert stay_id in window_cache
    assert key in final_payloads
    assert final_risk_sent[key] is True
    assert final_explanation_sent[key] is False