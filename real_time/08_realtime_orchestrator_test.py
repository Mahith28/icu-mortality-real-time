import importlib.util
import logging
import sys
import time
from pathlib import Path
import numpy as np
BASE_DIR = Path(__file__).resolve().parent
def load_sibling(module_name, file_name):
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    path = BASE_DIR / file_name
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module
import config
patient_state = load_sibling("patient_state", "04_patient_state.py")
change_detection = load_sibling("change_detection", "07_change_detection.py")
orchestrator_module = load_sibling("realtime_orchestrator", "08_realtime_orchestrator.py")
PatientStateManager = patient_state.PatientStateManager
ChangeDetector = change_detection.ChangeDetector
RealtimeOrchestrator = orchestrator_module.RealtimeOrchestrator
TEMPORAL_FEATURES = config.TEMPORAL_FEATURES
STATIC_FEATURES = config.STATIC_FEATURES
FEATURE_COUNT = config.FEATURE_COUNT
EXPECTED_RESULT_KEYS = {
    "accepted",
    "reason",
    "stay_id",
    "window_id",
    "timestamp",
    "sequence_length",
    "windows_accepted",
    "recovered",
    "inference",
    "change_detection",
    "latency_ms"
}
class FakeInference:
    def __init__(self, probability=0.8, prediction=None):
        self.probability = probability
        self.prediction = prediction
        self.calls = 0
    def predict(self, features, temporal, missing_mask, padding_mask, static):
        self.calls += 1
        prediction = self.prediction
        if prediction is None:
            prediction = int(self.probability >= config.ENSEMBLE_THRESHOLD)
        return {
            "ensemble_probability": self.probability,
            "prediction": prediction,
            "sequence_length": int((padding_mask == 0).sum())
        }
class RecordingInference(FakeInference):
    def __init__(self, probability=0.8, prediction=None):
        super().__init__(probability, prediction)
        self.last = None
    def predict(self, features, temporal, missing_mask, padding_mask, static):
        self.last = {
            "features": features.copy(),
            "temporal": temporal.copy(),
            "missing": missing_mask.copy(),
            "padding": padding_mask.copy(),
            "static": static.copy()
        }
        return super().predict(features, temporal, missing_mask, padding_mask, static)
class MutatingInference(FakeInference):
    def predict(self, features, temporal, missing_mask, padding_mask, static):
        result = super().predict(features, temporal, missing_mask, padding_mask, static)
        for arr in (features, temporal, missing_mask, padding_mask, static):
            if arr.flags.writeable:
                arr[:] = 999.0
        return result
class FailingInference(FakeInference):
    def predict(self, features, temporal, missing_mask, padding_mask, static):
        self.calls += 1
        raise RuntimeError("inference failed")
class SequenceLengthMismatchInference(FakeInference):
    def predict(self, features, temporal, missing_mask, padding_mask, static):
        self.calls += 1
        return {
            "ensemble_probability": 0.8,
            "prediction": 1,
            "sequence_length": int((padding_mask == 0).sum()) + 1
        }
class OverflowSequenceLengthInference(FakeInference):
    def predict(self, features, temporal, missing_mask, padding_mask, static):
        self.calls += 1
        return {
            "ensemble_probability": 0.8,
            "prediction": 1,
            "sequence_length": float("inf")
        }
class MissingProbabilityInference(FakeInference):
    def predict(self, features, temporal, missing_mask, padding_mask, static):
        self.calls += 1
        return {
            "prediction": 1,
            "sequence_length": int((padding_mask == 0).sum())
        }
class SequenceInference(FakeInference):
    def __init__(self, probabilities, threshold):
        super().__init__()
        self.probabilities = list(probabilities)
        self.threshold = threshold
        self.index = 0
    def predict(self, features, temporal, missing_mask, padding_mask, static):
        probability = self.probabilities[min(self.index, len(self.probabilities) - 1)]
        self.index += 1
        return {
            "ensemble_probability": probability,
            "prediction": int(probability >= self.threshold),
            "sequence_length": int((padding_mask == 0).sum())
        }
class FakeLogger:
    def __init__(self):
        self.events = []
    def record(self, event):
        self.events.append(event)
class FailingLogger(FakeLogger):
    def record(self, event):
        raise RuntimeError("logger failed")
class FailingDetector(ChangeDetector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fail_reset_for = set()
    def reset(self, stay_id):
        if stay_id in self.fail_reset_for:
            raise RuntimeError("detector reset failed")
        return super().reset(stay_id)
class FailingManager:
    def add_window(self, *args, **kwargs):
        raise RuntimeError("manager failed")
class BrokenState:
    def get_sequence(self):
        raise RuntimeError("state access failed")
    def get_identity(self):
        raise RuntimeError("state access failed")
    def get_windows_accepted(self):
        raise RuntimeError("state access failed")
class BrokenStateManager:
    def __init__(self):
        self.state = BrokenState()
    def add_window(self, *args, **kwargs):
        return self.state, True
class IdentityMismatchState:
    def get_sequence(self):
        temporal = np.ones((1, TEMPORAL_FEATURES), dtype=np.float32)
        missing_mask = np.zeros((1, TEMPORAL_FEATURES), dtype=np.float32)
        padding_mask = np.zeros(1, dtype=np.float32)
        static = np.ones(STATIC_FEATURES, dtype=np.float32)
        xgb = np.ones(FEATURE_COUNT, dtype=np.float32)
        return temporal, missing_mask, padding_mask, static, 1, xgb
    def get_identity(self):
        return 999, 1000
    def get_windows_accepted(self):
        return 1
class IdentityMismatchManager:
    def add_window(self, *args, **kwargs):
        return IdentityMismatchState(), True
class ExplodingSet(set):
    def discard(self, value):
        raise RuntimeError("discard failed")
def make_window(window_id, stay_id=1, value=1.0, nan_positions=()):
    temporal = np.full(TEMPORAL_FEATURES, value, dtype=np.float32)
    missing_mask = np.zeros(TEMPORAL_FEATURES, dtype=np.float32)
    static = np.arange(STATIC_FEATURES, dtype=np.float32)
    missing_mask[list(nan_positions)] = 1.0
    temporal[list(nan_positions)] = 50.0
    xgb = np.concatenate([np.where(missing_mask == 1.0, np.nan, temporal), static])
    timestamp = 1000 + window_id * 300
    return stay_id, window_id, timestamp, temporal, missing_mask, static, xgb
def build_orchestrator(inference=None, detector=None, logger=None, manager=None):
    if manager is None:
        manager = PatientStateManager(timeout_hours=1)
    if detector is None:
        detector = ChangeDetector()
    if inference is None:
        inference = FakeInference()
    if logger is None:
        logger = FakeLogger()
    return RealtimeOrchestrator(manager, detector, inference, logger)
def assert_result_schema(result):
    assert set(result.keys()) == EXPECTED_RESULT_KEYS
    assert isinstance(result["latency_ms"], float)
    assert result["latency_ms"] >= 0.0
def process(orch, window):
    result = orch.process_window(*window)
    assert_result_schema(result)
    return result
def test_basic_processing():
    orch = build_orchestrator()
    result = process(orch, make_window(0))
    assert result["accepted"] is True
    assert result["reason"] == "processed"
    assert result["sequence_length"] == 1
    assert result["windows_accepted"] == 1
    assert result["inference"]["ensemble_probability"] == 0.8
    print("Basic processing test passed")
def test_duplicate_window():
    orch = build_orchestrator()
    window = make_window(0)
    first = process(orch, window)
    second = process(orch, window)
    assert first["reason"] == "processed"
    assert second["accepted"] is False
    assert second["reason"] == "duplicate"
    print("Duplicate window test passed")
def test_inference_receives_consistent_inputs():
    inference = RecordingInference()
    orch = build_orchestrator(inference=inference)
    result = process(orch, make_window(0, value=3.0))
    assert result["reason"] == "processed"
    last = inference.last
    assert np.all(last["features"][:TEMPORAL_FEATURES] == 3.0)
    assert np.array_equal(last["features"][TEMPORAL_FEATURES:], np.arange(STATIC_FEATURES))
    assert last["temporal"].shape == (120, TEMPORAL_FEATURES)
    assert last["missing"].shape == (120, TEMPORAL_FEATURES)
    assert last["padding"].shape == (120,)
    assert last["static"].shape == (STATIC_FEATURES,)
    assert np.all(last["temporal"][1:] == 0.0)
    assert np.all(last["missing"][1:] == 0.0)
    assert np.all(last["padding"][1:] == 1.0)
    print("Inference input consistency test passed")
def test_rolling_sequence_caps_at_120():
    inference = RecordingInference()
    orch = build_orchestrator(inference=inference)
    for window_id in range(121):
        result = process(orch, make_window(window_id, value=float(window_id + 1)))
        assert result["reason"] == "processed"
    assert result["sequence_length"] == 120
    last = inference.last
    assert np.all(last["temporal"][0] == 2.0)
    assert np.all(last["temporal"][119] == 121.0)
    assert np.all(last["padding"] == 0.0)
    assert np.all(last["missing"] == 0.0)
    print("Rolling sequence 120-window test passed")
def test_nan_preservation():
    inference = RecordingInference()
    orch = build_orchestrator(inference=inference)
    result = process(orch, make_window(0, nan_positions=(2, 10, 20)))
    assert result["reason"] == "processed"
    assert inference.last["missing"][0, 2] == 1.0
    assert inference.last["missing"][0, 10] == 1.0
    assert inference.last["missing"][0, 20] == 1.0
    assert np.isnan(inference.last["features"][2])
    assert np.isnan(inference.last["features"][10])
    assert np.isnan(inference.last["features"][20])
    print("NaN preservation test passed")
def test_state_sequence_array_writeability():
    manager = PatientStateManager(timeout_hours=1)
    state, accepted = manager.add_window(*make_window(0))
    assert accepted is True
    temporal, missing_mask, padding_mask, static, sequence_length, xgb = state.get_sequence()
    copies = [
        temporal.copy(),
        missing_mask.copy(),
        padding_mask.copy(),
        static.copy(),
        xgb.copy()
    ]
    temporal[:] = 777.0
    missing_mask[:] = 777.0
    padding_mask[:] = 777.0
    static[:] = 777.0
    xgb[:] = 777.0
    stored = state.get_sequence()
    assert np.array_equal(stored[0], copies[0])
    assert np.array_equal(stored[1], copies[1])
    assert np.array_equal(stored[2], copies[2])
    assert np.array_equal(stored[3], copies[3])
    assert np.array_equal(stored[5], copies[4])
    print("State sequence mutation safety test passed")
def test_inference_does_not_mutate_state_arrays():
    orch = build_orchestrator(inference=MutatingInference())
    process(orch, make_window(0, value=1.0))
    process(orch, make_window(1, value=2.0))
    rec = RecordingInference()
    orch.inference = rec
    result = process(orch, make_window(2, value=3.0))
    assert result["reason"] == "processed"
    assert result["sequence_length"] == 3
    assert np.all(rec.last["temporal"][0] == 1.0)
    assert np.all(rec.last["temporal"][1] == 2.0)
    assert np.all(rec.last["temporal"][2] == 3.0)
    assert np.array_equal(rec.last["static"], np.arange(STATIC_FEATURES))
    assert np.all(rec.last["features"][:TEMPORAL_FEATURES] == 3.0)
    print("Inference mutation safety test passed")
def test_prediction_mismatch():
    inference = RecordingInference(probability=0.8, prediction=0)
    orch = build_orchestrator(inference=inference)
    result = process(orch, make_window(0))
    detector_result = result["change_detection"]
    assert result["reason"] == "processed"
    assert detector_result["supplied_prediction"] == 0
    assert detector_result["prediction"] == 1
    assert detector_result["prediction_mismatch"] is True
    assert orch.logger.events[-1]["prediction_mismatch"] is True
    print("Prediction mismatch test passed")
def test_stale_window():
    orch = build_orchestrator()
    process(orch, make_window(2))
    result = process(orch, make_window(1))
    assert result["accepted"] is False
    assert result["reason"] == "stale"
    print("Stale window test passed")
def test_conflicting_window():
    orch = build_orchestrator()
    process(orch, make_window(0))
    result = process(orch, make_window(0, value=99.0))
    assert result["accepted"] is False
    assert result["reason"] == "conflict"
    assert orch.error_counters["conflicting_window"] == 1
    print("Conflicting window test passed")
def test_static_changed():
    orch = build_orchestrator()
    process(orch, make_window(0))
    window = list(make_window(1))
    new_static = np.ones(STATIC_FEATURES, dtype=np.float32)
    new_temporal = window[3]
    new_mask = window[4]
    new_xgb = np.concatenate([np.where(new_mask == 1.0, np.nan, new_temporal), new_static])
    window[5] = new_static
    window[6] = new_xgb
    result = process(orch, tuple(window))
    assert result["accepted"] is False
    assert result["reason"] == "static_changed"
    print("Static changed test passed")
def test_timestamp_order():
    orch = build_orchestrator()
    process(orch, make_window(0))
    window = list(make_window(1))
    window[2] = 900
    result = process(orch, tuple(window))
    assert result["accepted"] is False
    assert result["reason"] == "timestamp_order"
    print("Timestamp ordering test passed")
def test_invalid_input_variants():
    cases = []
    window = list(make_window(0))
    window[4] = np.zeros(TEMPORAL_FEATURES - 1, dtype=np.float32)
    cases.append(("missing_mask_length", tuple(window)))
    window = list(make_window(0))
    window[5] = np.zeros(STATIC_FEATURES - 1, dtype=np.float32)
    cases.append(("static_length", tuple(window)))
    window = list(make_window(0))
    window[6] = np.zeros(FEATURE_COUNT - 1, dtype=np.float32)
    cases.append(("xgb_length", tuple(window)))
    window = list(make_window(0))
    window[3][0] = np.nan
    cases.append(("temporal_nan", tuple(window)))
    window = list(make_window(0))
    window[3][0] = np.inf
    cases.append(("temporal_inf", tuple(window)))
    window = list(make_window(0))
    window[3] = np.zeros((1, TEMPORAL_FEATURES), dtype=np.float32)
    cases.append(("temporal_2d", tuple(window)))
    failures = []
    for name, invalid_window in cases:
        orch = build_orchestrator()
        result = process(orch, invalid_window)
        if result["accepted"] is not False or result["reason"] != "invalid":
            failures.append((name, result["reason"]))
    assert not failures, failures
    print("Invalid input variants test passed")
def test_discharged_stay():
    orch = build_orchestrator()
    process(orch, make_window(0))
    orch.discharge(1)
    result = process(orch, make_window(1))
    assert result["accepted"] is False
    assert result["reason"] == "discharged"
    print("Discharged stay test passed")
def test_inference_error():
    inference = FailingInference()
    orch = build_orchestrator(inference=inference)
    result = process(orch, make_window(0))
    assert result["accepted"] is True
    assert result["reason"] == "inference_error"
    assert result["windows_accepted"] == 1
    assert inference.calls == 1
    assert orch.error_counters["inference_error"] == 1
    assert orch.result_counters["inference_error"] == 1
    assert orch.detector.get_state(1) is None
    assert orch.logger.events == []
    print("Inference error test passed")
def test_inference_error_follow_up():
    inference = FakeInference(probability=0.8)
    orch = build_orchestrator(inference=inference)
    class OneShotInference(FakeInference):
        def __init__(self):
            super().__init__()
            self.fail = True
        def predict(self, features, temporal, missing_mask, padding_mask, static):
            if self.fail:
                self.fail = False
                self.calls += 1
                raise RuntimeError("inference failed")
            return super().predict(features, temporal, missing_mask, padding_mask, static)
    inference = OneShotInference()
    orch = build_orchestrator(inference=inference)
    first = process(orch, make_window(0))
    second = process(orch, make_window(1))
    assert first["reason"] == "inference_error"
    assert second["reason"] == "processed"
    assert inference.calls == 2
    print("Inference error follow-up test passed")
def test_temporal_integer_cast():
    orch = build_orchestrator()
    window = list(make_window(0))
    window[3] = np.zeros(TEMPORAL_FEATURES, dtype=np.int64)
    result = process(orch, tuple(window))
    assert result["accepted"] is True
    assert result["reason"] == "processed"
    state = orch.manager.get(1)
    assert state.temporal[-1].dtype == np.float32
    print("Temporal integer cast test passed")
def test_temporal_list_is_coerced():
    inference = RecordingInference()
    orch = build_orchestrator(inference=inference)
    window = list(make_window(0))
    window[3] = [1.0] * TEMPORAL_FEATURES
    result = process(orch, tuple(window))
    assert result["reason"] == "processed"
    assert inference.last["temporal"].dtype == np.float32
    assert inference.last["temporal"].shape == (120, TEMPORAL_FEATURES)
    print("Temporal list coercion test passed")
def test_sequence_length_mismatch():
    orch = build_orchestrator(inference=SequenceLengthMismatchInference())
    result = process(orch, make_window(0))
    assert result["accepted"] is True
    assert result["reason"] == "invalid_inference_output"
    assert orch.error_counters["invalid_inference_output"] == 1
    print("Sequence length mismatch test passed")
def test_sequence_length_overflow():
    orch = build_orchestrator(inference=OverflowSequenceLengthInference())
    result = process(orch, make_window(0))
    assert result["accepted"] is True
    assert result["reason"] == "invalid_inference_output"
    assert orch.error_counters["invalid_inference_output"] == 1
    print("Sequence length overflow test passed")
def test_probability_bounds():
    for probability in (1.5, -0.1, float("inf"), float("-inf"), float("nan")):
        inference = RecordingInference(probability=probability, prediction=1)
        orch = build_orchestrator(inference=inference)
        result = process(orch, make_window(0))
        assert result["accepted"] is True
        assert result["reason"] == "invalid_inference_output"
    print("Probability bounds test passed")
def test_missing_probability():
    orch = build_orchestrator(inference=MissingProbabilityInference())
    result = process(orch, make_window(0))
    assert result["accepted"] is True
    assert result["reason"] == "invalid_inference_output"
    print("Missing probability test passed")
def test_invalid_inference_output_counted_once():
    inference = RecordingInference(probability=1.5, prediction=1)
    orch = build_orchestrator(inference=inference)
    result = process(orch, make_window(0))
    assert result["reason"] == "invalid_inference_output"
    assert orch.result_counters["invalid_inference_output"] == 1
    assert orch.error_counters["invalid_inference_output"] == 1
    print("Invalid inference output counted once test passed")
def test_numpy_prediction_types():
    for prediction in (np.int64(1), np.float32(1.0), True, 0.0):
        inference = RecordingInference(probability=0.8, prediction=prediction)
        orch = build_orchestrator(inference=inference)
        result = process(orch, make_window(0))
        assert result["reason"] == "processed"
    print("NumPy and boolean prediction type test passed")
def test_prediction_invalid():
    inference = RecordingInference(probability=0.8, prediction=2)
    orch = build_orchestrator(inference=inference)
    result = process(orch, make_window(0))
    assert result["accepted"] is True
    assert result["reason"] == "invalid_inference_output"
    print("Invalid prediction test passed")
def test_identity_state_error():
    orch = build_orchestrator(manager=BrokenStateManager())
    result = process(orch, make_window(0))
    assert result["accepted"] is True
    assert result["reason"] == "state_error"
    assert orch.error_counters["state_error"] == 1
    print("State access error test passed")
def test_identity_mismatch():
    orch = build_orchestrator(manager=IdentityMismatchManager())
    result = process(orch, make_window(0))
    assert result["accepted"] is True
    assert result["reason"] == "state_error"
    assert orch.error_counters["state_error"] == 1
    print("Identity mismatch state error test passed")
def test_unexpected_error_from_add_window():
    orch = build_orchestrator(manager=FailingManager())
    result = process(orch, make_window(0))
    assert result["accepted"] is False
    assert result["reason"] == "unexpected_error"
    assert orch.error_counters["unexpected_error"] == 1
    print("Unexpected add_window error test passed")
def test_detector_error():
    class BrokenDetector:
        threshold = config.ENSEMBLE_THRESHOLD
        def update(self, *args, **kwargs):
            raise RuntimeError("detector failed")
        def reset(self, stay_id):
            pass
    orch = build_orchestrator(detector=BrokenDetector())
    result = process(orch, make_window(0))
    assert result["accepted"] is True
    assert result["reason"] == "detector_error"
    assert orch.error_counters["detector_error"] == 1
    print("Detector error test passed")
def test_logger_failure():
    orch = build_orchestrator(logger=FailingLogger())
    result = process(orch, make_window(0))
    assert result["accepted"] is True
    assert result["reason"] == "logger_error"
    assert result["change_detection"] is not None
    assert 1 not in orch.evicted_stays
    print("Logger failure test passed")
def test_evicted_stay_returns_recovered():
    """Characterization test; expected to change if recovery policy changes."""
    inference = FakeInference(probability=0.8)
    orch = build_orchestrator(inference=inference)
    process(orch, make_window(0))
    expired = orch.cleanup(1000 + 3600 + 1)
    assert 1 in expired
    assert 1 in orch.evicted_stays
    result = process(orch, make_window(1))
    assert result["reason"] == "processed"
    assert result["recovered"] is True
    assert result["windows_accepted"] == 1
    assert result["change_detection"]["alert"] is True
    assert result["change_detection"]["alert_reason"] == "initial_high_risk"
    assert orch.logger.events[-1]["recovered"] is True
    assert 1 not in orch.evicted_stays
    print("Evicted stay recovery characterization test passed")
def test_recovered_flag_survives_failed_first_window():
    """Characterization test; expected to change if recovery policy changes."""
    inference = FailingInference()
    orch = build_orchestrator(inference=inference)
    process(orch, make_window(0))
    orch.cleanup(1000 + 3600 + 1)
    result = process(orch, make_window(1))
    assert result["reason"] == "inference_error"
    assert result["recovered"] is True
    assert 1 in orch.evicted_stays
    print("Recovered flag failure characterization test passed")
def test_logger_failure_does_not_lose_recovery():
    orch = build_orchestrator(logger=FailingLogger())
    process(orch, make_window(0))
    orch.cleanup(1000 + 3600 + 1)
    result = process(orch, make_window(1))
    assert result["reason"] == "logger_error"
    assert result["recovered"] is True
    assert 1 not in orch.evicted_stays
    assert orch.detector.get_state(1) is not None
    print("Logger failure recovery test passed")
def test_cleanup():
    manager = PatientStateManager(timeout_hours=1)
    orch = build_orchestrator(manager=manager)
    process(orch, make_window(0, stay_id=1))
    process(orch, make_window(0, stay_id=2))
    expired = orch.cleanup(1000 + 3600 + 1)
    assert set(expired) == {1, 2}
    assert manager.get(1) is None
    assert manager.get(2) is None
    assert orch.detector.get_state(1) is None
    assert orch.detector.get_state(2) is None
    assert orch.event_counters["evicted"] == 2
    assert 1 in orch.evicted_stays
    assert 2 in orch.evicted_stays
    print("Cleanup test passed")
def test_cleanup_just_inside_timeout():
    manager = PatientStateManager(timeout_hours=1)
    orch = build_orchestrator(manager=manager)
    process(orch, make_window(0))
    first_timestamp = 1000
    timeout_seconds = 3600
    expired = orch.cleanup(first_timestamp + timeout_seconds - 1)
    assert not expired
    assert manager.get(1) is not None
    assert orch.detector.get_state(1) is not None
    assert orch.event_counters["evicted"] == 0
    print("Cleanup inside-timeout boundary test passed")
def test_cleanup_exact_timeout():
    manager = PatientStateManager(timeout_hours=1)
    orch = build_orchestrator(manager=manager)
    process(orch, make_window(0))
    first_timestamp = 1000
    timeout_seconds = 3600
    expired = orch.cleanup(first_timestamp + timeout_seconds)
    assert not expired
    assert manager.get(1) is not None
    assert orch.detector.get_state(1) is not None
    assert orch.event_counters["evicted"] == 0
    print("Cleanup exact-timeout boundary test passed")
def test_cleanup_one_second_after_timeout():
    manager = PatientStateManager(timeout_hours=1)
    orch = build_orchestrator(manager=manager)
    process(orch, make_window(0))
    first_timestamp = 1000
    timeout_seconds = 3600
    expired = orch.cleanup(first_timestamp + timeout_seconds + 1)
    assert 1 in expired
    assert manager.get(1) is None
    assert orch.detector.get_state(1) is None
    assert orch.event_counters["evicted"] == 1
    print("Cleanup one-second-after-timeout test passed")
def test_discharge_clears_evicted_marker():
    orch = build_orchestrator()
    process(orch, make_window(0))
    orch.cleanup(1000 + 3600 + 1)
    assert 1 in orch.evicted_stays
    orch.discharge(1)
    assert 1 not in orch.evicted_stays
    assert orch.detector.get_state(1) is None
    print("Discharge clears evicted marker test passed")
def test_detector_reset_error():
    detector = FailingDetector()
    detector.fail_reset_for.add(1)
    orch = build_orchestrator(detector=detector)
    process(orch, make_window(0, stay_id=1))
    process(orch, make_window(0, stay_id=2))
    expired = orch.cleanup(1000 + 3600 + 1)
    assert set(expired) == {1, 2}
    assert 1 in orch.evicted_stays
    assert 2 in orch.evicted_stays
    assert orch.error_counters["detector_reset_error"] == 1
    print("Detector reset error test passed")
def test_unknown_discharge_clears_detector_state():
    orch = build_orchestrator()
    process(orch, make_window(0, stay_id=999))
    assert orch.detector.get_state(999) is not None
    orch.manager.cleanup(1000 + 3600 + 1, evict_active=True)
    assert orch.manager.get(999) is None
    assert orch.detector.get_state(999) is not None
    orch.discharge(999)
    assert orch.detector.get_state(999) is None
    print("Unknown discharge detector cleanup test passed")
def test_interleaved_stays():
    inference = RecordingInference()
    orch = build_orchestrator(inference=inference)
    result1 = process(orch, make_window(0, stay_id=38, value=38.0))
    features38_first = inference.last["features"].copy()
    result2 = process(orch, make_window(10, stay_id=39, value=39.0))
    features39_first = inference.last["features"].copy()
    result3 = process(orch, make_window(1, stay_id=38, value=38.0))
    features38_second = inference.last["features"].copy()
    result4 = process(orch, make_window(11, stay_id=39, value=39.0))
    features39_second = inference.last["features"].copy()
    assert result1["windows_accepted"] == 1
    assert result2["windows_accepted"] == 1
    assert result3["windows_accepted"] == 2
    assert result4["windows_accepted"] == 2
    assert np.all(features38_first[:TEMPORAL_FEATURES] == 38.0)
    assert np.all(features38_second[:TEMPORAL_FEATURES] == 38.0)
    assert np.all(features39_first[:TEMPORAL_FEATURES] == 39.0)
    assert np.all(features39_second[:TEMPORAL_FEATURES] == 39.0)
    state38 = orch.detector.get_state(38)
    state39 = orch.detector.get_state(39)
    assert state38 is not None
    assert state39 is not None
    assert state38["last_window_id"] == 1
    assert state39["last_window_id"] == 11
    print("Interleaved stays test passed")
def test_latency():
    class SlowInference(FakeInference):
        def predict(self, features, temporal, missing_mask, padding_mask, static):
            time.sleep(0.05)
            return super().predict(features, temporal, missing_mask, padding_mask, static)
    orch = build_orchestrator(inference=SlowInference())
    result = process(orch, make_window(0))
    assert result["latency_ms"] >= 45.0
    print("Latency test passed")
def test_threshold_boundary():
    detector = ChangeDetector(threshold=config.ENSEMBLE_THRESHOLD)
    inference = FakeInference(probability=config.ENSEMBLE_THRESHOLD, prediction=1)
    orch = build_orchestrator(inference=inference, detector=detector)
    result = process(orch, make_window(0))
    assert result["reason"] == "processed"
    assert result["inference"]["ensemble_probability"] == config.ENSEMBLE_THRESHOLD
    assert result["change_detection"]["prediction"] == 1
    print("Threshold boundary test passed")
def test_cooldown_flapping():
    threshold = 0.5
    probs = [0.4, 0.6, 0.4, 0.6, 0.4, 0.6]
    detector = ChangeDetector(threshold=threshold, cooldown_windows=2)
    orch = build_orchestrator(inference=SequenceInference(probs, threshold), detector=detector)
    results = [process(orch, make_window(i)) for i in range(6)]
    assert [r["change_detection"]["alert"] for r in results] == [False, True, False, True, False, True]
    assert all(r["change_detection"]["prediction_mismatch"] is False for r in results)
    detector = ChangeDetector(threshold=threshold, cooldown_windows=3)
    orch = build_orchestrator(inference=SequenceInference(probs, threshold), detector=detector)
    results = [process(orch, make_window(i)) for i in range(6)]
    changes = [r["change_detection"] for r in results]
    assert changes[1]["alert"] is True
    assert changes[3]["alert"] is False
    assert changes[3]["alert_reason"] == "threshold_crossing_suppressed"
    assert changes[5]["alert"] is True
    print([(c["alert"], c.get("alert_reason")) for c in changes])
def test_result_schema():
    orch = build_orchestrator()
    results = [
        process(orch, make_window(0)),
        process(orch, make_window(0)),
        process(orch, make_window(2)),
        process(orch, make_window(1))
    ]
    for result in results:
        assert_result_schema(result)
    assert results[0]["reason"] == "processed"
    assert results[1]["reason"] == "duplicate"
    assert results[2]["reason"] == "processed"
    assert results[3]["reason"] == "stale"
    print("Result schema test passed")
def test_result_counter_total():
    orch = build_orchestrator()
    windows = [
        make_window(0),
        make_window(0),
        make_window(1),
        make_window(2)
    ]
    for window in windows:
        process(orch, window)
    assert sum(orch.result_counters.values()) == len(windows)
    print("Result counter total test passed")
def test_unexpected_error_after_acceptance_is_acceptance_unknown():
    """Characterization test; expected to change if acceptance semantics change."""
    orch = build_orchestrator()
    orch.evicted_stays = ExplodingSet()
    result = process(orch, make_window(0))
    assert result["accepted"] is False
    assert result["reason"] == "unexpected_error"
    assert orch.manager.get(1) is not None
    print("Unexpected post-acceptance error characterization test passed")
def run_test(name, function):
    try:
        function()
        return True
    except Exception as exc:
        print(f"{name} FAILED: {type(exc).__name__}: {exc}")
        return False
def main():
    tests = [
        test_basic_processing,
        test_duplicate_window,
        test_inference_receives_consistent_inputs,
        test_rolling_sequence_caps_at_120,
        test_nan_preservation,
        test_state_sequence_array_writeability,
        test_inference_does_not_mutate_state_arrays,
        test_prediction_mismatch,
        test_stale_window,
        test_conflicting_window,
        test_static_changed,
        test_timestamp_order,
        test_invalid_input_variants,
        test_temporal_integer_cast,
        test_discharged_stay,
        test_temporal_list_is_coerced,
        test_inference_error,
        test_inference_error_follow_up,
        test_sequence_length_mismatch,
        test_sequence_length_overflow,
        test_probability_bounds,
        test_missing_probability,
        test_invalid_inference_output_counted_once,
        test_numpy_prediction_types,
        test_prediction_invalid,
        test_identity_state_error,
        test_identity_mismatch,
        test_unexpected_error_from_add_window,
        test_detector_error,
        test_logger_failure,
        test_evicted_stay_returns_recovered,
        test_recovered_flag_survives_failed_first_window,
        test_logger_failure_does_not_lose_recovery,
        test_cleanup,
        test_cleanup_just_inside_timeout,
        test_cleanup_exact_timeout,
        test_cleanup_one_second_after_timeout,
        test_discharge_clears_evicted_marker,
        test_detector_reset_error,
        test_unknown_discharge_clears_detector_state,
        test_interleaved_stays,
        test_latency,
        test_threshold_boundary,
        test_cooldown_flapping,
        test_result_schema,
        test_result_counter_total,
        test_unexpected_error_after_acceptance_is_acceptance_unknown
    ]
    logger = logging.getLogger("realtime_orchestrator")
    previous_level = logger.level
    logger.setLevel(logging.CRITICAL)
    passed = 0
    try:
        for test in tests:
            if run_test(test.__name__, test):
                passed += 1
    finally:
        logger.setLevel(previous_level)
    print(f"Stage D tests: {passed}/{len(tests)} passed")
    if passed != len(tests):
        raise SystemExit(1)
    print("STAGE D REALTIME ORCHESTRATOR TEST PASSED")
if __name__ == "__main__":
    main()