import importlib.util
from collections import defaultdict

PATH = "Model/real_time/09_kafka_realtime.py"
spec = importlib.util.spec_from_file_location("kafka_realtime_discharge_test", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class Future:
    def __init__(self, error=None):
        self.error = error

    def get(self, timeout=None):
        if self.error is not None:
            raise self.error
        return None

class Producer:
    def __init__(self):
        self.messages = []

    def send(self, topic, key=None, value=None):
        self.messages.append((topic, key, value))
        return Future()

class Preprocessor:
    def __init__(self):
        self.active_windows = {}

class LiveStateManager:
    def __init__(self):
        self.cleared = []

    def clear_window(self, stay_id, window_id):
        self.cleared.append((stay_id, window_id))

def make_state():
    stay_id = 1001
    static_data = {stay_id: {"intime": "2026-01-01T00:00:00Z"}}
    preprocessor = Preprocessor()
    producer = Producer()
    inference = object()
    shap_engine = object()
    window_cache = defaultdict(dict)
    finalized = set()
    final_attempts = defaultdict(int)
    final_retry_after = {}
    window_last_charttime = {stay_id: {1: "2026-01-01T00:09:00Z"}}
    source_modes = {stay_id: "realtime"}
    window_metadata = defaultdict(dict)
    live_state_manager = LiveStateManager()
    live_chart_batches = defaultdict(dict)
    finalized_risk = {}
    final_payloads = {}
    final_risk_sent = {}
    final_explanation_sent = {}
    last_finalized_window = {}
    error_counts = defaultdict(int)
    return (stay_id, preprocessor, producer, inference, shap_engine, static_data, window_cache, finalized, final_attempts, final_retry_after, window_last_charttime, source_modes, window_metadata, live_state_manager, live_chart_batches, finalized_risk, final_payloads, final_risk_sent, final_explanation_sent, last_finalized_window, error_counts)

def test_active_window_discovery():
    state = make_state()
    stay_id = state[0]
    preprocessor = state[1]
    finalized = state[7]
    calls = []

    preprocessor.active_windows[stay_id] = {1: [{"vital_sign": "heart_rate", "value": 80.0}]}

    original = module.attempt_final

    def attempt(*args):
        calls.append((args[0], args[1]))
        finalized.add((args[0], args[1]))
        return "ok"

    module.attempt_final = attempt
    try:
        module.flush_discharge_finals(*state)
    finally:
        module.attempt_final = original

    if calls != [(stay_id, 1)]:
        raise AssertionError(f"active window was not discovered correctly: {calls}")

def test_failed_window_publishes_final_failed():
    state = make_state()
    stay_id = state[0]
    preprocessor = state[1]
    producer = state[2]
    finalized = state[7]

    preprocessor.active_windows[stay_id] = {1: [{"vital_sign": "heart_rate", "value": 80.0}]}

    original = module.attempt_final

    def attempt(*args):
        return "retry"

    module.attempt_final = attempt
    try:
        module.flush_discharge_finals(*state, hard_deadline=True)
    finally:
        module.attempt_final = original

    failed = [message for message in producer.messages if message[2].get("status") == "FINAL_FAILED"]
    if len(failed) != 1:
        raise AssertionError(f"expected one FINAL_FAILED message, found {len(failed)}")
    payload = failed[0][2]
    if payload["final_id"] != f"{stay_id}:1:FINAL":
        raise AssertionError(f"unexpected final_id: {payload['final_id']}")
    if payload["reason"] != "discharge_flush_incomplete":
        raise AssertionError(f"unexpected failure reason: {payload['reason']}")
    if (stay_id, 1) not in finalized:
        raise AssertionError("failed discharge window was not finalized")

def test_discovery_exception_is_contained():
    state = make_state()
    stay_id = state[0]
    preprocessor = state[1]
    error_counts = state[-1]
    preprocessor.active_windows = []

    module.flush_discharge_finals(*state)

    if error_counts["discharge_flush_errors"] != 1:
        raise AssertionError("discovery exception was not contained and counted")

def test_duplicate_flush_does_not_repeat_finalization():
    state = make_state()
    stay_id = state[0]
    preprocessor = state[1]
    finalized = state[7]
    calls = []

    preprocessor.active_windows[stay_id] = {1: [{"vital_sign": "heart_rate", "value": 80.0}]}

    original = module.attempt_final

    def attempt(*args):
        calls.append((args[0], args[1]))
        finalized.add((args[0], args[1]))
        return "ok"

    module.attempt_final = attempt
    try:
        module.flush_discharge_finals(*state)
        module.flush_discharge_finals(*state)
    finally:
        module.attempt_final = original

    if calls != [(stay_id, 1)]:
        raise AssertionError(f"duplicate discharge flush repeated finalization: {calls}")
def test_pending_final_explanation_is_not_finalized_at_deadline():
    state = make_state()
    stay_id = state[0]
    preprocessor = state[1]
    finalized = state[7]
    final_payloads = state[17]
    final_risk_sent = state[18]
    final_explanation_sent = state[19]

    key = (stay_id, 1)
    preprocessor.active_windows[stay_id] = {
        1: [{"vital_sign": "heart_rate", "value": 80.0}]
    }

    # Risk was published, but its SHAP explanation is still pending.
    final_payloads[key] = {"final_id": f"{stay_id}:1:FINAL"}
    final_risk_sent[key] = True
    final_explanation_sent[key] = False

    original = module.attempt_final

    def attempt(*args):
        return "retry"

    module.attempt_final = attempt
    try:
        module.flush_discharge_finals(*state, hard_deadline=True)
    finally:
        module.attempt_final = original

    assert key not in finalized, (
        "a FINAL with a pending SHAP explanation must not be marked finalized"
    )
    assert key in final_payloads, (
        "pending FINAL payload must be retained for retry"
    )
    assert final_risk_sent[key] is True
    assert final_explanation_sent[key] is False
test_active_window_discovery()
test_failed_window_publishes_final_failed()
test_discovery_exception_is_contained()
test_duplicate_flush_does_not_repeat_finalization()
print("DISCHARGE BEHAVIOR TEST PASSED")
