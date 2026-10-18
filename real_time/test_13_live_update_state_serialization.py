"""Expanded serialization/recovery tests for LiveUpdateStateManager."""
import importlib.util
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULE_PATH = Path(__file__).with_name("13_live_update_state.py")
spec = importlib.util.spec_from_file_location("live_update_state_under_test", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

LiveUpdateState = module.LiveUpdateState
LiveUpdateStateManager = module.LiveUpdateStateManager
LiveUpdateStateValidationError = module.LiveUpdateStateValidationError
ClosedWindowError = module.ClosedWindowError
OutOfOrderUpdateError = module.OutOfOrderUpdateError


def make_state(window_id=2, charttime="2026-01-01T10:10:00.123456789Z", stay_id=1001):
    features = [float(i) for i in range(137)]
    features[4] = float("nan")
    return LiveUpdateState(
        stay_id=stay_id, window_id=window_id, charttime=charttime,
        ensemble_probability=0.21, xgb_probability=0.20, lstm_probability=0.22,
        features=features, shap_values=[0.0] * 137,
        observations={"Heart Rate": 88.0, "Temperature": None},
    )


def new_manager_with_state(window_id=2):
    manager = LiveUpdateStateManager()
    manager.update(make_state(window_id=window_id))
    return manager


def test_json_safe_round_trip_and_replay_duplicate():
    original = new_manager_with_state()
    payload = original.to_state_dict()
    # The real store uses allow_nan=False; snapshots must survive strict JSON.
    decoded = json.loads(json.dumps(payload, allow_nan=False))
    restored = LiveUpdateStateManager()
    restored.restore_state_dict(decoded)
    state = restored.get(1001)
    assert state is not None
    assert math.isnan(state.features[4])
    assert state.charttime.isoformat() == "2026-01-01T10:10:00.123456789+00:00"

    previous, status = restored.check(make_state(), allow_same_charttime_refresh=False)
    assert status == "duplicate"
    assert previous is not None

    older = make_state(charttime="2026-01-01T10:10:00.123456788Z")
    try:
        restored.check(older)
    except OutOfOrderUpdateError:
        pass
    else:
        raise AssertionError("older charttime should be rejected after restore")


def test_real_store_end_to_end_if_available():
    store_path = HERE / "14_realtime_state_store.py"
    if not store_path.exists():
        # Allows running this test file separately before the store is copied.
        return
    store_spec = importlib.util.spec_from_file_location("realtime_state_store_under_test", store_path)
    store_module = importlib.util.module_from_spec(store_spec)
    store_spec.loader.exec_module(store_module)

    original = new_manager_with_state()
    with __import__("tempfile").TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "state.db"
        with store_module.RealtimeStateStore(db_path) as store:
            store.save_snapshot({"live_update": original.to_state_dict()})
        with store_module.RealtimeStateStore(db_path) as store:
            loaded = store.load()
        restored = LiveUpdateStateManager()
        restored.restore_state_dict(loaded.snapshot["live_update"])
        assert restored.get(1001).window_id == 2
        assert math.isnan(restored.get(1001).features[4])


def test_closed_window_cross_check_is_atomic():
    manager = new_manager_with_state(window_id=2)
    before = json.dumps(manager.to_state_dict(), allow_nan=False, sort_keys=True)
    invalid = manager.to_state_dict()
    invalid["last_closed_window"] = {"1001": 2}
    try:
        manager.restore_state_dict(invalid)
    except LiveUpdateStateValidationError:
        pass
    else:
        raise AssertionError("active state at its closed window should be rejected")
    after = json.dumps(manager.to_state_dict(), allow_nan=False, sort_keys=True)
    assert after == before


def test_feature_length_mismatch_rejected_atomically():
    manager = new_manager_with_state()
    before = json.dumps(manager.to_state_dict(), allow_nan=False, sort_keys=True)
    invalid = manager.to_state_dict()
    invalid["feature_lengths"]["1001"] = 136
    try:
        manager.restore_state_dict(invalid)
    except LiveUpdateStateValidationError:
        pass
    else:
        raise AssertionError("feature-length mismatch should be rejected")
    assert json.dumps(manager.to_state_dict(), allow_nan=False, sort_keys=True) == before


def test_expected_feature_count_mismatch_rejected():
    manager = new_manager_with_state()
    payload = manager.to_state_dict()
    wrong_manager = LiveUpdateStateManager(expected_n_features=136)
    try:
        wrong_manager.restore_state_dict(payload)
    except LiveUpdateStateValidationError:
        pass
    else:
        raise AssertionError("expected_n_features mismatch should be rejected")


def test_restore_replaces_existing_state():
    manager = new_manager_with_state(window_id=2)
    replacement = LiveUpdateStateManager()
    replacement.update(make_state(window_id=4, charttime="2026-01-01T10:20:00Z", stay_id=2002))
    manager.restore_state_dict(replacement.to_state_dict())
    assert manager.active_stays() == ("2002",)
    assert manager.get(1001) is None
    assert manager.get(2002).window_id == 4


def test_strict_schema_version_and_state_keys():
    manager = new_manager_with_state()
    payload = manager.to_state_dict()
    invalid_version = manager.to_state_dict()
    invalid_version["schema_version"] = True
    invalid_state_keys = manager.to_state_dict()
    invalid_state_keys["states"]["1001"]["extra"] = "not allowed"
    for invalid in (invalid_version, invalid_state_keys):
        try:
            manager.restore_state_dict(invalid)
        except LiveUpdateStateValidationError:
            pass
        else:
            raise AssertionError("strict snapshot validation should reject malformed payload")


def main():
    tests = [
        test_json_safe_round_trip_and_replay_duplicate,
        test_real_store_end_to_end_if_available,
        test_closed_window_cross_check_is_atomic,
        test_feature_length_mismatch_rejected_atomically,
        test_expected_feature_count_mismatch_rejected,
        test_restore_replaces_existing_state,
        test_strict_schema_version_and_state_keys,
    ]
    for test in tests:
        test()
    print(f"LIVE UPDATE STATE SERIALIZATION TESTS PASSED ({len(tests)} tests)")


if __name__ == "__main__":
    main()
