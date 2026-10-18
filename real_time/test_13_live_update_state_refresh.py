import importlib.util
from pathlib import Path
import numpy as np

PATH = Path(__file__).with_name("13_live_update_state.py")
SPEC = importlib.util.spec_from_file_location("live_update_state_refresh_test", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

def make_state(window_id=1, charttime="2026-01-01T00:00:00Z", feature_value=0.0):
    features = np.zeros(137, dtype=np.float32)
    features[0] = feature_value
    return MODULE.LiveUpdateState(1001, window_id, charttime, 0.2, features, np.zeros(137), 0.25, 0.15, {})

def test_same_timestamp_requires_explicit_refresh():
    manager = MODULE.LiveUpdateStateManager(137)
    first = make_state()
    manager.update(first)
    duplicate, status = manager.check(make_state(feature_value=1.0))
    assert status == "duplicate"
    previous, status = manager.check(make_state(feature_value=1.0), allow_same_charttime_refresh=True)
    assert status == "updated" and previous.charttime == first.charttime
    previous, current, status = manager.update(make_state(feature_value=1.0), allow_same_charttime_refresh=True)
    assert status == "updated" and current.features[0] == 1.0
    assert manager.get(1001).features[0] == 1.0

def test_refresh_does_not_allow_out_of_order_or_stale_windows():
    manager = MODULE.LiveUpdateStateManager(137)
    manager.update(make_state())
    try:
        manager.check(make_state(charttime="2025-12-31T23:59:59Z"), allow_same_charttime_refresh=True)
    except MODULE.OutOfOrderUpdateError:
        pass
    else:
        raise AssertionError("refresh accepted an out-of-order charttime")
    try:
        manager.check(make_state(window_id=0), allow_same_charttime_refresh=True)
    except MODULE.StaleWindowError:
        pass
    else:
        raise AssertionError("refresh accepted a stale window")

def test_refresh_does_not_reopen_closed_window():
    manager = MODULE.LiveUpdateStateManager(137)
    manager.update(make_state())
    manager.clear_window(1001, 1)
    try:
        manager.check(make_state(), allow_same_charttime_refresh=True)
    except MODULE.ClosedWindowError:
        pass
    else:
        raise AssertionError("refresh reopened a closed window")

if __name__ == "__main__":
    test_same_timestamp_requires_explicit_refresh()
    test_refresh_does_not_allow_out_of_order_or_stale_windows()
    test_refresh_does_not_reopen_closed_window()
    print("LIVE UPDATE REFRESH STATE TESTS PASSED")
