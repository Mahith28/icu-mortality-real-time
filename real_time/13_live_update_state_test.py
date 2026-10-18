import sys
from pathlib import Path
from importlib import import_module
from datetime import datetime
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
mod = import_module("13_live_update_state")
LiveUpdateState = mod.LiveUpdateState
LiveUpdateStateManager = mod.LiveUpdateStateManager
LiveUpdateStateValidationError = mod.LiveUpdateStateValidationError
OutOfOrderUpdateError = mod.OutOfOrderUpdateError
StaleWindowError = mod.StaleWindowError
ClosedWindowError = mod.ClosedWindowError
def assert_raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type:
        return
    raise AssertionError(f"{exc_type.__name__} not raised")
def utc(minute):
    return pd.Timestamp(2026, 1, 1, 10, minute, 0, tz="UTC")
def make_features(value=1.0):
    return np.full(137, value, dtype=np.float32)
def make_shap(value=0.1):
    return np.full(137, value, dtype=np.float32)
def make_observations(value=80.0):
    return {"Heart Rate": value, "SBP": 120.0, "DBP": 70.0}
def make_state(stay_id=1001, window_id=0, minute=2, value=0.1, **overrides):
    kwargs = dict(
        stay_id=stay_id,
        window_id=window_id,
        charttime=datetime(2026, 1, 1, 10, minute, 0),
        ensemble_probability=value,
        xgb_probability=value - 0.01,
        lstm_probability=value + 0.01,
        features=make_features(value),
        shap_values=make_shap(value),
        observations=make_observations(80.0 + minute),
    )
    kwargs.update(overrides)
    return LiveUpdateState(**kwargs)
def make_update(manager, **kw):
    return manager.update(make_state(**kw))
def test_first_update():
    m = LiveUpdateStateManager(137)
    previous, current, status = make_update(m, minute=2, value=0.10)
    assert previous is None
    assert current.stay_id == "1001"
    assert current.window_id == 0
    assert current.ensemble_probability == 0.10
    assert status == "updated"
def test_second_update_same_window():
    m = LiveUpdateStateManager(137)
    make_update(m, minute=2, value=0.10)
    previous, current, status = make_update(m, minute=3, value=0.20)
    assert previous is not None
    assert previous.ensemble_probability == 0.10
    assert current.ensemble_probability == 0.20
    assert status == "updated"
def test_duplicate_charttime():
    m = LiveUpdateStateManager(137)
    make_update(m, minute=2, value=0.10)
    previous, current, status = make_update(m, minute=2, value=0.20)
    assert previous is None
    assert status == "duplicate"
    assert current.ensemble_probability == 0.10
    assert m.get(1001).ensemble_probability == 0.10
def test_out_of_order_charttime():
    m = LiveUpdateStateManager(137)
    make_update(m, minute=4, value=0.20)
    assert_raises(OutOfOrderUpdateError, make_update, m, minute=3, value=0.10)
    assert m.get(1001).ensemble_probability == 0.20
def test_stale_window_rejected():
    m = LiveUpdateStateManager(137)
    make_update(m, window_id=2, minute=2)
    assert_raises(StaleWindowError, make_update, m, window_id=1, minute=3)
    assert m.get(1001).window_id == 2
def test_new_window_replaces_previous_state():
    m = LiveUpdateStateManager(137)
    make_update(m, window_id=0, minute=4, value=0.20)
    previous, current, status = make_update(m, window_id=1, minute=2, value=0.30)
    assert previous is None
    assert current.window_id == 1
    assert status == "updated"
    assert m.get(1001).window_id == 1
def test_late_update_after_close_rejected():
    m = LiveUpdateStateManager(137)
    make_update(m, window_id=0, minute=2)
    assert m.clear_window(1001, 0) is True
    assert_raises(ClosedWindowError, make_update, m, window_id=0, minute=3)
def test_feature_length_mismatch():
    m = LiveUpdateStateManager(137)
    bad = make_state(features=np.zeros(136), shap_values=np.zeros(136))
    assert_raises(LiveUpdateStateValidationError, m.update, bad)
    assert m.get(1001) is None
def test_feature_shap_length_mismatch():
    assert_raises(
        LiveUpdateStateValidationError,
        make_state,
        shap_values=np.zeros(136),
    )
def test_state_is_not_mutated_by_invalid_update():
    m = LiveUpdateStateManager(137)
    make_update(m, minute=2, value=0.20)
    bad = make_state(
        minute=3,
        features=np.zeros(136),
        shap_values=np.zeros(136),
    )
    assert_raises(LiveUpdateStateValidationError, m.update, bad)
    state = m.get(1001)
    assert state.ensemble_probability == 0.20
    assert state.charttime == utc(2)
def test_shap_nan_rejected():
    shap = make_shap()
    shap[54] = np.nan
    assert_raises(
        LiveUpdateStateValidationError,
        make_state,
        shap_values=shap,
    )
def test_probability_bounds():
    assert_raises(
        LiveUpdateStateValidationError,
        make_state,
        ensemble_probability=1.1,
    )
def test_boolean_window_id_rejected():
    assert_raises(
        LiveUpdateStateValidationError,
        make_state,
        window_id=True,
    )
def test_nan_features_allowed():
    m = LiveUpdateStateManager(137)
    features = make_features()
    features[54] = np.nan
    previous, current, status = m.update(make_state(features=features))
    assert previous is None
    assert status == "updated"
    assert np.isnan(current.features[54])
def test_ensemble_probability_required():
    assert_raises(
        LiveUpdateStateValidationError,
        make_state,
        ensemble_probability=None,
    )
def test_optional_model_probabilities():
    state = make_state(
        xgb_probability=None,
        lstm_probability=None,
    )
    assert state.xgb_probability is None
    assert state.lstm_probability is None
def test_naive_and_utc_equal():
    m = LiveUpdateStateManager(137)
    m.update(make_state(minute=2))
    _, _, status = m.update(
        make_state(
            minute=2,
            charttime=utc(2),
        )
    )
    assert status == "duplicate"
def test_final_for_newer_window_closes_it():
    m = LiveUpdateStateManager(137)
    make_update(m, window_id=0, minute=2)
    assert m.clear_window(1001, 1) is True
    assert_raises(
        ClosedWindowError,
        make_update,
        m,
        window_id=1,
        minute=3,
    )
def test_final_without_update_closes_window():
    m = LiveUpdateStateManager(137)
    assert m.clear_window(1001, 0) is False
    assert_raises(
        ClosedWindowError,
        make_update,
        m,
        window_id=0,
        minute=2,
    )
def test_window_rollover_after_final():
    m = LiveUpdateStateManager(137)
    make_update(m, window_id=0, minute=4)
    assert m.clear_window(1001, 0) is True
    previous, current, status = make_update(
        m,
        window_id=1,
        minute=2,
    )
    assert previous is None
    assert current.window_id == 1
    assert status == "updated"
def test_numpy_window_id():
    m = LiveUpdateStateManager(137)
    _, current, _ = make_update(
        m,
        window_id=np.int64(3),
    )
    assert current.window_id == 3
def test_stay_id_normalization():
    m = LiveUpdateStateManager(137)
    make_update(m, stay_id=1001.0)
    assert m.get("1001").stay_id == "1001"
    assert_raises(
        LiveUpdateStateValidationError,
        make_state,
        stay_id=float("nan"),
    )
def test_probability_clipped():
    assert make_state(
        ensemble_probability=1.0000001
    ).ensemble_probability == 1.0
def test_inf_feature_rejected():
    features = make_features()
    features[0] = np.inf
    assert_raises(
        LiveUpdateStateValidationError,
        make_state,
        features=features,
    )
def test_returned_state_is_a_copy():
    m = LiveUpdateStateManager(137)
    _, current, _ = make_update(m, minute=2)
    current.features[0] = 999.0
    assert m.get(1001).features[0] != 999.0
def test_clearing_one_stay_keeps_another():
    m = LiveUpdateStateManager(137)
    make_update(m, stay_id=1001, minute=2)
    make_update(m, stay_id=1002, minute=2)
    m.clear_window(1001, 0)
    assert m.get(1001) is None
    assert m.get(1002) is not None
def test_get():
    m = LiveUpdateStateManager(137)
    make_update(m, window_id=2, minute=2)
    state = m.get(1001)
    assert state is not None
    assert state.window_id == 2
def test_get_missing():
    m = LiveUpdateStateManager(137)
    assert m.get(1001) is None
def test_contains():
    m = LiveUpdateStateManager(137)
    assert m.contains(1001) is False
    make_update(m)
    assert m.contains(1001) is True
def test_contains_after_clear():
    m = LiveUpdateStateManager(137)
    make_update(m)
    m.clear_window(1001, 0)
    assert m.contains(1001) is False
def test_clear_all():
    m = LiveUpdateStateManager(137)
    make_update(m, stay_id=1001)
    make_update(m, stay_id=1002)
    assert m.size() == 2
    m.clear_all()
    assert m.size() == 0
    assert m.get(1001) is None
    assert m.get(1002) is None
def test_reset():
    m = LiveUpdateStateManager(137)
    make_update(m)
    result = m.reset(1001)
    assert result is None
    assert m.get(1001) is None
    assert m.contains(1001) is False
def test_reset_allows_window_reuse():
    m = LiveUpdateStateManager(137)
    make_update(m, window_id=2)
    assert m.clear_window(1001, 2) is True
    assert_raises(
        ClosedWindowError,
        make_update,
        m,
        window_id=2,
        minute=3,
    )
    m.reset(1001)
    previous, current, status = make_update(
        m,
        window_id=2,
        minute=3,
    )
    assert previous is None
    assert current.window_id == 2
    assert status == "updated"
def test_clear_window_missing_state():
    m = LiveUpdateStateManager(137)
    assert m.clear_window(1001, 0) is False
    assert m.get(1001) is None
    assert_raises(
        ClosedWindowError,
        make_update,
        m,
        window_id=0,
        minute=2,
    )
def test_clear_window_older_than_active_state():
    m = LiveUpdateStateManager(137)
    make_update(m, window_id=2, minute=2)
    assert m.clear_window(1001, 1) is False
    assert m.get(1001).window_id == 2
def test_clear_window_newer_than_active_state():
    m = LiveUpdateStateManager(137)
    make_update(m, window_id=2, minute=2)
    assert m.clear_window(1001, 3) is True
    assert m.get(1001) is None
    assert_raises(
        ClosedWindowError,
        make_update,
        m,
        window_id=3,
        minute=4,
    )
def test_active_stays():
    m = LiveUpdateStateManager(137)
    make_update(m, stay_id=1001)
    make_update(m, stay_id=1002)
    assert set(m.active_stays()) == {"1001", "1002"}
def test_size():
    m = LiveUpdateStateManager(137)
    assert m.size() == 0
    make_update(m, stay_id=1001)
    assert m.size() == 1
    make_update(m, stay_id=1002)
    assert m.size() == 2
    m.clear_window(1001, 0)
    assert m.size() == 1
def test_evict():
    m = LiveUpdateStateManager(137)
    make_update(m, stay_id=1001)
    assert m.evict(1001) is True
    assert m.get(1001) is None
    assert m.contains(1001) is False
def test_evict_missing():
    m = LiveUpdateStateManager(137)
    assert m.evict(1001) is False
def test_evict_closed_state():
    m = LiveUpdateStateManager(137)
    make_update(m, stay_id=1001, window_id=2)
    assert m.clear_window(1001, 2) is True
    assert m.evict(1001) is True
def test_state_to_dict():
    state = make_state(
        stay_id=1001,
        window_id=2,
        minute=2,
        value=0.20,
    )
    result = state.to_dict()
    assert result["stay_id"] == "1001"
    assert result["window_id"] == 2
    assert result["charttime"] == "2026-01-01T10:02:00+00:00"
    assert result["ensemble_probability"] == 0.20
    assert abs(result["xgb_probability"] - 0.19) < 1e-9
    assert abs(result["lstm_probability"] - 0.21) < 1e-9
    assert len(result["features"]) == 137
    assert len(result["shap_values"]) == 137
    assert result["observations"]["Heart Rate"] == 82.0
def test_observation_none_allowed():
    state = make_state(
        observations={
            "Heart Rate": None,
            "SBP": 120.0,
        }
    )
    assert state.observations["Heart Rate"] is None
    assert state.observations["SBP"] == 120.0
def test_observation_nan_rejected():
    assert_raises(
        LiveUpdateStateValidationError,
        make_state,
        observations={"Heart Rate": np.nan},
    )
def test_observation_inf_rejected():
    assert_raises(
        LiveUpdateStateValidationError,
        make_state,
        observations={"Heart Rate": np.inf},
    )
def test_observation_bool_rejected():
    assert_raises(
        LiveUpdateStateValidationError,
        make_state,
        observations={"Heart Rate": True},
    )
def test_invalid_stay_id():
    assert_raises(
        LiveUpdateStateValidationError,
        make_state,
        stay_id=None,
    )
def test_empty_stay_id():
    assert_raises(
        LiveUpdateStateValidationError,
        make_state,
        stay_id="   ",
    )
def test_boolean_stay_id_rejected():
    assert_raises(
        LiveUpdateStateValidationError,
        make_state,
        stay_id=True,
    )
def test_invalid_window_id():
    assert_raises(
        LiveUpdateStateValidationError,
        make_state,
        window_id=-1,
    )
def test_invalid_window_id_string():
    assert_raises(
        LiveUpdateStateValidationError,
        make_state,
        window_id="abc",
    )
def test_invalid_window_id_decimal_string():
    assert_raises(
        LiveUpdateStateValidationError,
        make_state,
        window_id="3.0",
    )
def test_float_window_id():
    m = LiveUpdateStateManager(137)
    _, current, status = make_update(
        m,
        window_id=3.0,
    )
    assert current.window_id == 3
    assert status == "updated"
def test_charttime_string():
    state = make_state(
        charttime="2026-01-01T10:02:00Z",
    )
    assert state.charttime == utc(2)
def test_charttime_numpy_datetime():
    state = make_state(
        charttime=np.datetime64("2026-01-01T10:02:00"),
    )
    assert state.charttime == utc(2)
def test_invalid_charttime_number():
    assert_raises(
        LiveUpdateStateValidationError,
        make_state,
        charttime=12345,
    )
def test_invalid_charttime_string():
    assert_raises(
        LiveUpdateStateValidationError,
        make_state,
        charttime="not-a-date",
    )
def test_charttime_nat_rejected():
    assert_raises(
        LiveUpdateStateValidationError,
        make_state,
        charttime=pd.NaT,
    )
def test_empty_features_rejected():
    assert_raises(
        LiveUpdateStateValidationError,
        make_state,
        features=[],
        shap_values=[],
    )
def test_empty_shap_rejected():
    assert_raises(
        LiveUpdateStateValidationError,
        make_state,
        shap_values=[],
        features=[],
    )
def test_none_feature_becomes_nan():
    features = make_features()
    features = list(features)
    features[10] = None
    state = make_state(features=features)
    assert np.isnan(state.features[10])
def test_none_shap_rejected():
    shap = make_shap()
    shap = list(shap)
    shap[10] = None
    assert_raises(
        LiveUpdateStateValidationError,
        make_state,
        shap_values=shap,
    )
def test_boolean_feature_rejected():
    features = make_features()
    features = list(features)
    features[10] = True
    assert_raises(
        LiveUpdateStateValidationError,
        make_state,
        features=features,
    )
def test_boolean_shap_rejected():
    shap = make_shap()
    shap = list(shap)
    shap[10] = True
    assert_raises(
        LiveUpdateStateValidationError,
        make_state,
        shap_values=shap,
    )
def test_probability_nan_rejected():
    assert_raises(
        LiveUpdateStateValidationError,
        make_state,
        ensemble_probability=np.nan,
    )
def test_probability_inf_rejected():
    assert_raises(
        LiveUpdateStateValidationError,
        make_state,
        ensemble_probability=np.inf,
    )
def test_probability_negative_tolerance_clipped():
    state = make_state(
        ensemble_probability=-0.0000001,
    )
    assert state.ensemble_probability == 0.0
def test_probability_negative_outside_tolerance_rejected():
    assert_raises(
        LiveUpdateStateValidationError,
        make_state,
        ensemble_probability=-0.001,
    )
def test_probability_upper_tolerance_clipped():
    state = make_state(
        ensemble_probability=1.0000001,
    )
    assert state.ensemble_probability == 1.0
def test_probability_upper_outside_tolerance_rejected():
    assert_raises(
        LiveUpdateStateValidationError,
        make_state,
        ensemble_probability=1.001,
    )
def test_xgb_probability_optional():
    state = make_state(
        xgb_probability=None,
    )
    assert state.xgb_probability is None
def test_lstm_probability_optional():
    state = make_state(
        lstm_probability=None,
    )
    assert state.lstm_probability is None
def test_expected_feature_count():
    m = LiveUpdateStateManager(137)
    state = make_state()
    previous, current, status = m.update(state)
    assert previous is None
    assert len(current.features) == 137
    assert status == "updated"
def test_feature_length_change_across_windows():
    m = LiveUpdateStateManager(137)
    make_update(m, window_id=0, minute=2)
    bad = make_state(
        window_id=1,
        minute=2,
        features=np.zeros(136),
        shap_values=np.zeros(136),
    )
    assert_raises(
        LiveUpdateStateValidationError,
        m.update,
        bad,
    )
    assert m.get(1001).window_id == 0
def test_observations_are_copied():
    observations = {
        "Heart Rate": 80.0,
        "SBP": 120.0,
    }
    state = make_state(observations=observations)
    observations["Heart Rate"] = 999.0
    assert state.observations["Heart Rate"] == 80.0
def test_features_are_copied():
    features = make_features()
    state = make_state(features=features)
    features[0] = 999.0
    assert state.features[0] != 999.0
def test_shap_values_are_copied():
    shap = make_shap()
    state = make_state(shap_values=shap)
    shap[0] = 999.0
    assert state.shap_values[0] != 999.0
def test_to_dict_returns_copies():
    state = make_state()
    result = state.to_dict()
    result["features"][0] = 999.0
    result["shap_values"][0] = 999.0
    result["observations"]["Heart Rate"] = 999.0
    assert state.features[0] != 999.0
    assert state.shap_values[0] != 999.0
    assert state.observations["Heart Rate"] != 999.0
def test_duplicate_does_not_replace_state():
    m = LiveUpdateStateManager(137)
    make_update(m, minute=2, value=0.10)
    make_update(m, minute=2, value=0.90)
    state = m.get(1001)
    assert state.ensemble_probability == 0.10
    assert state.charttime == utc(2)
def test_new_window_does_not_return_previous_window_as_delta():
    m = LiveUpdateStateManager(137)
    make_update(m, window_id=0, minute=4, value=0.20)
    previous, current, status = make_update(
        m,
        window_id=1,
        minute=2,
        value=0.30,
    )
    assert previous is None
    assert current.ensemble_probability == 0.30
    assert status == "updated"
def test_clear_window_records_closed_window():
    m = LiveUpdateStateManager(137)
    assert m.clear_window(1001, 5) is False
    assert_raises(
        ClosedWindowError,
        make_update,
        m,
        window_id=5,
        minute=2,
    )
def test_clear_window_lower_than_closed_window():
    m = LiveUpdateStateManager(137)
    assert m.clear_window(1001, 5) is False
    assert m.clear_window(1001, 4) is False
def test_clear_window_same_as_closed_window():
    m = LiveUpdateStateManager(137)
    assert m.clear_window(1001, 5) is False
    assert m.clear_window(1001, 5) is False
def test_clear_window_active_same_window():
    m = LiveUpdateStateManager(137)
    make_update(m, window_id=5, minute=2)
    assert m.clear_window(1001, 5) is True
    assert m.get(1001) is None
def test_clear_window_active_newer_window():
    m = LiveUpdateStateManager(137)
    make_update(m, window_id=5, minute=2)
    assert m.clear_window(1001, 6) is True
    assert m.get(1001) is None
def test_clear_window_active_older_window():
    m = LiveUpdateStateManager(137)
    make_update(m, window_id=5, minute=2)
    assert m.clear_window(1001, 4) is False
    assert m.get(1001) is not None
def test_closed_window_blocks_equal_window():
    m = LiveUpdateStateManager(137)
    make_update(m, window_id=5, minute=2)
    m.clear_window(1001, 5)
    assert_raises(
        ClosedWindowError,
        make_update,
        m,
        window_id=5,
        minute=3,
    )
def test_closed_window_blocks_older_window():
    m = LiveUpdateStateManager(137)
    make_update(m, window_id=5, minute=2)
    m.clear_window(1001, 5)
    assert_raises(
        ClosedWindowError,
        make_update,
        m,
        window_id=4,
        minute=3,
    )
def test_newer_window_after_closed_window_allowed():
    m = LiveUpdateStateManager(137)
    make_update(m, window_id=5, minute=2)
    m.clear_window(1001, 5)
    previous, current, status = make_update(
        m,
        window_id=6,
        minute=2,
    )
    assert previous is None
    assert current.window_id == 6
    assert status == "updated"
def test_evict_active_state():
    m = LiveUpdateStateManager(137)
    make_update(m, window_id=5, minute=2)
    assert m.evict(1001) is True
    assert m.get(1001) is None
    previous, current, status = make_update(
        m,
        window_id=5,
        minute=3,
    )
    assert previous is None
    assert current.window_id == 5
    assert status == "updated"
def test_evict_closed_state_removes_closed_marker():
    m = LiveUpdateStateManager(137)
    make_update(m, window_id=5, minute=2)
    m.clear_window(1001, 5)
    assert_raises(
        ClosedWindowError,
        make_update,
        m,
        window_id=5,
        minute=3,
    )
    assert m.evict(1001) is True
    previous, current, status = make_update(
        m,
        window_id=5,
        minute=3,
    )
    assert previous is None
    assert current.window_id == 5
    assert status == "updated"
def test_clear_all_removes_closed_markers():
    m = LiveUpdateStateManager(137)
    make_update(m, window_id=5, minute=2)
    m.clear_window(1001, 5)
    assert_raises(
        ClosedWindowError,
        make_update,
        m,
        window_id=5,
        minute=3,
    )
    m.clear_all()
    previous, current, status = make_update(
        m,
        window_id=5,
        minute=3,
    )
    assert previous is None
    assert current.window_id == 5
    assert status == "updated"
def test_feature_length_locked_per_stay():
    m = LiveUpdateStateManager(None)
    first = make_state()
    previous, current, status = m.update(first)
    assert previous is None
    assert current is not None
    assert status == "updated"
    second = make_state(
        window_id=1,
        features=np.zeros(137),
        shap_values=np.zeros(137),
    )
    previous, current, status = m.update(second)
    assert previous is None
    assert current.window_id == 1
    assert status == "updated"
    bad = make_state(
        window_id=2,
        features=np.zeros(136),
        shap_values=np.zeros(136),
    )
    assert_raises(
        LiveUpdateStateValidationError,
        m.update,
        bad,
    )
def test_expected_feature_count_validation():
    assert_raises(
        LiveUpdateStateValidationError,
        LiveUpdateStateManager,
        0,
    )
    assert_raises(
        LiveUpdateStateValidationError,
        LiveUpdateStateManager,
        -1,
    )
    assert_raises(
        LiveUpdateStateValidationError,
        LiveUpdateStateManager,
        True,
    )
def test_manager_none_feature_count():
    m = LiveUpdateStateManager(None)
    state = make_state()
    previous, current, status = m.update(state)
    assert previous is None
    assert current is not None
    assert status == "updated"
def test_active_stays_after_reset():
    m = LiveUpdateStateManager(137)
    make_update(m, stay_id=1001)
    make_update(m, stay_id=1002)
    m.reset(1001)
    assert m.active_stays() == ("1002",)
def test_active_stays_after_evict():
    m = LiveUpdateStateManager(137)
    make_update(m, stay_id=1001)
    make_update(m, stay_id=1002)
    m.evict(1001)
    assert m.active_stays() == ("1002",)
def test_manager_stay_id_normalization():
    m = LiveUpdateStateManager(137)
    make_update(m, stay_id=np.float32(1001.0))
    assert m.contains("1001")
def test_stay_id_string():
    state = make_state(stay_id="abc")
    assert state.stay_id == "abc"
def test_stay_id_integral():
    state = make_state(stay_id=np.int64(1001))
    assert state.stay_id == "1001"
def test_stay_id_non_integer_float():
    state = make_state(stay_id=1001.5)
    assert state.stay_id == "1001.5"
def test_feature_nan_preserved():
    features = make_features()
    features[0] = np.nan
    state = make_state(features=features)
    assert np.isnan(state.features[0])
def test_shap_inf_rejected():
    shap = make_shap()
    shap[0] = np.inf
    assert_raises(
        LiveUpdateStateValidationError,
        make_state,
        shap_values=shap,
    )
def test_feature_negative_inf_rejected():
    features = make_features()
    features[0] = -np.inf
    assert_raises(
        LiveUpdateStateValidationError,
        make_state,
        features=features,
    )
def test_observations_are_numeric():
    state = make_state(
        observations={
            "Heart Rate": 80,
            "SBP": 120.5,
        }
    )
    assert state.observations["Heart Rate"] == 80.0
    assert state.observations["SBP"] == 120.5
def test_to_dict_charttime_is_iso_safe():
    state = make_state(
        charttime=pd.Timestamp(
            "2026-01-01T10:02:00",
            tz="Asia/Kolkata",
        )
    )
    result = state.to_dict()
    assert result["charttime"] == "2026-01-01T04:32:00+00:00"
def test_get_returns_copy():
    m = LiveUpdateStateManager(137)
    make_update(m)
    first = m.get(1001)
    first.features[0] = 999.0
    second = m.get(1001)
    assert second.features[0] != 999.0
def test_duplicate_returns_copy():
    m = LiveUpdateStateManager(137)
    make_update(m)
    _, duplicate, status = make_update(m, minute=2, value=0.90)
    assert status == "duplicate"
    duplicate.features[0] = 999.0
    assert m.get(1001).features[0] != 999.0
def main():
    tests = [
        test_first_update,
        test_second_update_same_window,
        test_duplicate_charttime,
        test_out_of_order_charttime,
        test_stale_window_rejected,
        test_new_window_replaces_previous_state,
        test_late_update_after_close_rejected,
        test_feature_length_mismatch,
        test_feature_shap_length_mismatch,
        test_state_is_not_mutated_by_invalid_update,
        test_shap_nan_rejected,
        test_probability_bounds,
        test_boolean_window_id_rejected,
        test_nan_features_allowed,
        test_ensemble_probability_required,
        test_optional_model_probabilities,
        test_naive_and_utc_equal,
        test_final_for_newer_window_closes_it,
        test_final_without_update_closes_window,
        test_window_rollover_after_final,
        test_numpy_window_id,
        test_stay_id_normalization,
        test_probability_clipped,
        test_inf_feature_rejected,
        test_returned_state_is_a_copy,
        test_clearing_one_stay_keeps_another,
        test_get,
        test_get_missing,
        test_contains,
        test_contains_after_clear,
        test_clear_all,
        test_reset,
        test_reset_allows_window_reuse,
        test_clear_window_missing_state,
        test_clear_window_older_than_active_state,
        test_clear_window_newer_than_active_state,
        test_active_stays,
        test_size,
        test_evict,
        test_evict_missing,
        test_evict_closed_state,
        test_state_to_dict,
        test_observation_none_allowed,
        test_observation_nan_rejected,
        test_observation_inf_rejected,
        test_observation_bool_rejected,
        test_invalid_stay_id,
        test_empty_stay_id,
        test_boolean_stay_id_rejected,
        test_invalid_window_id,
        test_invalid_window_id_string,
        test_invalid_window_id_decimal_string,
        test_float_window_id,
        test_charttime_string,
        test_charttime_numpy_datetime,
        test_invalid_charttime_number,
        test_invalid_charttime_string,
        test_charttime_nat_rejected,
        test_empty_features_rejected,
        test_empty_shap_rejected,
        test_none_feature_becomes_nan,
        test_none_shap_rejected,
        test_boolean_feature_rejected,
        test_boolean_shap_rejected,
        test_probability_nan_rejected,
        test_probability_inf_rejected,
        test_probability_negative_tolerance_clipped,
        test_probability_negative_outside_tolerance_rejected,
        test_probability_upper_tolerance_clipped,
        test_probability_upper_outside_tolerance_rejected,
        test_xgb_probability_optional,
        test_lstm_probability_optional,
        test_expected_feature_count,
        test_feature_length_change_across_windows,
        test_observations_are_copied,
        test_features_are_copied,
        test_shap_values_are_copied,
        test_to_dict_returns_copies,
        test_duplicate_does_not_replace_state,
        test_new_window_does_not_return_previous_window_as_delta,
        test_clear_window_records_closed_window,
        test_clear_window_lower_than_closed_window,
        test_clear_window_same_as_closed_window,
        test_clear_window_active_same_window,
        test_clear_window_active_newer_window,
        test_clear_window_active_older_window,
        test_closed_window_blocks_equal_window,
        test_closed_window_blocks_older_window,
        test_newer_window_after_closed_window_allowed,
        test_evict_active_state,
        test_evict_closed_state_removes_closed_marker,
        test_clear_all_removes_closed_markers,
        test_feature_length_locked_per_stay,
        test_expected_feature_count_validation,
        test_manager_none_feature_count,
        test_active_stays_after_reset,
        test_active_stays_after_evict,
        test_manager_stay_id_normalization,
        test_stay_id_string,
        test_stay_id_integral,
        test_stay_id_non_integer_float,
        test_feature_nan_preserved,
        test_shap_inf_rejected,
        test_feature_negative_inf_rejected,
        test_observations_are_numeric,
        test_to_dict_charttime_is_iso_safe,
        test_get_returns_copy,
        test_duplicate_returns_copy,
    ]
    for test in tests:
        test()
    print("STAGE C LIVE UPDATE STATE TEST PASSED")
if __name__ == "__main__":
    main()