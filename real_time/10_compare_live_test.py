import importlib
import json
import numpy as np
import sys
sys.path.insert(0, "/home/mahith/BDA_PROJECT/Model/real_time")
from importlib.util import spec_from_file_location, module_from_spec
spec = importlib.util.spec_from_file_location("realtime_shap", "/home/mahith/BDA_PROJECT/Model/real_time/10_realtime_shap.py")
module = module_from_spec(spec)
spec.loader.exec_module(module)
RealtimeSHAP = module.RealtimeSHAP
def make_engine():
    engine = object.__new__(RealtimeSHAP)
    engine.expected_features = 137
    engine.top_k = 10
    engine.feature_names = [f"f{i}" for i in range(137)]
    engine.validate_json_safe = RealtimeSHAP.validate_json_safe.__get__(engine)
    engine._validate_features = RealtimeSHAP._validate_features.__get__(engine)
    engine._validate_probability = RealtimeSHAP._validate_probability.__get__(engine)
    engine._normalize_window_id = RealtimeSHAP._normalize_window_id.__get__(engine)
    engine._top_features = RealtimeSHAP._top_features.__get__(engine)
    engine._changed_features = RealtimeSHAP._changed_features.__get__(engine)
    engine._changed_vitals = RealtimeSHAP._changed_vitals
    return engine
def state(window_id=1, charttime="2026-10-08T10:00:00", ensemble=0.2, xgb=0.15, lstm=0.25, feature_value=82.0, shap_value=0.08, observations=None):
    features = np.zeros(137, dtype=np.float32)
    features[0] = feature_value
    shap_values = np.zeros(137, dtype=np.float64)
    shap_values[0] = shap_value
    return {"window_id": window_id, "charttime": charttime, "features": features, "shap_values": shap_values, "ensemble_probability": ensemble, "xgb_probability": xgb, "lstm_probability": lstm, "observations": observations or {"heart_rate": feature_value, "spo2": 97.0}}
def test_first_update():
    engine = make_engine()
    result = engine.compare_live(None, state())
    assert result["previous_window_id"] is None
    assert result["ensemble_probability_delta"] is None
    assert result["changed_features"] == []
    assert result["changed_vitals"] == []
def test_deltas_and_changed_feature():
    engine = make_engine()
    previous = state(ensemble=0.2, xgb=0.15, lstm=0.25, feature_value=82.0, shap_value=0.08, observations={"heart_rate": 82.0, "spo2": 97.0})
    current = state(charttime="2026-10-08T10:01:00", ensemble=0.35, xgb=0.25, lstm=0.45, feature_value=118.0, shap_value=0.21, observations={"heart_rate": 118.0, "spo2": 96.0})
    result = engine.compare_live(previous, current)
    assert abs(result["ensemble_probability_delta"] - 0.15) < 1e-9
    assert abs(result["xgb_probability_delta"] - 0.10) < 1e-9
    assert abs(result["lstm_probability_delta"] - 0.20) < 1e-9
    assert result["changed_features"][0]["feature"] == "f0"
    assert abs(result["changed_features"][0]["value_delta"] - 36.0) < 1e-9
    assert abs(result["changed_features"][0]["shap_delta"] - 0.13) < 1e-9
    assert {x["vital"] for x in result["changed_vitals"]} == {"heart_rate", "spo2"}
    assert result["charttime"] == "2026-10-08T10:01:00"
    assert result["previous_charttime"] == "2026-10-08T10:00:00"
def test_optional_probability_deltas():
    engine = make_engine()
    previous = state(xgb=None, lstm=None)
    current = state(xgb=None, lstm=0.4)
    result = engine.compare_live(previous, current)
    assert result["xgb_probability_delta"] is None
    assert result["lstm_probability_delta"] is None
def test_identical_states_have_no_changed_features():
    engine = make_engine()
    previous = state()
    current = state(charttime="2026-10-08T10:01:00")
    result = engine.compare_live(previous, current)
    assert result["changed_features"] == []
    assert result["changed_vitals"] == []
def test_nan_to_value():
    engine = make_engine()
    previous = state()
    previous["features"][0] = np.nan
    previous["observations"]["heart_rate"] = None
    current = state(feature_value=118.0, shap_value=0.2, charttime="2026-10-08T10:01:00")
    result = engine.compare_live(previous, current)
    row = result["changed_features"][0]
    assert row["value_changed"] is True
    assert row["value_delta"] is None
def test_window_mismatch():
    engine = make_engine()
    previous = state(window_id=1)
    current = state(window_id=2)
    try:
        engine.compare_live(previous, current)
    except ValueError as exc:
        assert str(exc) == "compare_live requires previous and current from the same window"
    else:
        raise AssertionError("Expected ValueError")
def test_invalid_shap():
    engine = make_engine()
    current = state()
    current["shap_values"] = np.zeros(136)
    try:
        engine.compare_live(None, current)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected RuntimeError")
    current = state()
    current["shap_values"][0] = np.nan
    try:
        engine.compare_live(None, current)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected RuntimeError")
def test_missing_keys():
    engine = make_engine()
    current = state()
    current.pop("window_id")
    try:
        engine.compare_live(None, current)
    except KeyError as exc:
        assert str(exc) == "'current is missing window_id'"
    else:
        raise AssertionError("Expected KeyError")
def test_json_safe():
    engine = make_engine()
    result = engine.compare_live(None, state())
    json.dumps(result, allow_nan=False)
if __name__ == "__main__":
    tests = [value for name, value in globals().items() if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
    print(f"COMPARE LIVE TEST PASSED: {len(tests)} tests")
