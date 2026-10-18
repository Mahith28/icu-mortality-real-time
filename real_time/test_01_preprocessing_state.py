"""Round-trip tests for RealtimePreprocessor runtime-state serialization."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import types

ROOT = Path(__file__).resolve().parents[2]
REALTIME_DIR = ROOT / "Model" / "real_time"
if not (REALTIME_DIR / "01_realtime_preprocessing.py").exists():
    REALTIME_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REALTIME_DIR))
from importlib.util import module_from_spec, spec_from_file_location

# State tests do not exercise Spark, so provide import-only stubs when PySpark
# is unavailable. Real Spark-dependent paths are intentionally not tested here.
def _install_import_stubs():
    if "pyspark" not in sys.modules:
        pyspark = types.ModuleType("pyspark")
        sql = types.ModuleType("pyspark.sql")
        sql_types = types.ModuleType("pyspark.sql.types")
        ml = types.ModuleType("pyspark.ml")
        ml_functions = types.ModuleType("pyspark.ml.functions")
        class _Dummy:
            def __init__(self, *args, **kwargs): pass
        sql.SparkSession = _Dummy
        for name in ("StructType", "StructField", "DoubleType", "StringType"):
            setattr(sql_types, name, _Dummy)
        ml.PipelineModel = _Dummy
        ml_functions.vector_to_array = lambda *args, **kwargs: None
        sys.modules.update({"pyspark": pyspark, "pyspark.sql": sql,
                            "pyspark.sql.types": sql_types, "pyspark.ml": ml,
                            "pyspark.ml.functions": ml_functions})
    if "config" not in sys.modules:
        config = types.ModuleType("config")
        config.WINDOW_MINUTES = 5
        config.ASSEMBLER_INPUT_COUNT = 61
        config.TEMPORAL_FEATURES = 54
        config.STATIC_FEATURES = 83
        config.FEATURE_COUNT = 137
        config.MAX_SEQ_LEN = 120
        config.CLEAN_PIPELINE = "unused"
        config.ARTIFACT_DIR = "unused"
        sys.modules["config"] = config

_install_import_stubs()
spec = spec_from_file_location("preprocessing_under_test", REALTIME_DIR / "01_realtime_preprocessing.py")
module = module_from_spec(spec)
spec.loader.exec_module(module)
RealtimePreprocessor = module.RealtimePreprocessor


def test_round_trip_active_windows_and_locf():
    original = RealtimePreprocessor()
    original.add_observation(1001, "2026-01-01 10:01:00", "2026-01-01 10:00:00", "Heart Rate", 82.5)
    original.add_observation(1001, "2026-01-01 10:02:00", "2026-01-01 10:00:00", "SBP", np.nan)
    original.add_observation(1002, "2026-01-01 10:06:00", "2026-01-01 10:00:00", "MAP", 70)
    original.locf_state = {
        1001: {"Heart_Rate_mean": 82.5, "SBP_mean": float("nan"), "Temperature_last": None},
        1002: {"MAP_mean": np.float64(70.0)},
    }

    payload = original.to_state_dict()
    encoded = json.dumps(payload, allow_nan=False)
    restored = RealtimePreprocessor()
    restored.restore_state_dict(json.loads(encoded))

    assert set(restored.active_windows) == {1001, 1002}
    expected_window_1001 = original.get_window_id("2026-01-01 10:01:00", "2026-01-01 10:00:00")
    expected_window_1002 = original.get_window_id("2026-01-01 10:06:00", "2026-01-01 10:00:00")
    assert set(restored.active_windows[1001]) == {expected_window_1001}
    assert set(restored.active_windows[1002]) == {expected_window_1002}
    obs = restored.active_windows[1001][expected_window_1001]
    assert isinstance(obs[0]["charttime"], pd.Timestamp)
    assert obs[0]["vital_sign"] == "Heart Rate" and obs[0]["value"] == 82.5
    assert np.isnan(obs[1]["value"])
    assert np.isnan(restored.locf_state[1001]["SBP_mean"])
    assert restored.locf_state[1001]["Temperature_last"] is None
    assert restored.locf_state[1002]["MAP_mean"] == 70.0


def test_invalid_state_does_not_mutate_existing_state():
    pre = RealtimePreprocessor()
    pre.add_observation(77, "2026-01-01 10:01:00", "2026-01-01 10:00:00", "Heart Rate", 80)
    before = pre.to_state_dict()
    invalid = copy.deepcopy(before)
    invalid["active_windows"].append(copy.deepcopy(invalid["active_windows"][0]))
    try:
        pre.restore_state_dict(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate active-window entry should be rejected")
    assert pre.to_state_dict() == before


def test_invalid_late_section_is_atomic():
    pre = RealtimePreprocessor()
    pre.add_observation(77, "2026-01-01 10:01:00", "2026-01-01 10:00:00", "Heart Rate", 80)
    pre.locf_state = {77: {"Heart_Rate_mean": 80.0}}
    before = pre.to_state_dict()
    invalid = copy.deepcopy(before)
    invalid["locf_state"][0]["values"] = {"NOT_A_FEATURE": 1.0}
    try:
        pre.restore_state_dict(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("Unknown LOCF feature should be rejected")
    assert pre.to_state_dict() == before


def test_semantic_round_trip_and_bad_values():
    original = RealtimePreprocessor()
    original.add_observation(1001, "2026-01-01 10:01:00", "2026-01-01 10:00:00", "Heart Rate", 82.5)
    original.add_observation(1001, "2026-01-01 10:02:00", "2026-01-01 10:00:00", "SBP", 110.0)
    original.locf_state = {1001: {"Heart_Rate_mean": 70.0}}
    observations = pd.DataFrame(original.active_windows[1001][original.get_window_id("2026-01-01 10:01:00", "2026-01-01 10:00:00")])
    expected = original.build_temporal_features(observations, 1001)
    restored = RealtimePreprocessor()
    restored.restore_state_dict(json.loads(json.dumps(original.to_state_dict(), allow_nan=False)))
    restored_observations = pd.DataFrame(restored.active_windows[1001][original.get_window_id("2026-01-01 10:01:00", "2026-01-01 10:00:00")])
    actual = restored.build_temporal_features(restored_observations, 1001)
    assert expected.keys() == actual.keys()
    for key in expected:
        if np.isnan(expected[key]):
            assert np.isnan(actual[key]), key
        else:
            assert expected[key] == actual[key], key

    invalid_cases = []
    negative_window = copy.deepcopy(original.to_state_dict())
    negative_window["active_windows"][0]["window_id"] = -1
    invalid_cases.append(negative_window)
    bad_locf_name = copy.deepcopy(original.to_state_dict())
    bad_locf_name["locf_state"][0]["values"] = {"UNKNOWN": 1.0}
    invalid_cases.append(bad_locf_name)
    bad_locf_value = copy.deepcopy(original.to_state_dict())
    bad_locf_value["locf_state"][0]["values"] = {"Heart_Rate_mean": "abc"}
    invalid_cases.append(bad_locf_value)
    empty_observations = copy.deepcopy(original.to_state_dict())
    empty_observations["active_windows"][0]["observations"] = []
    invalid_cases.append(empty_observations)
    for invalid in invalid_cases:
        try:
            restored.restore_state_dict(invalid)
        except ValueError:
            continue
        raise AssertionError("Invalid state should be rejected")


def test_store_end_to_end():
    store_spec = spec_from_file_location("state_store_under_test", REALTIME_DIR / "14_realtime_state_store.py")
    store_module = module_from_spec(store_spec)
    store_spec.loader.exec_module(store_module)
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "runtime.db"
        original = RealtimePreprocessor()
        original.add_observation(1001, "2026-01-01 10:01:00", "2026-01-01 10:00:00", "Heart Rate", 82.5)
        original.locf_state = {1001: {"Heart_Rate_mean": float("nan")}}
        with store_module.RealtimeStateStore(path) as store:
            store.save_snapshot(original.to_state_dict(), {("icu_observations", 0): 12})
        with store_module.RealtimeStateStore(path) as store:
            loaded = store.load()
        restored = RealtimePreprocessor()
        restored.restore_state_dict(loaded.snapshot)
        assert loaded.offsets == {("icu_observations", 0): 12}
        assert np.isnan(restored.locf_state[1001]["Heart_Rate_mean"])
        assert restored.to_state_dict() == original.to_state_dict()


def test_bad_schema_rejected():
    pre = RealtimePreprocessor()
    try:
        pre.restore_state_dict({"schema_version": 999, "active_windows": [], "locf_state": []})
    except ValueError:
        pass
    else:
        raise AssertionError("Unknown schema version should be rejected")


if __name__ == "__main__":
    test_round_trip_active_windows_and_locf()
    test_invalid_state_does_not_mutate_existing_state()
    test_bad_schema_rejected()
    test_invalid_late_section_is_atomic()
    test_semantic_round_trip_and_bad_values()
    test_store_end_to_end()
    print("PREPROCESSOR STATE TESTS PASSED (6 tests)")
