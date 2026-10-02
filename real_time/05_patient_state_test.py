import importlib.util
import numpy as np
import config as C
spec = importlib.util.spec_from_file_location("patient_state", "04_patient_state.py")
patient_state = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patient_state)
PatientStateManager = patient_state.PatientStateManager
PatientStateError = patient_state.PatientStateError
ConflictingWindowError = patient_state.ConflictingWindowError
StaleWindowError = patient_state.StaleWindowError
DischargedStayError = patient_state.DischargedStayError
TimestampOrderError = patient_state.TimestampOrderError
StaticChangedError = patient_state.StaticChangedError
def make_window(window_id, timestamp, value=0.0):
    temporal = np.full(C.TEMPORAL_FEATURES, value, dtype=np.float32)
    missing_mask = np.zeros(C.TEMPORAL_FEATURES, dtype=np.float32)
    static = np.arange(C.STATIC_FEATURES, dtype=np.float32)
    xgb_features = np.full(C.FEATURE_COUNT, value, dtype=np.float32)
    return window_id, timestamp, temporal, missing_mask, static, xgb_features
def add(manager, stay_id, window_id, timestamp, value=0.0):
    data = make_window(window_id, timestamp, value)
    return manager.add_window(stay_id, *data)
def test_basic_sequence():
    manager = PatientStateManager()
    state, accepted = add(manager, 1, 0, 1000, 0.0)
    assert accepted is True
    assert len(manager) == 1
    state, accepted = add(manager, 1, 1, 1300, 1.0)
    assert accepted is True
    result = state.get_sequence()
    assert result is not None
    temporal, missing_mask, padding_mask, static, sequence_length, xgb_features = result
    assert sequence_length == 2
    assert temporal.shape == (C.MAX_SEQ_LEN, C.TEMPORAL_FEATURES)
    assert missing_mask.shape == (C.MAX_SEQ_LEN, C.TEMPORAL_FEATURES)
    assert padding_mask.shape == (C.MAX_SEQ_LEN,)
    assert static.shape == (C.STATIC_FEATURES,)
    assert xgb_features.shape == (C.FEATURE_COUNT,)
    assert np.array_equal(temporal[0], np.zeros(C.TEMPORAL_FEATURES, dtype=np.float32))
    assert np.array_equal(temporal[1], np.ones(C.TEMPORAL_FEATURES, dtype=np.float32))
    assert np.array_equal(padding_mask[:2], np.zeros(2, dtype=np.float32))
    assert np.array_equal(padding_mask[2:], np.ones(C.MAX_SEQ_LEN - 2, dtype=np.float32))
    print("Basic sequence test passed")
def test_identity_accessor():
    manager = PatientStateManager()
    state, accepted = add(manager, 1, 0, 1000, 0.0)
    assert accepted is True
    assert state.get_identity() == (0, 1000.0)
    add(manager, 1, 1, 1300, 1.0)
    assert state.get_identity() == (1, 1300.0)
    print("Identity accessor test passed")
def test_identity_unchanged_after_rejection():
    manager = PatientStateManager()
    state, accepted = add(manager, 21, 0, 1000, 1.0)
    assert accepted is True
    identity_before = state.get_identity()
    try:
        add(manager, 21, 1, 900, 2.0)
        raise AssertionError("Rejected timestamp window was accepted")
    except TimestampOrderError:
        pass
    assert state.get_identity() == identity_before
    print("Identity rejection test passed")
def test_windows_accepted_counter():
    manager = PatientStateManager()
    state, _ = add(manager, 30, 0, 1000, 1.0)
    assert state.get_windows_accepted() == 1
    add(manager, 30, 0, 1000, 1.0)
    assert state.get_windows_accepted() == 1
    try:
        add(manager, 30, 0, 1000, 2.0)
    except ConflictingWindowError:
        pass
    try:
        add(manager, 30, 1, 900, 2.0)
    except TimestampOrderError:
        pass
    assert state.get_windows_accepted() == 1
    add(manager, 30, 1, 1300, 2.0)
    assert state.get_windows_accepted() == 2
    print("Windows accepted counter test passed")
def test_cleanup_evicts_idle_active():
    manager = PatientStateManager(timeout_hours=1)
    add(manager, 31, 0, 1000, 1.0)
    add(manager, 32, 0, 1000 + 3600 * 2, 1.0)
    expired = manager.cleanup(1000 + 3600 * 2 + 10)
    assert expired == [31]
    assert manager.get(31) is None
    assert manager.get(32) is not None
    print("Cleanup evicts idle active test passed")
def test_failed_first_window():
    manager = PatientStateManager()
    temporal = np.zeros(C.TEMPORAL_FEATURES, dtype=np.float32)
    missing_mask = np.zeros(C.TEMPORAL_FEATURES, dtype=np.float32)
    static = np.zeros(C.STATIC_FEATURES, dtype=np.float32)
    try:
        manager.add_window(2, 0, 1000, temporal[:-1], missing_mask, static)
        raise AssertionError("Invalid first window was accepted")
    except ValueError:
        pass
    assert len(manager) == 0
    assert manager.get(2) is None
    print("Failed first window test passed")
def test_cleanup_evict_active_false():
    manager = PatientStateManager(timeout_hours=1)
    add(manager, 33, 0, 1000, 1.0)
    add(manager, 34, 0, 1000, 1.0)
    manager.discharge(34)
    expired = manager.cleanup(1000 + 3600 * 3, evict_active=False)
    assert expired == [34]
    assert manager.get(33) is not None
    print("Cleanup evict active false test passed")
def test_rejected_first_window_leaves_no_state():
    manager = PatientStateManager()
    temporal = np.zeros(C.TEMPORAL_FEATURES, dtype=np.float32)
    missing_mask = np.zeros(C.TEMPORAL_FEATURES, dtype=np.float32)
    static = np.zeros(C.STATIC_FEATURES, dtype=np.float32)
    try:
        manager.add_window(20, 0, 1000, temporal[:-1], missing_mask, static)
        raise AssertionError("Rejected first window was accepted")
    except ValueError:
        pass
    assert manager.get(20) is None
    assert len(manager) == 0
    print("Rejected first window state test passed")
def test_remove_then_recreate():
    manager = PatientStateManager()
    add(manager, 35, 0, 1000, 1.0)
    add(manager, 35, 1, 1300, 2.0)
    manager.remove(35)
    state, accepted = add(manager, 35, 2, 1600, 3.0)
    assert accepted is True
    assert state.get_windows_accepted() == 1
    assert state.get_sequence()[4] == 1
    print("Remove then recreate test passed")
def test_duplicate_window():
    manager = PatientStateManager()
    add(manager, 3, 0, 1000, 1.0)
    _, accepted = add(manager, 3, 0, 1000, 1.0)
    assert accepted is False
    state = manager.get(3)
    assert len(state.temporal) == 1
    print("Duplicate window test passed")
def test_cleanup_rejects_bad_watermark():
    manager = PatientStateManager()
    for bad in (True, float("nan"), float("inf")):
        try:
            manager.cleanup(bad)
            raise AssertionError("Bad watermark accepted")
        except (TypeError, ValueError):
            pass
    print("Cleanup rejects bad watermark test passed")
def test_duplicate_preserves_identity():
    manager = PatientStateManager()
    state, accepted = add(manager, 22, 0, 1000, 1.0)
    assert accepted is True
    identity_before = state.get_identity()
    _, accepted = add(manager, 22, 0, 1000, 1.0)
    assert accepted is False
    assert state.get_identity() == identity_before
    assert len(state.temporal) == 1
    print("Duplicate identity test passed")
def test_conflicting_duplicate():
    manager = PatientStateManager()
    add(manager, 4, 0, 1000, 1.0)
    try:
        add(manager, 4, 0, 1000, 2.0)
        raise AssertionError("Conflicting duplicate was accepted")
    except ConflictingWindowError as e:
        assert isinstance(e, ValueError)
    assert len(manager.get(4).temporal) == 1
    print("Conflicting duplicate test passed")
def test_conflict_leaves_state_unchanged():
    manager = PatientStateManager()
    add(manager, 23, 0, 1000, 1.0)
    state = manager.get(23)
    before = state.get_sequence()
    try:
        add(manager, 23, 0, 1000, 2.0)
        raise AssertionError("Conflicting duplicate was accepted")
    except ConflictingWindowError:
        pass
    after = state.get_sequence()
    assert state.get_identity() == (0, 1000.0)
    for a, b in zip(before, after):
        if isinstance(a, np.ndarray):
            assert np.array_equal(a, b, equal_nan=True)
        else:
            assert a == b
    print("Conflict state preservation test passed")
def test_typed_conflicting_duplicate():
    manager = PatientStateManager()
    add(manager, 17, 0, 1000, 1.0)
    try:
        add(manager, 17, 0, 1000, 2.0)
        raise AssertionError("Conflicting duplicate did not raise")
    except ConflictingWindowError as e:
        assert isinstance(e, ValueError)
    print("Typed conflicting duplicate test passed")
def test_out_of_order():
    manager = PatientStateManager()
    add(manager, 5, 10, 1000, 1.0)
    try:
        add(manager, 5, 9, 1300, 2.0)
        raise AssertionError("Out-of-order window was accepted")
    except ValueError:
        pass
    assert manager.get(5).last_window_id == 10
    print("Out-of-order test passed")
def test_typed_stale_window():
    manager = PatientStateManager()
    add(manager, 18, 10, 1000, 1.0)
    try:
        add(manager, 18, 9, 1300, 2.0)
        raise AssertionError("Stale window did not raise")
    except StaleWindowError as e:
        assert isinstance(e, ValueError)
    print("Typed stale window test passed")
def test_timestamp_order():
    manager = PatientStateManager()
    add(manager, 6, 0, 1000, 1.0)
    try:
        add(manager, 6, 1, 900, 2.0)
        raise AssertionError("Non-monotonic timestamp was accepted")
    except TimestampOrderError as e:
        assert isinstance(e, ValueError)
    assert manager.get(6).last_timestamp == 1000.0
    print("Timestamp ordering test passed")
def test_rolling_120():
    manager = PatientStateManager()
    for window_id in range(C.MAX_SEQ_LEN + 1):
        add(manager, 7, window_id, 1000 + window_id * 300, float(window_id))
    state = manager.get(7)
    result = state.get_sequence()
    temporal, missing_mask, padding_mask, static, sequence_length, xgb_features = result
    assert sequence_length == C.MAX_SEQ_LEN
    assert np.all(temporal[:, 0] == np.arange(1, C.MAX_SEQ_LEN + 1, dtype=np.float32))
    assert np.all(padding_mask == 0.0)
    assert xgb_features[0] == float(C.MAX_SEQ_LEN)
    print("120-window rolling test passed")
def test_evicted_redelivery():
    manager = PatientStateManager()
    for window_id in range(C.MAX_SEQ_LEN + 1):
        add(manager, 19, window_id, 1000 + window_id * 300, float(window_id))
    state = manager.get(19)
    before = state.get_sequence()
    try:
        add(manager, 19, 0, 1000, 0.0)
        raise AssertionError("Evicted window was accepted")
    except StaleWindowError:
        pass
    after = state.get_sequence()
    assert state.last_window_id == C.MAX_SEQ_LEN
    assert state.last_timestamp == 1000 + C.MAX_SEQ_LEN * 300
    for a, b in zip(before, after):
        if isinstance(a, np.ndarray):
            assert np.array_equal(a, b, equal_nan=True)
        else:
            assert a == b
    print("Evicted redelivery test passed")
def test_mutation_safety():
    manager = PatientStateManager()
    add(manager, 8, 0, 1000, 5.0)
    state = manager.get(8)
    result = state.get_sequence()
    temporal, missing_mask, padding_mask, static, sequence_length, xgb_features = result
    temporal[0, 0] = 999.0
    missing_mask[0, 0] = 1.0
    padding_mask[0] = 1.0
    static[0] = 999.0
    xgb_features[0] = 999.0
    result_again = state.get_sequence()
    temporal_again, missing_mask_again, padding_mask_again, static_again, sequence_length_again, xgb_features_again = result_again
    assert temporal_again[0, 0] == 5.0
    assert missing_mask_again[0, 0] == 0.0
    assert padding_mask_again[0] == 0.0
    assert static_again[0] == 0.0
    assert xgb_features_again[0] == 5.0
    print("Mutation safety test passed")
def test_xgb_nan_preservation():
    manager = PatientStateManager()
    data = make_window(0, 1000, 1.0)
    xgb_features = data[5].copy()
    xgb_features[[0, 16, 80]] = np.nan
    _, accepted = manager.add_window(12, data[0], data[1], data[2], data[3], data[4], xgb_features)
    assert accepted is True
    stored = manager.get(12).get_sequence()[5]
    assert np.array_equal(stored, xgb_features, equal_nan=True)
    print("XGBoost NaN preservation test passed")
def test_xgb_nan_duplicate_and_conflict():
    manager = PatientStateManager()
    data = make_window(0, 1000, 1.0)
    nan_vec = data[5].copy()
    nan_vec[[0, 16, 80]] = np.nan
    _, accepted = manager.add_window(13, data[0], data[1], data[2], data[3], data[4], nan_vec)
    assert accepted is True
    _, accepted = manager.add_window(13, data[0], data[1], data[2], data[3], data[4], nan_vec.copy())
    assert accepted is False
    other = data[5].copy()
    other[[1, 17, 81]] = np.nan
    try:
        manager.add_window(13, data[0], data[1], data[2], data[3], data[4], other)
        raise AssertionError("Different NaN pattern for same window id was accepted")
    except ConflictingWindowError as e:
        assert isinstance(e, ValueError)
    assert len(manager.get(13).temporal) == 1
    print("XGBoost NaN duplicate and conflict test passed")
def test_xgb_nan_input_independence():
    manager = PatientStateManager()
    data = make_window(0, 1000, 1.0)
    vec = data[5].copy()
    vec[0] = np.nan
    _, accepted = manager.add_window(16, data[0], data[1], data[2], data[3], data[4], vec)
    assert accepted is True
    vec[0] = 123.0
    stored = manager.get(16).get_sequence()[5]
    assert np.isnan(stored[0])
    stored[1] = 999.0
    assert manager.get(16).get_sequence()[5][1] == 1.0
    print("XGBoost NaN input independence test passed")
def test_xgb_inf_rejected():
    manager = PatientStateManager()
    add(manager, 14, 0, 1000, 1.0)
    state = manager.get(14)
    before = state.get_sequence()
    data = make_window(1, 1300, 2.0)
    bad = data[5].copy()
    bad[5] = np.inf
    try:
        manager.add_window(14, data[0], data[1], data[2], data[3], data[4], bad)
        raise AssertionError("Infinite XGBoost feature was accepted")
    except ValueError:
        pass
    after = state.get_sequence()
    assert state.last_window_id == 0
    assert state.last_timestamp == 1000.0
    for a, b in zip(before, after):
        if isinstance(a, np.ndarray):
            assert np.array_equal(a, b, equal_nan=True)
        else:
            assert a == b
    print("XGBoost infinity rejection test passed")
def test_nan_rejected_outside_xgb():
    for which in ("temporal", "static"):
        manager = PatientStateManager()
        data = list(make_window(0, 1000, 1.0))
        index = 2 if which == "temporal" else 4
        arr = data[index].copy()
        arr[0] = np.nan
        data[index] = arr
        try:
            manager.add_window(15, *data)
            raise AssertionError(f"NaN in {which} was accepted")
        except ValueError as e:
            assert which in str(e).lower()
        assert manager.get(15) is None
    print("NaN rejection outside XGBoost test passed")
def test_discharge():
    manager = PatientStateManager()
    add(manager, 9, 0, 1000, 1.0)
    manager.discharge(9)
    _, accepted = add(manager, 9, 0, 1000, 1.0)
    assert accepted is False
    try:
        add(manager, 9, 1, 1300, 2.0)
        raise AssertionError("New window after discharge was accepted")
    except DischargedStayError as e:
        assert isinstance(e, ValueError)
    print("Discharge test passed")
def test_static_consistency():
    manager = PatientStateManager()
    add(manager, 10, 0, 1000, 1.0)
    temporal = np.ones(C.TEMPORAL_FEATURES, dtype=np.float32)
    missing_mask = np.zeros(C.TEMPORAL_FEATURES, dtype=np.float32)
    static = np.ones(C.STATIC_FEATURES, dtype=np.float32)
    xgb_features = np.ones(C.FEATURE_COUNT, dtype=np.float32)
    try:
        manager.add_window(10, 1, 1300, temporal, missing_mask, static, xgb_features)
        raise AssertionError("Changed static features were accepted")
    except StaticChangedError as e:
        assert isinstance(e, ValueError)
    assert len(manager.get(10).temporal) == 1
    print("Static consistency test passed")
def test_determinism():
    manager_a = PatientStateManager()
    manager_b = PatientStateManager()
    for window_id in range(10):
        add(manager_a, 11, window_id, 1000 + window_id * 300, float(window_id))
        add(manager_b, 11, window_id, 1000 + window_id * 300, float(window_id))
    result_a = manager_a.get(11).get_sequence()
    result_b = manager_b.get(11).get_sequence()
    for array_a, array_b in zip(result_a[:4], result_b[:4]):
        assert np.array_equal(array_a, array_b)
    assert result_a[4] == result_b[4]
    assert np.array_equal(result_a[5], result_b[5])
    print("Determinism test passed")
def main():
    test_basic_sequence()
    test_identity_accessor()
    test_identity_unchanged_after_rejection()
    test_windows_accepted_counter()
    test_cleanup_evicts_idle_active()
    test_cleanup_evict_active_false()
    test_remove_then_recreate()
    test_cleanup_rejects_bad_watermark()
    test_failed_first_window()
    test_rejected_first_window_leaves_no_state()
    test_duplicate_window()
    test_duplicate_preserves_identity()
    test_conflicting_duplicate()
    test_conflict_leaves_state_unchanged()
    test_typed_conflicting_duplicate()
    test_out_of_order()
    test_typed_stale_window()
    test_timestamp_order()
    test_rolling_120()
    test_evicted_redelivery()
    test_mutation_safety()
    test_xgb_nan_preservation()
    test_xgb_nan_duplicate_and_conflict()
    test_xgb_inf_rejected()
    test_nan_rejected_outside_xgb()
    test_xgb_nan_input_independence()
    test_discharge()
    test_static_consistency()
    test_determinism()
    print("STAGE B PATIENT STATE UNIT TEST PASSED")
if __name__ == "__main__":
    main()