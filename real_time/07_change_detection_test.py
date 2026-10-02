import sys
from pathlib import Path
from importlib import import_module
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
change_detection = import_module("07_change_detection")
ChangeDetector = change_detection.ChangeDetector
AlertLogger = change_detection.AlertLogger
def test_initial_high_risk_alert():
    detector = ChangeDetector(threshold=0.108, cooldown_windows=3)
    r = detector.update(1001, 1, 0.12, 1)
    assert r["alert"] is True
    assert r["alert_reason"] == "initial_high_risk"
    assert r["risk_direction"] == "initial"
def test_threshold_crossing():
    detector = ChangeDetector(threshold=0.108, cooldown_windows=3)
    detector.update(1002, 1, 0.05, 0)
    r = detector.update(1002, 2, 0.12, 1)
    assert r["alert"] is True
    assert r["alert_reason"] == "threshold_crossing"
def test_no_alert_while_high():
    detector = ChangeDetector(threshold=0.108, cooldown_windows=3)
    detector.update(1003, 1, 0.05, 0)
    detector.update(1003, 2, 0.12, 1)
    r = detector.update(1003, 3, 0.15, 1)
    assert r["alert"] is False
def test_pending_alert_fires():
    detector = ChangeDetector(threshold=0.108, cooldown_windows=3)
    detector.update(1004, 1, 0.12, 1)
    detector.update(1004, 2, 0.05, 0)
    r = detector.update(1004, 3, 0.12, 1)
    assert r["alert"] is False and r["pending_alert"] is True
    r = detector.update(1004, 4, 0.13, 1)
    assert r["alert"] is True
    assert r["alert_reason"] == "pending_threshold_crossing"
    assert r["pending_alert"] is False
def test_new_crossing_after_cooldown_expires():
    detector = ChangeDetector(threshold=0.108, cooldown_windows=3)
    detector.update(1022, 1, 0.12, 1)
    detector.update(1022, 2, 0.05, 0)
    detector.update(1022, 3, 0.12, 1)
    detector.update(1022, 4, 0.05, 0)
    r = detector.update(1022, 5, 0.12, 1)
    assert r["alert"] is True
    assert r["alert_reason"] == "threshold_crossing"
def test_cooldown():
    detector = ChangeDetector(threshold=0.108, cooldown_windows=3)
    detector.update(1005, 1, 0.05, 0)
    r = detector.update(1005, 2, 0.12, 1)
    assert r["cooldown_remaining"] == 3
    r = detector.update(1005, 3, 0.13, 1)
    assert r["cooldown_remaining"] == 2
    r = detector.update(1005, 4, 0.14, 1)
    assert r["cooldown_remaining"] == 1
    r = detector.update(1005, 5, 0.15, 1)
    assert r["cooldown_remaining"] == 0
def test_escalation():
    detector = ChangeDetector(threshold=0.108, cooldown_windows=2, escalation_delta=0.10)
    detector.update(1006, 1, 0.12, 1)
    detector.update(1006, 2, 0.15, 1)
    r = detector.update(1006, 3, 0.25, 1)
    assert r["alert"] is True
    assert r["alert_reason"] == "risk_escalation"
def test_direction():
    detector = ChangeDetector(threshold=0.108, min_rise=0.01)
    detector.update(1007, 1, 0.05, 0)
    r = detector.update(1007, 2, 0.08, 0)
    assert r["risk_direction"] == "rising"
    r = detector.update(1007, 3, 0.04, 0)
    assert r["risk_direction"] == "falling"
    r = detector.update(1007, 4, 0.041, 0)
    assert r["risk_direction"] == "stable"
def test_prediction_mismatch():
    detector = ChangeDetector(threshold=0.108)
    r = detector.update(1008, 1, 0.12, 0)
    assert r["prediction"] == 1
    assert r["supplied_prediction"] == 0
    assert r["prediction_mismatch"] is True
def test_duplicate_window():
    detector = ChangeDetector(threshold=0.108)
    detector.update(1009, 1, 0.05, 0)
    r = detector.update(1009, 1, 0.12, 1)
    assert r["duplicate"] is True
    assert r["fresh_evaluation"] is False
    assert r["ensemble_probability"] is None
    assert r["prediction"] is None
    assert r["alert_reason"] == "duplicate_suppressed"
def test_stale_window():
    detector = ChangeDetector(threshold=0.108)
    detector.update(1010, 5, 0.05, 0)
    r = detector.update(1010, 3, 0.12, 1)
    assert r["stale"] is True
    assert r["fresh_evaluation"] is False
    assert r["ensemble_probability"] is None
    assert r["prediction"] is None
    assert r["alert_reason"] == "stale_suppressed"
def test_reset():
    detector = ChangeDetector(threshold=0.108)
    detector.update(1011, 1, 0.05, 0)
    assert detector.get_state(1011) is not None
    detector.reset(1011)
    assert detector.get_state(1011) is None
def test_threshold_boundary():
    detector = ChangeDetector(threshold=0.108)
    detector.update(1012, 1, 0.107, 0)
    r = detector.update(1012, 2, 0.108, 1)
    assert r["alert"] is True
def test_below_threshold():
    detector = ChangeDetector(threshold=0.108)
    detector.update(1013, 1, 0.05, 0)
    r = detector.update(1013, 2, 0.107999, 0)
    assert r["alert"] is False
def test_invalid_probability():
    detector = ChangeDetector(threshold=0.108)
    for value in [np.nan, np.inf, -np.inf, -0.1, 1.1, True]:
        try:
            detector.update(1014, 1, value, 0)
            raise AssertionError("Invalid probability accepted")
        except (ValueError, TypeError):
            pass
def test_invalid_window_id():
    detector = ChangeDetector(threshold=0.108)
    try:
        detector.update(1015, -1, 0.05, 0)
        raise AssertionError("Negative window_id accepted")
    except ValueError:
        pass
def test_boolean_window_id():
    detector = ChangeDetector(threshold=0.108)
    try:
        detector.update(1016, True, 0.05, 0)
        raise AssertionError("Boolean window_id accepted")
    except TypeError:
        pass
def test_invalid_prediction():
    detector = ChangeDetector(threshold=0.108)
    for prediction in [2, -1, True]:
        try:
            detector.update(1017, 1, 0.05, prediction)
            raise AssertionError("Invalid prediction accepted")
        except (ValueError, TypeError):
            pass
def test_none_prediction_rejected_cleanly():
    detector = ChangeDetector(threshold=0.108)
    try:
        detector.update(1030, 1, 0.05, None)
        raise AssertionError("None prediction accepted")
    except TypeError:
        pass
    assert detector.get_state(1030) is None
def test_constructor_validation():
    invalid_values = [
        {"threshold": np.nan},
        {"threshold": np.inf},
        {"escalation_delta": np.nan},
        {"escalation_delta": np.inf},
        {"min_rise": np.nan},
        {"min_rise": np.inf},
        {"escalation_delta": True},
        {"min_rise": True}
    ]
    for kwargs in invalid_values:
        try:
            ChangeDetector(**kwargs)
            raise AssertionError("Invalid constructor value accepted")
        except (ValueError, TypeError):
            pass
    try:
        ChangeDetector(cooldown_windows=2.5)
        raise AssertionError("Non-integer cooldown accepted")
    except TypeError:
        pass
def test_combined_pending_escalation():
    detector = ChangeDetector(threshold=0.108, cooldown_windows=3, escalation_delta=0.10)
    detector.update(1018, 1, 0.12, 1)
    detector.update(1018, 2, 0.05, 0)
    r = detector.update(1018, 3, 0.12, 1)
    assert r["alert"] is False and r["pending_alert"] is True
    r = detector.update(1018, 4, 0.25, 1)
    assert r["alert"] is True
    assert r["alert_reason"] == "risk_escalation"
    assert r["pending_alert"] is False
def test_pending_clears_after_drop():
    detector = ChangeDetector(threshold=0.108, cooldown_windows=3)
    detector.update(1019, 1, 0.12, 1)
    detector.update(1019, 2, 0.05, 0)
    detector.update(1019, 3, 0.12, 1)
    assert detector.get_state(1019)["pending_alert"] is True
    detector.update(1019, 4, 0.05, 0)
    assert detector.get_state(1019)["pending_alert"] is False
def test_pending_across_gap():
    detector = ChangeDetector(threshold=0.108, cooldown_windows=3)
    detector.update(1020, 1, 0.12, 1)
    detector.update(1020, 2, 0.05, 0)
    r = detector.update(1020, 3, 0.12, 1)
    assert r["pending_alert"] is True
    r = detector.update(1020, 7, 0.13, 1)
    assert r["alert"] is True
    assert r["alert_reason"] == "pending_threshold_crossing"
def test_sustained_climb():
    detector = ChangeDetector(threshold=0.108, cooldown_windows=3, escalation_delta=0.10)
    detector.update(1021, 1, 0.12, 1)
    r2 = detector.update(1021, 2, 0.132, 1)
    r3 = detector.update(1021, 3, 0.144, 1)
    r4 = detector.update(1021, 4, 0.156, 1)
    r5 = detector.update(1021, 5, 0.168, 1)
    assert r2["alert"] is False
    assert r3["alert"] is False
    assert r4["alert"] is False
    assert r5["alert"] is False
def test_zero_cooldown():
    detector = ChangeDetector(threshold=0.108, cooldown_windows=0)
    detector.update(1023, 1, 0.05, 0)
    r = detector.update(1023, 2, 0.12, 1)
    assert r["alert"] is True
    detector.update(1023, 3, 0.05, 0)
    r = detector.update(1023, 4, 0.12, 1)
    assert r["alert"] is True
def test_interleaved_stays():
    detector = ChangeDetector(threshold=0.108, cooldown_windows=3)
    detector.update(1024, 1, 0.05, 0)
    detector.update(1025, 1, 0.05, 0)
    r1 = detector.update(1024, 2, 0.12, 1)
    r2 = detector.update(1025, 2, 0.12, 1)
    assert r1["alert"] is True
    assert r2["alert"] is True
    assert detector.get_state(1024)["last_alert_probability"] == 0.12
    assert detector.get_state(1025)["last_alert_probability"] == 0.12
def test_gap_cooldown():
    detector = ChangeDetector(threshold=0.108, cooldown_windows=3)
    detector.update(1026, 1, 0.05, 0)
    detector.update(1026, 2, 0.12, 1)
    r = detector.update(1026, 5, 0.13, 1)
    assert r["cooldown_remaining"] == 0
def test_rejected_call_leaves_state_unchanged():
    detector = ChangeDetector(threshold=0.108)
    detector.update(1027, 1, 0.05, 0)
    before = detector.get_state(1027)
    try:
        detector.update(1027, 2, np.nan, 0)
        raise AssertionError("Invalid probability accepted")
    except ValueError:
        pass
    assert detector.get_state(1027) == before
def test_rejected_prediction_leaves_state_unchanged():
    detector = ChangeDetector(threshold=0.108)
    detector.update(1028, 1, 0.05, 0)
    before = detector.get_state(1028)
    try:
        detector.update(1028, 2, 0.12, None)
        raise AssertionError("None prediction accepted")
    except TypeError:
        pass
    assert detector.get_state(1028) == before
def test_get_state_returns_copy():
    detector = ChangeDetector(threshold=0.108)
    detector.update(1029, 1, 0.05, 0)
    state = detector.get_state(1029)
    state["last_probability"] = 0.99
    assert detector.get_state(1029)["last_probability"] == 0.05
def test_alert_logger():
    logger = AlertLogger(max_events=2)
    logger.record({"alert": False, "prediction_mismatch": False, "alert_reason": None})
    assert len(logger.get_events()) == 0
    logger.record({"alert": True, "prediction_mismatch": False, "alert_reason": "threshold_crossing"})
    logger.record({"alert": False, "prediction_mismatch": True, "alert_reason": None})
    logger.record({"alert": True, "prediction_mismatch": False, "alert_reason": "risk_escalation"})
    assert len(logger.get_events()) == 2
def test_logger_cap():
    logger = AlertLogger(max_events=2)
    for window_id in range(1, 6):
        logger.record({"alert": True, "prediction_mismatch": False, "alert_reason": "threshold_crossing", "window_id": window_id})
    events = logger.get_events()
    assert len(events) == 2
    assert events[0]["window_id"] == 4
    assert events[1]["window_id"] == 5
def test_logger_ordinary_result():
    logger = AlertLogger(max_events=10)
    detector = ChangeDetector(threshold=0.108)
    result = detector.update(1031, 1, 0.05, 0)
    logger.record(result)
    assert len(logger.get_events()) == 0
def test_logger_clear():
    logger = AlertLogger(max_events=10)
    logger.record({"alert": True, "prediction_mismatch": False, "alert_reason": "threshold_crossing"})
    assert len(logger.get_events()) == 1
    logger.clear()
    assert len(logger.get_events()) == 0
def main():
    tests = [
        test_initial_high_risk_alert,
        test_threshold_crossing,
        test_no_alert_while_high,
        test_pending_alert_fires,
        test_new_crossing_after_cooldown_expires,
        test_cooldown,
        test_escalation,
        test_direction,
        test_prediction_mismatch,
        test_duplicate_window,
        test_stale_window,
        test_reset,
        test_threshold_boundary,
        test_below_threshold,
        test_invalid_probability,
        test_invalid_window_id,
        test_boolean_window_id,
        test_invalid_prediction,
        test_none_prediction_rejected_cleanly,
        test_constructor_validation,
        test_combined_pending_escalation,
        test_pending_clears_after_drop,
        test_pending_across_gap,
        test_sustained_climb,
        test_zero_cooldown,
        test_interleaved_stays,
        test_gap_cooldown,
        test_rejected_call_leaves_state_unchanged,
        test_rejected_prediction_leaves_state_unchanged,
        test_get_state_returns_copy,
        test_alert_logger,
        test_logger_cap,
        test_logger_ordinary_result,
        test_logger_clear
    ]
    for test in tests:
        test()
        print(f"{test.__name__} passed")
    print("STAGE C CHANGE DETECTION TEST PASSED")
if __name__ == "__main__":
    main()