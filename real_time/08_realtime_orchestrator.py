from collections import Counter, deque
import importlib.util
import logging
import math
import sys
import time
from pathlib import Path
from unittest import result
def load_sibling(module_name, file_name):
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    path = Path(__file__).resolve().parent / file_name
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
patient_state = load_sibling("patient_state", "04_patient_state.py")
PatientStateManager = patient_state.PatientStateManager
ConflictingWindowError = patient_state.ConflictingWindowError
StaleWindowError = patient_state.StaleWindowError
DischargedStayError = patient_state.DischargedStayError
TimestampOrderError = patient_state.TimestampOrderError
StaticChangedError = patient_state.StaticChangedError
realtime_shap = load_sibling("realtime_shap", "10_realtime_shap.py")
RealtimeSHAP = realtime_shap.RealtimeSHAP
class RealtimeOrchestrator:
    def __init__(self, manager, detector, inference, logger, shap_engine=None):
        self.manager = manager
        self.detector = detector
        self.inference = inference
        self.logger = logger
        self.shap_engine = shap_engine
        self.data_errors = deque(maxlen=1000)
        self.result_counters = Counter()
        self.error_counters = Counter()
        self.event_counters = Counter()
        self.evicted_stays = set()
        self.error_logger = logging.getLogger("realtime_orchestrator")
    def _result(self, stay_id, window_id, accepted, reason, **fields):
        result = {
            "accepted": accepted,
            "reason": reason,
            "stay_id": stay_id,
            "window_id": window_id,
            "timestamp": None,
            "sequence_length": None,
            "windows_accepted": None,
            "recovered": False,
            "inference": None,
            "change_detection": None,
            "latency_ms": 0.0
        }
        result.update(fields)
        return result
    def process_window(self, stay_id, window_id, timestamp, temporal, missing_mask, static, xgb_features):
        start_time = time.perf_counter()
        try:
            result = self._process_window(stay_id, window_id, timestamp, temporal, missing_mask, static, xgb_features)
        except Exception as exc:
            self._record_data_error(stay_id, window_id, "unexpected_error", str(exc), True)
            result = self._result(stay_id, window_id, False, "unexpected_error")
        result["latency_ms"] = (time.perf_counter() - start_time) * 1000.0
        self.result_counters[result["reason"]] += 1
        return result
    def _process_window(self, stay_id, window_id, timestamp, temporal, missing_mask, static, xgb_features):
        recovered = stay_id in self.evicted_stays
        try:
            state, accepted = self.manager.add_window(
                stay_id,
                window_id,
                timestamp,
                temporal,
                missing_mask,
                static,
                xgb_features
            )
        except StaleWindowError:
            return self._result(stay_id, window_id, False, "stale")
        except DischargedStayError:
            return self._result(stay_id, window_id, False, "discharged")
        except ConflictingWindowError as exc:
            self._record_data_error(stay_id, window_id, "conflicting_window", str(exc), False)
            return self._result(stay_id, window_id, False, "conflict")
        except StaticChangedError as exc:
            self._record_data_error(stay_id, window_id, "static_changed", str(exc), False)
            return self._result(stay_id, window_id, False, "static_changed")
        except TimestampOrderError as exc:
            self._record_data_error(stay_id, window_id, "timestamp_order", str(exc), False)
            return self._result(stay_id, window_id, False, "timestamp_order")
        except (TypeError, ValueError) as exc:
            self._record_data_error(stay_id, window_id, "invalid_input", str(exc), False)
            return self._result(stay_id, window_id, False, "invalid")
        if not accepted:
            return self._result(stay_id, window_id, False, "duplicate")
        try:
            temporal_seq, missing_seq, padding_seq, static_seq, sequence_length, xgb_vector = state.get_sequence()
            last_window_id, last_timestamp = state.get_identity()
            windows_accepted = state.get_windows_accepted()
            if last_window_id != window_id:
                raise RuntimeError(f"identity mismatch: expected {window_id}, got {last_window_id}")
        except Exception as exc:
            self._record_data_error(stay_id, window_id, "state_error", str(exc), True)
            return self._result(
                stay_id,
                window_id,
                True,
                "state_error",
                timestamp=timestamp,
                recovered=recovered
            )
        try:
            inference_result = self.inference.predict(
                xgb_vector,
                temporal_seq,
                missing_seq,
                padding_seq,
                static_seq
            )
        except Exception as exc:
            self._record_data_error(stay_id, window_id, "inference_error", str(exc), True)
            return self._result(
                stay_id,
                last_window_id,
                True,
                "inference_error",
                timestamp=last_timestamp,
                sequence_length=sequence_length,
                windows_accepted=windows_accepted,
                recovered=recovered
            )
        try:
            ensemble_probability = float(inference_result["ensemble_probability"])
            raw_prediction = inference_result["prediction"]
            if isinstance(raw_prediction, (str, bytes)) or raw_prediction not in (0, 1):
                raise ValueError(f"invalid prediction: {raw_prediction!r}")
            prediction = int(raw_prediction)
            if not math.isfinite(ensemble_probability) or not 0.0 <= ensemble_probability <= 1.0:
                raise ValueError(f"invalid inference output: p={ensemble_probability}, pred={prediction}")
            inference_sequence_length = int(inference_result["sequence_length"])
            if inference_sequence_length != sequence_length:
                raise ValueError(f"sequence length mismatch: state={sequence_length}, inference={inference_sequence_length}")
        except (TypeError, ValueError, KeyError, OverflowError) as exc:
            self._record_data_error(stay_id, window_id, "invalid_inference_output", str(exc), True)
            return self._result(
                stay_id,
                last_window_id,
                True,
                "invalid_inference_output",
                timestamp=last_timestamp,
                sequence_length=sequence_length,
                windows_accepted=windows_accepted,
                recovered=recovered,
                inference=inference_result
            )
        shap_result = None
        if self.shap_engine is not None:
            try:
                shap_result = self.shap_engine.update(
                    stay_id,
                    last_window_id,
                    last_timestamp,
                    xgb_vector,
                    inference_result
                )
            except Exception as exc:
                self._record_data_error(stay_id, window_id, "shap_error", str(exc), True)
                shap_result = {
                    "status": "error",
                    "message": str(exc)
                }
        try:
            detector_result = self.detector.update(
                stay_id,
                last_window_id,
                ensemble_probability,
                prediction
            )
        except Exception as exc:
            self._record_data_error(stay_id, window_id, "detector_error", str(exc), True)
            return self._result(
                stay_id,
                last_window_id,
                True,
                "detector_error",
                timestamp=last_timestamp,
                sequence_length=sequence_length,
                windows_accepted=windows_accepted,
                recovered=recovered,
                inference=inference_result
            )
        self.evicted_stays.discard(stay_id)
        try:
            logger_event = {**detector_result, "recovered": recovered, "shap": shap_result}
            self.logger.record(logger_event)
        except Exception as exc:
            self._record_data_error(stay_id, window_id, "logger_error", str(exc), True)
            return self._result(
                stay_id,
                last_window_id,
                True,
                "logger_error",
                timestamp=last_timestamp,
                sequence_length=sequence_length,
                windows_accepted=windows_accepted,
                recovered=recovered,
                inference=inference_result,
                change_detection=detector_result
            )
        return self._result(
            stay_id,
            last_window_id,
            True,
            "processed",
            timestamp=last_timestamp,
            sequence_length=sequence_length,
            windows_accepted=windows_accepted,
            recovered=recovered,
            inference=inference_result,
            change_detection=detector_result
        )
    def discharge(self, stay_id):
        try:
            if self.manager.get(stay_id) is not None:
                self.manager.discharge(stay_id)
        finally:
            try:
                self.detector.reset(stay_id)
            finally:
                try:
                    if self.shap_engine is not None:
                        self.shap_engine.reset(stay_id)
                finally:
                    self.evicted_stays.discard(stay_id)
    def cleanup(self, watermark_timestamp, evict_active=True):
        expired_stays = self.manager.cleanup(watermark_timestamp, evict_active)
        if expired_stays is None:
            expired_stays = []
        for stay_id in expired_stays:
            try:
                self.detector.reset(stay_id)
            except Exception as exc:
                self._record_data_error(stay_id, None, "detector_reset_error", str(exc), True)
            try:
                if self.shap_engine is not None:
                    self.shap_engine.reset(stay_id)
            except Exception as exc:
                self._record_data_error(stay_id, None, "shap_reset_error", str(exc), True)
            self.evicted_stays.add(stay_id)
            self.event_counters["evicted"] += 1
        return expired_stays
    def _record_data_error(self, stay_id, window_id, error_type, message, with_traceback=False):
        error = {
            "stay_id": stay_id,
            "window_id": window_id,
            "error_type": error_type,
            "message": message
        }
        self.data_errors.append(error)
        self.error_counters[error_type] += 1
        if with_traceback:
            self.error_logger.exception(
                "stay_id=%s window_id=%s error_type=%s message=%s",
                stay_id,
                window_id,
                error_type,
                message
            )
        else:
            self.error_logger.error(
                "stay_id=%s window_id=%s error_type=%s message=%s",
                stay_id,
                window_id,
                error_type,
                message
            )
    def get_data_errors(self):
        return list(self.data_errors)
    def get_counters(self):
        return {
            "result_counters": dict(self.result_counters),
            "error_counters": dict(self.error_counters),
            "event_counters": dict(self.event_counters)
        }