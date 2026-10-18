import os
import sys
import math
import json
import importlib.util
import signal
import time
from datetime import datetime, timezone, timedelta
import logging
from collections import defaultdict, deque, OrderedDict
import numpy as np
from kafka import KafkaConsumer, KafkaProducer, TopicPartition
from kafka.errors import KafkaTimeoutError, KafkaConnectionError
from kafka.structs import OffsetAndMetadata
PROJECT_ROOT = os.environ.get("BDA_PROJECT_ROOT", "/home/mahith/BDA_PROJECT")
REAL_TIME_DIR = os.path.join(PROJECT_ROOT, "Model", "real_time")
sys.path.insert(0, REAL_TIME_DIR)
import config as C

# SQL persistence uses the existing VITALIS database and ORM models.
_VITALIS_ROOT = os.path.join(PROJECT_ROOT, "VITALIS")
sys.path.insert(0, _VITALIS_ROOT)
_previous_cwd = os.getcwd()
try:
    os.chdir(_VITALIS_ROOT)
    from backend.app.db.database import SessionLocal
    from backend.app.models.patient_stay import PatientStay
    from backend.app.models.risk_event import RiskEvent
finally:
    os.chdir(_previous_cwd)
BOOTSTRAP_SERVERS = C.KAFKA_BOOTSTRAP_SERVERS
STATIC_TOPIC = C.STATIC_TOPIC
OBSERVATION_TOPIC = C.OBSERVATION_TOPIC
RISK_TOPIC = C.RISK_TOPIC
EXPLANATION_TOPIC = getattr(C, "EXPLANATION_TOPIC", "risk_explanations")
LIFECYCLE_TOPIC = os.environ.get("ICU_LIFECYCLE_TOPIC", "icu_lifecycle")
STATIC_GROUP_ID = "icu_realtime_static_v3"
OBSERVATION_GROUP_ID = "icu_realtime_observations_v2"
DLQ_TOPIC = os.environ.get("ICU_DLQ_TOPIC", "icu_realtime_dlq")
ENABLE_DLQ = os.environ.get("ICU_ENABLE_DLQ", "0") == "1"
PENDING_MAX_PER_STAY = int(os.environ.get("ICU_PENDING_MAX_PER_STAY", "1000"))
SCHEMA_VERSION = "1.0"
FROZEN_XGB_WEIGHT = 0.48
FROZEN_LSTM_WEIGHT = 0.52
FROZEN_ENSEMBLE_THRESHOLD = 0.108
IDLE_FINALIZE_SECONDS = float(os.environ.get("ICU_IDLE_FINALIZE_SECONDS", "900"))
DISABLE_IDLE_FINALIZATION = os.environ.get("ICU_DISABLE_IDLE_FINALIZATION", "0") == "1"
WATERMARK_SECONDS = float(C.WATERMARK_MINUTES) * 60.0
IDLE_SWEEP_SECONDS = float(os.environ.get("ICU_IDLE_SWEEP_SECONDS", "60"))
EVICT_AFTER_SECONDS = float(C.STATE_TIMEOUT_HOURS) * 3600.0
PENDING_TTL_SECONDS = float(os.environ.get("ICU_PENDING_TTL_SECONDS", "3600"))
PENDING_MAX_TOTAL = int(os.environ.get("ICU_PENDING_MAX_TOTAL", "100000"))
ALLOWED_LATENESS_SECONDS = float(os.environ.get("ICU_ALLOWED_LATENESS_SECONDS", "60"))
MAX_EVENT_IDS_PER_STAY = int(os.environ.get("ICU_MAX_EVENT_IDS_PER_STAY", "5000"))
MAX_DISCHARGED_TOMBSTONES = int(os.environ.get("ICU_MAX_DISCHARGED_TOMBSTONES", "50000"))
if MAX_DISCHARGED_TOMBSTONES <= 0:
    raise ValueError("ICU_MAX_DISCHARGED_TOMBSTONES must be positive")
MAX_FINAL_ATTEMPTS = int(os.environ.get("ICU_MAX_FINAL_ATTEMPTS", "5"))
FINAL_RETRY_BASE_SECONDS = float(os.environ.get("ICU_FINAL_RETRY_BASE_SECONDS", "5"))
IDLE_SWEEP_BUDGET_SECONDS = float(os.environ.get("ICU_IDLE_SWEEP_BUDGET_SECONDS", "20"))
DISCHARGE_GRACE_SECONDS = float(os.environ.get("ICU_DISCHARGE_GRACE_SECONDS", "30"))
DISCHARGE_HARD_DEADLINE_SECONDS = float(os.environ.get("ICU_DISCHARGE_HARD_DEADLINE_SECONDS", "300"))
OBSERVATION_MAX_POLL_RECORDS = int(os.environ.get("ICU_OBSERVATION_MAX_POLL_RECORDS", "10"))
OBSERVATION_MAX_POLL_INTERVAL_MS = int(os.environ.get("ICU_OBSERVATION_MAX_POLL_INTERVAL_MS", "300000"))
LIVE_UPDATE_VITALS = frozenset(C.REALTIME_VITAL_NAME_MAP[vital] for vital in C.EXPECTED_PRODUCER_VITALS)
LIVE_UPDATE_REFRESH_SECONDS = float(os.environ.get("ICU_LIVE_UPDATE_REFRESH_SECONDS", "300"))
DISABLE_LIVE_REFRESH = os.environ.get("ICU_DISABLE_LIVE_REFRESH", "0") == "1"
if LIVE_UPDATE_REFRESH_SECONDS < 0:
    raise ValueError("ICU_LIVE_UPDATE_REFRESH_SECONDS must be non-negative")
_last_live_update_monotonic = {}
_last_live_update_window = {}
LIVE_REFRESH_RETRY_SECONDS = float(os.environ.get("ICU_LIVE_REFRESH_RETRY_SECONDS", "30"))
LIVE_EXPLANATION_MAX_PER_STAY = int(os.environ.get("ICU_LIVE_EXPLANATION_MAX_PER_STAY", "100"))
LIVE_EXPLANATION_QUEUE_DROPS = 0
LIVE_EXPLANATION_MAX_TOTAL = int(os.environ.get("ICU_LIVE_EXPLANATION_MAX_TOTAL", "10000"))
LIVE_EXPLANATION_TTL_SECONDS = float(os.environ.get("ICU_LIVE_EXPLANATION_TTL_SECONDS", "3600"))
LIVE_REFRESH_SWEEP_BUDGET_SECONDS = float(os.environ.get("ICU_LIVE_REFRESH_SWEEP_BUDGET_SECONDS", "10"))
KAFKA_BREAKER_COOLDOWN_SECONDS = float(os.environ.get("ICU_KAFKA_BREAKER_COOLDOWN_SECONDS", "15"))
_kafka_breaker_until = 0.0
if LIVE_REFRESH_RETRY_SECONDS <= 0 or LIVE_EXPLANATION_MAX_PER_STAY <= 0:
    raise ValueError("Live refresh retry and explanation queue cap must be positive")
if LIVE_EXPLANATION_MAX_TOTAL <= 0 or LIVE_EXPLANATION_TTL_SECONDS <= 0:
    raise ValueError("Live explanation global cap and TTL must be positive")
if LIVE_REFRESH_SWEEP_BUDGET_SECONDS <= 0 or KAFKA_BREAKER_COOLDOWN_SECONDS <= 0:
    raise ValueError("Refresh sweep budget and Kafka breaker cooldown must be positive")

BROKER_DOWN = (KafkaTimeoutError, KafkaConnectionError)

def is_broker_down(exc):
    return isinstance(exc, BROKER_DOWN)

def kafka_breaker_active():
    return time.monotonic() < _kafka_breaker_until

def trip_kafka_breaker(reason):
    global _kafka_breaker_until
    _kafka_breaker_until = max(_kafka_breaker_until, time.monotonic() + KAFKA_BREAKER_COOLDOWN_SECONDS)
    logger.error("Kafka circuit breaker tripped for %.1fs: %s", KAFKA_BREAKER_COOLDOWN_SECONDS, reason)

# In-memory retry queue for UPDATE explanations only. FINAL has its own durable-in-process retry path.
_pending_live_explanations = {}
_next_live_explanation_id = 0
LIVE_EXPLANATION_RETRY_BASE_SECONDS = float(os.environ.get("ICU_LIVE_EXPLANATION_RETRY_BASE_SECONDS", "5"))
LIVE_EXPLANATION_RETRY_MAX_SECONDS = float(os.environ.get("ICU_LIVE_EXPLANATION_RETRY_MAX_SECONDS", "300"))
if LIVE_EXPLANATION_RETRY_BASE_SECONDS <= 0 or LIVE_EXPLANATION_RETRY_MAX_SECONDS <= 0:
    raise ValueError("Live explanation retry intervals must be positive")
def load_module(name, filename):
    path = os.path.join(REAL_TIME_DIR, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
logger = logging.getLogger("icu_realtime")
_shutdown = False
def request_shutdown(signum, frame):
    global _shutdown
    _shutdown = True
def parse_timestamp(value):
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    timestamp = datetime.fromisoformat(text)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.astimezone(timezone.utc).replace(tzinfo=None)
    return timestamp
def create_producer():
    return KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        acks="all",
        retries=5,
        linger_ms=5,
        value_serializer=lambda value: json.dumps(value, allow_nan=False).encode("utf-8"),
        key_serializer=lambda key: None if key is None else str(key).encode("utf-8"),
        max_in_flight_requests_per_connection=1
    )
def create_consumer(topic, group_id, auto_offset_reset, auto_commit=False, max_poll_records=20):
    return KafkaConsumer(
        topic,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        auto_offset_reset=auto_offset_reset,
        enable_auto_commit=auto_commit,
        group_id=group_id,
        max_poll_records=(OBSERVATION_MAX_POLL_RECORDS if topic == OBSERVATION_TOPIC else max_poll_records),
        max_poll_interval_ms=OBSERVATION_MAX_POLL_INTERVAL_MS,
        value_deserializer=None,
    )
def create_static_consumer():
    consumer = KafkaConsumer(bootstrap_servers=BOOTSTRAP_SERVERS, auto_offset_reset="earliest", enable_auto_commit=False, group_id=None, max_poll_records=500, value_deserializer=None)
    partitions = None
    for _ in range(50):
        partitions = consumer.partitions_for_topic(STATIC_TOPIC)
        if partitions:
            break
        time.sleep(0.2)
    if not partitions:
        consumer.close()
        raise RuntimeError(f"No partitions found for {STATIC_TOPIC}")
    tps = [TopicPartition(STATIC_TOPIC, partition) for partition in sorted(partitions)]
    consumer.assign(tps)
    consumer.seek_to_beginning(*tps)
    return consumer
def validate_observation(data):
    required = ["schema_version", "event_id", "stay_id", "charttime", "ingestion_time", "source_mode", "source_type", "vital_sign", "value"]
    missing = [field for field in required if field not in data]
    if missing:
        raise RuntimeError(f"Observation message missing fields: {missing}")
    if str(data["schema_version"]) != SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported observation schema_version={data['schema_version']}")
    if not str(data["event_id"]).strip():
        raise RuntimeError("Observation event_id is empty")
    parse_timestamp(data["charttime"])
    parse_timestamp(data["ingestion_time"])
def validate_static(data):
    required = ["stay_id", "intime"]
    missing = [field for field in required if field not in data]
    if missing:
        raise RuntimeError(f"Static message missing fields: {missing}")
    int(data["stay_id"])
    parse_timestamp(data["intime"])
def observation_dedupe_key(data):
    event_id = str(data.get("event_id", "")).strip()
    if event_id:
        return ("event_id", event_id)
    return ("fallback", int(data["stay_id"]), str(data["vital_sign"]), parse_timestamp(data["charttime"]).isoformat())
def normalize_observation_vital(data):
    raw = str(data["vital_sign"]).strip().lower()
    canonical = C.REALTIME_VITAL_NAME_MAP.get(raw)
    if canonical is None:
        raise RuntimeError(f"Unknown vital_sign={data['vital_sign']}")
    data["vital_sign"] = canonical
    return data
def observation_fallback_key(data):
    return (int(data["stay_id"]), str(data["vital_sign"]), parse_timestamp(data["charttime"]).isoformat())
def validate_observation_values(data):
    vital_sign = str(data["vital_sign"])
    if vital_sign not in C.REALTIME_VITAL_SIGNS:
        raise RuntimeError(f"Unknown vital_sign={vital_sign}")
    value = float(data["value"])
    if not np.isfinite(value):
        raise RuntimeError("Observation value is not finite")
    low, high = C.REALTIME_VITAL_VALIDATION_RANGES[vital_sign]
    if not low <= value <= high:
        raise RuntimeError(f"Out-of-range {vital_sign} value={value}")
def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [safe_json(v) for v in value]
    if isinstance(value, tuple):
        return [safe_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe_json(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value
def build_sequence(preprocessor, stay_id, window_cache, current_window_id):
    windows = []
    for window_id in sorted(window_cache.get(stay_id, {})):
        if int(window_id) <= int(current_window_id):
            item = window_cache[stay_id][window_id]
            windows.append({"window_id": int(window_id), "temporal": item["temporal"], "missing_mask": item["missing_mask"], "static": item["static"]})
    if not windows:
        raise RuntimeError(f"No sequence windows for stay {stay_id}")
    return preprocessor.build_sequence(windows)
def top_features_of(shap_result):
    if not shap_result:
        return []
    return (shap_result.get("shap") or {}).get("top_features", [])
def build_live_state(LiveUpdateState, stay_id, window_id, charttime, result, inference_result, shap_result, observations):
    if not isinstance(shap_result, dict) or shap_result.get("status") != "ok":
        raise RuntimeError(f"Live SHAP unavailable for stay_id={stay_id} window_id={window_id}")
    explanation = shap_result.get("shap") or {}
    shap_values = explanation.get("shap_values")
    if shap_values is None:
        raise RuntimeError(f"Live SHAP values unavailable for stay_id={stay_id} window_id={window_id}")
    return LiveUpdateState(
        stay_id=int(stay_id),
        window_id=int(window_id),
        charttime=parse_timestamp(charttime),
        features=np.asarray(result["features"], dtype=np.float32),
        shap_values=np.asarray(shap_values, dtype=np.float64),
        ensemble_probability=float(inference_result["ensemble_probability"]),
        xgb_probability=float(inference_result.get("xgb_probability")) if inference_result.get("xgb_probability") is not None else None,
        lstm_probability=float(inference_result.get("lstm_probability")) if inference_result.get("lstm_probability") is not None else None,
        observations=dict(observations)
    )
def state_to_dict(state):
    if state is None:
        return None
    if hasattr(state, "to_dict"):
        return state.to_dict()
    if isinstance(state, dict):
        return state
    raise TypeError(f"Unsupported live state type: {type(state).__name__}")
def queue_live_explanation(stay_id, window_id, event_id, message, error, error_counts=None):
    """Retain a failed live explanation for ordered, independent Kafka retry."""
    global _next_live_explanation_id, LIVE_EXPLANATION_QUEUE_DROPS
    stay_id = int(stay_id)
    now = time.monotonic()
    # Expire old entries before enforcing queue limits.
    for qid, queued in list(_pending_live_explanations.items()):
        if now - queued["queued_at"] >= LIVE_EXPLANATION_TTL_SECONDS:
            _pending_live_explanations.pop(qid, None)
            LIVE_EXPLANATION_QUEUE_DROPS += 1
            logger.warning("Expired pending live explanation stay_id=%s window_id=%s dropped_total=%s", queued["stay_id"], queued["window_id"], LIVE_EXPLANATION_QUEUE_DROPS)
            if error_counts is not None:
                error_counts["live_explanation_queue_drops"] += 1
    per_stay = [(qid, item) for qid, item in _pending_live_explanations.items()
                if item["stay_id"] == stay_id]
    while len(per_stay) >= LIVE_EXPLANATION_MAX_PER_STAY:
        oldest_id, oldest = per_stay.pop(0)
        _pending_live_explanations.pop(oldest_id, None)
        LIVE_EXPLANATION_QUEUE_DROPS += 1
        if error_counts is not None:
            error_counts["live_explanation_queue_drops"] += 1
        logger.warning("Dropped oldest pending explanation at per-stay cap stay_id=%s window_id=%s dropped_total=%s",
                       stay_id, oldest["window_id"], LIVE_EXPLANATION_QUEUE_DROPS)
    while len(_pending_live_explanations) >= LIVE_EXPLANATION_MAX_TOTAL:
        oldest_id = next(iter(_pending_live_explanations))
        oldest = _pending_live_explanations.pop(oldest_id)
        LIVE_EXPLANATION_QUEUE_DROPS += 1
        if error_counts is not None:
            error_counts["live_explanation_queue_drops"] += 1
        logger.warning("Dropped oldest pending explanation at global cap stay_id=%s window_id=%s dropped_total=%s",
                       oldest["stay_id"], oldest["window_id"], LIVE_EXPLANATION_QUEUE_DROPS)
    _next_live_explanation_id += 1
    now = time.monotonic()
    _pending_live_explanations[_next_live_explanation_id] = {
        "stay_id": stay_id,
        "window_id": int(window_id),
        "event_id": str(event_id or ""),
        "message": safe_json(message),
        "attempts": 0,
        "queued_at": now,
        "retry_at": now + LIVE_EXPLANATION_RETRY_BASE_SECONDS,
    }
    logger.info(
        "Queued live explanation for retry stay_id=%s window_id=%s event_id=%s error=%s",
        stay_id, window_id, event_id, str(error),
    )


def discard_pending_live_explanations(stay_id, final_window_id):
    """Prevent pending UPDATE explanations from overwriting a completed FINAL."""
    stay_id = int(stay_id)
    final_window_id = int(final_window_id)
    removed = 0
    for queue_id, item in list(_pending_live_explanations.items()):
        if item["stay_id"] == stay_id and int(item["window_id"]) <= final_window_id:
            _pending_live_explanations.pop(queue_id, None)
            removed += 1
    if removed:
        logger.info(
            "Discarded pending live explanations covered by FINAL stay_id=%s "
            "final_window_id=%s count=%s",
            stay_id, final_window_id, removed,
        )
    return removed


def retry_pending_live_explanations(producer, error_counts=None):
    """Retry the oldest pending explanation per stay, preserving per-stay order."""
    if kafka_breaker_active():
        return
    now = time.monotonic()
    blocked_stays = set()
    for queue_id, item in list(_pending_live_explanations.items()):
        stay_id = item["stay_id"]
        if stay_id in blocked_stays:
            continue
        blocked_stays.add(stay_id)
        if item["retry_at"] > now:
            continue
        try:
            producer.send(EXPLANATION_TOPIC, key=stay_id, value=item["message"]).get(timeout=10)
        except Exception as exc:
            item["attempts"] += 1
            delay = min(
                LIVE_EXPLANATION_RETRY_MAX_SECONDS,
                LIVE_EXPLANATION_RETRY_BASE_SECONDS * (2 ** min(item["attempts"], 16)),
            )
            item["retry_at"] = time.monotonic() + delay
            logger.exception(
                "Live explanation retry failed stay_id=%s window_id=%s event_id=%s attempt=%s",
                stay_id, item["window_id"], item["event_id"], item["attempts"],
            )
            send_dlq(producer, {
                "type": "live_explanation_retry_failed",
                "stay_id": stay_id,
                "window_id": item["window_id"],
                "event_id": item["event_id"],
                "attempt": item["attempts"],
                "error": str(exc),
            })
            if is_broker_down(exc):
                trip_kafka_breaker(f"live explanation retry failed: {exc}")
            if error_counts is not None:
                error_counts["live_explanation_retry_failures"] += 1
            # A broker outage is likely shared; avoid N blocking 10-second waits.
            if is_broker_down(exc):
                break
            continue
        else:
            _pending_live_explanations.pop(queue_id, None)
            logger.info(
                "Live explanation retry succeeded stay_id=%s window_id=%s event_id=%s",
                stay_id, item["window_id"], item["event_id"],
            )


def publish_live_update(producer, stay_id, window_id, charttime, result, sequence, inference, shap_engine, live_state_module, live_state_manager, live_observations, finalized_risk, source_mode, source_type, ingestion_time, event_id, window_start=None, window_end=None, refresh=False, error_counts=None):
    if kafka_breaker_active():
        if error_counts is not None:
            error_counts["live_updates_skipped_breaker"] += 1
        return False
    inference_result, shap_result = run_scoring(stay_id, window_id, charttime, result, sequence, inference, shap_engine, "UPDATE")
    if not isinstance(shap_result, dict) or shap_result.get("status") != "ok":
        # Do not publish a partial UPDATE with a different schema or bypass
        # the live-state ordering checks when its explanation is unavailable.
        if error_counts is not None:
            error_counts["live_shap_failures"] += 1
        logger.error("Suppressing LIVE UPDATE because SHAP failed stay_id=%s window_id=%s", stay_id, window_id)
        return False
    state = build_live_state(live_state_module.LiveUpdateState, stay_id, window_id, charttime, result, inference_result, shap_result, live_observations)
    try:
        previous, state_status = live_state_manager.check(state, allow_same_charttime_refresh=refresh)
        if state_status == "duplicate":
            return False
        if state_status != "updated":
            raise RuntimeError(f"Unexpected live update state status={state_status}")
        current = state
    except (live_state_module.OutOfOrderUpdateError, live_state_module.StaleWindowError, live_state_module.ClosedWindowError) as exc:
        logger.warning("Live UPDATE state rejected stay_id=%s window_id=%s charttime=%s reason=%s", stay_id, window_id, charttime, str(exc))
        return False
    current_dict = state_to_dict(current)
    previous_dict = state_to_dict(previous)
    comparison_previous = previous_dict if previous_dict is not None and int(previous_dict["window_id"]) == int(window_id) else None
    previous_final = finalized_risk.get(stay_id)
    compare = shap_engine.compare_live(comparison_previous, current_dict)
    probability_tolerance = float(getattr(live_state_module.LiveUpdateState, "PROBABILITY_TOLERANCE", 1e-6))
    risk_changed = previous_dict is None or int(previous_dict["window_id"]) != int(window_id)
    if not risk_changed:
        for probability_name in ("ensemble_probability", "xgb_probability", "lstm_probability"):
            old_probability = previous_dict.get(probability_name)
            new_probability = current_dict.get(probability_name)
            if old_probability is None or new_probability is None:
                if old_probability != new_probability:
                    risk_changed = True
                    break
            elif abs(float(new_probability) - float(old_probability)) > probability_tolerance:
                risk_changed = True
                break
    meaningful_change = previous_dict is None or int(previous_dict["window_id"]) != int(window_id) or bool(compare.get("changed_vitals")) or bool(compare.get("changed_features")) or risk_changed
    if not refresh and not meaningful_change:
        return False
    current_shap = np.asarray(current_dict["shap_values"], dtype=np.float64)
    previous_shap = None if comparison_previous is None else np.asarray(comparison_previous["shap_values"], dtype=np.float64)
    shap_delta = None if previous_shap is None else (current_shap - previous_shap).tolist()
    risk_previous = previous_dict if previous_dict is not None else previous_final
    prediction_time = datetime.now(timezone.utc).replace(tzinfo=None)
    update_data = safe_json({
        "update_type": "LIVE",
        "update_reason": "periodic_refresh" if refresh else "initial_update" if previous_dict is None else "meaningful_change",
        "refresh_interval_seconds": LIVE_UPDATE_REFRESH_SECONDS,
        "exclude_from_latency_metrics": bool(refresh),
        "previous_risk": None if risk_previous is None else risk_previous["ensemble_probability"],
        "current_risk": current_dict["ensemble_probability"],
        "risk_delta": None if risk_previous is None else current_dict["ensemble_probability"] - risk_previous["ensemble_probability"],
        "previous_xgb_probability": None if risk_previous is None else risk_previous.get("xgb_probability"),
        "current_xgb_probability": current_dict.get("xgb_probability"),
        "xgb_probability_delta": None if risk_previous is None or risk_previous.get("xgb_probability") is None or current_dict.get("xgb_probability") is None else current_dict["xgb_probability"] - risk_previous["xgb_probability"],
        "previous_lstm_probability": None if risk_previous is None else risk_previous.get("lstm_probability"),
        "current_lstm_probability": current_dict.get("lstm_probability"),
        "lstm_probability_delta": None if risk_previous is None or risk_previous.get("lstm_probability") is None or current_dict.get("lstm_probability") is None else current_dict["lstm_probability"] - risk_previous["lstm_probability"],
        "changed_vitals": compare.get("changed_vitals", []),
        "changed_features": compare.get("changed_features", []),
        "current_shap": current_dict["shap_values"],
        "shap_delta": shap_delta,
        "active_window": {
            "window_id": int(window_id),
            "charttime": parse_timestamp(charttime).isoformat() + "Z",
            "window_start": None if window_start is None else parse_timestamp(window_start).isoformat() + "Z",
            "window_end": None if window_end is None else parse_timestamp(window_end).isoformat() + "Z"
        },
        "previous_charttime": None if risk_previous is None else parse_timestamp(risk_previous["charttime"]).isoformat() + "Z",
        "previous_window_id": None if risk_previous is None else int(risk_previous["window_id"])
    })
    risk_message = make_risk_message(stay_id, window_id, charttime, "UPDATE", result, inference_result, shap_result, window_start, window_end, source_mode, source_type, ingestion_time, prediction_time, event_id)
    risk_message.update(update_data)
    persist_risk_event_sql(risk_message)
    try:
        producer.send(RISK_TOPIC, key=stay_id, value=safe_json(risk_message)).get(timeout=3)
    except Exception as exc:
        if is_broker_down(exc):
            trip_kafka_breaker(f"live risk send failed: {exc}")
        raise
    live_state_manager.update(state, allow_same_charttime_refresh=refresh)
    _last_live_update_monotonic[int(stay_id)] = time.monotonic()
    _last_live_update_window[int(stay_id)] = int(window_id)
    # Keep live-change details while including the canonical SHAP payload
    # required by VITALIS to persist a complete explanation.
    explanation_message = safe_json(shap_result)
    explanation_message.update(safe_json(compare))
    explanation_message.update(update_data)
    explanation_message["schema_version"] = SCHEMA_VERSION
    explanation_message["event_time"] = parse_timestamp(charttime).isoformat() + "Z"
    explanation_message["stay_id"] = int(stay_id)
    explanation_message["window_id"] = int(window_id)
    explanation_message["window_status"] = "UPDATE"
    explanation_message["event_id"] = str(event_id or "")
    explanation_message["window_start"] = None if window_start is None else parse_timestamp(window_start).isoformat() + "Z"
    explanation_message["window_end"] = None if window_end is None else parse_timestamp(window_end).isoformat() + "Z"
    explanation_message["source_mode"] = source_mode
    explanation_message["source_type"] = source_type
    explanation_message["ingestion_time"] = None if ingestion_time is None else parse_timestamp(ingestion_time).isoformat() + "Z"
    explanation_message["prediction_time"] = prediction_time.isoformat() + "Z"
    # If an earlier explanation for this stay is pending, queue this one too so
    # explanations are delivered in the same order as their risk UPDATEs.
    has_earlier_pending = any(
        item["stay_id"] == int(stay_id)
        for item in _pending_live_explanations.values()
    )
    if has_earlier_pending:
        queue_live_explanation(stay_id, window_id, event_id, explanation_message, "earlier explanation pending", error_counts)
    else:
        try:
            producer.send(EXPLANATION_TOPIC, key=stay_id, value=explanation_message).get(timeout=10)
        except Exception as exc:
            logger.exception("explanation publish failed stay_id=%s window_id=%s status=UPDATE", stay_id, window_id)
            queue_live_explanation(stay_id, window_id, event_id, explanation_message, exc, error_counts)
            send_dlq(producer, {"type": "explanation_publish_failed", "stay_id": int(stay_id), "window_id": int(window_id), "event_id": str(event_id or ""), "status": "UPDATE", "error": str(exc)})
            if is_broker_down(exc):
                trip_kafka_breaker(f"live explanation send failed: {exc}")
            if error_counts is not None:
                error_counts["explanation_publish_failures"] += 1
    print(f"UPDATE: stay_id={stay_id} window_id={window_id} ensemble={float(inference_result['ensemble_probability']):.6f} prediction={int(inference_result['prediction'])} delta={update_data['risk_delta']}")
    return True
def run_scoring(stay_id, window_id, timestamp, result, sequence, inference, shap_engine, window_status):
    features = np.asarray(result["features"], dtype=np.float32)
    inference_result = inference.predict(features, np.asarray(sequence["temporal"], dtype=np.float32), np.asarray(sequence["missing_mask"], dtype=np.float32), np.asarray(sequence["padding_mask"], dtype=np.float32), np.asarray(sequence["static"], dtype=np.float32))
    if "threshold" not in inference_result or abs(float(inference_result["threshold"]) - FROZEN_ENSEMBLE_THRESHOLD) >= 1e-9:
        raise RuntimeError(f"Frozen inference threshold drift or missing threshold: {inference_result.get('threshold')}")
    timestamp_value = parse_timestamp(timestamp).isoformat() + "Z"
    try:
        xgb_raw_margin = inference_result.get("xgb_raw_margin")
        if window_status == "FINAL":
            shap_result = shap_engine.update(stay_id, window_id, timestamp, features, inference_result, window_status="FINAL")
            shap_result["stay_id"] = int(stay_id)
        else:
            explanation = shap_engine.explain_live(features, xgb_raw_probability=float(inference_result["xgb_raw_probability"]), xgb_raw_margin=xgb_raw_margin, verify=True)
            shap_result = {"status": "ok", "stay_id": int(stay_id), "window_id": int(window_id), "timestamp": timestamp_value, "window_status": window_status, "previous_window_id": None, "window_gap": None, "previous_discarded": False, "xgb_raw_probability": float(inference_result["xgb_raw_probability"]), "xgb_raw_margin": None if xgb_raw_margin is None else float(xgb_raw_margin), "xgb_probability": float(inference_result["xgb_probability"]), "lstm_probability": float(inference_result["lstm_probability"]), "ensemble_probability": float(inference_result["ensemble_probability"]), "risk": {"ensemble_probability": float(inference_result["ensemble_probability"]), "ensemble_probability_previous": None, "ensemble_probability_delta": None}, "xgb_component": {"raw_probability": float(inference_result["xgb_raw_probability"]), "raw_margin": None if xgb_raw_margin is None else float(xgb_raw_margin), "probability": float(inference_result["xgb_probability"]), "probability_previous": None, "probability_delta": None, "weight": FROZEN_XGB_WEIGHT}, "lstm_component": {"probability": float(inference_result["lstm_probability"]), "probability_previous": None, "probability_delta": None, "weight": FROZEN_LSTM_WEIGHT}, "shap": explanation, "changed_features": [], "latency_ms": 0.0}
    except Exception as exc:
        logger.exception("SHAP failed stay_id=%s window_id=%s", stay_id, window_id)
        shap_result = {"status": "error", "stay_id": int(stay_id), "window_id": int(window_id), "timestamp": timestamp_value, "window_status": window_status, "error_type": type(exc).__name__, "error": str(exc), "shap": {"top_features": []}}
    return inference_result, safe_json(shap_result)


def _sql_datetime(value):
    """Convert ISO timestamps to naive UTC datetimes for SQL DateTime columns."""
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def persist_risk_event_sql(message):
    """Idempotently upsert the current risk event for a stay/window."""
    status = str(message.get("status", "")).upper()
    if status not in {"UPDATE", "FINAL"}:
        return False

    stay_id = int(message["stay_id"])
    window_id = int(message["window_id"])

    db = SessionLocal()
    try:
        stay = (
            db.query(PatientStay)
            .filter(PatientStay.stay_id == stay_id)
            .one_or_none()
        )
        if stay is None:
            raise LookupError(
                f"No PatientStay found for stay_id={stay_id}; "
                "risk event was not persisted"
            )

        event = (
            db.query(RiskEvent)
            .filter(
                RiskEvent.stay_id == stay_id,
                RiskEvent.window_id == window_id,
            )
            .order_by(RiskEvent.id.desc())
            .first()
        )

        # Never let a delayed live refresh downgrade a finalized window.
        if event is not None and event.status == "FINAL" and status == "UPDATE":
            db.rollback()
            return False

        values = {
            "patient_id": stay.patient_id,
            "stay_id": stay_id,
            "window_id": window_id,
            "status": status,
            "event_time": _sql_datetime(message.get("event_time")),
            "window_start": _sql_datetime(message.get("window_start")),
            "window_end": _sql_datetime(message.get("window_end")),
            "sequence_length": message.get("sequence_length"),
            "prediction": message.get("prediction"),
            "threshold": message.get("threshold"),
            "xgb_raw_probability": message.get("xgb_raw_probability"),
            "xgb_probability": message.get("xgb_probability"),
            "lstm_raw_probability": message.get("lstm_raw_probability"),
            "lstm_probability": message.get("lstm_probability"),
            "ensemble_probability": message.get("ensemble_probability"),
            "model_version": message.get("model_version"),
            "latency_ms": message.get("latency_ms"),
            "shap_status": message.get("shap_status"),
            "top_features": message.get("top_features"),
        }

        if event is None:
            event = RiskEvent(**values)
            db.add(event)
        else:
            for field, value in values.items():
                setattr(event, field, value)

        db.commit()
        logger.info(
            "SQL risk persisted stay_id=%s window_id=%s status=%s",
            stay_id, window_id, status,
        )
        return True
    except Exception:
        db.rollback()
        logger.exception(
            "SQL risk persistence failed stay_id=%s window_id=%s status=%s",
            stay_id, window_id, status,
        )
        raise
    finally:
        db.close()

def make_risk_message(stay_id, window_id, charttime, status, result, inference, shap_result, window_start=None, window_end=None, source_mode=None, source_type=None, ingestion_time=None, prediction_time=None, event_id=None):
    event_time = parse_timestamp(charttime).isoformat() + "Z"
    return safe_json({
        "schema_version": SCHEMA_VERSION,
        "event_time": event_time,
        "source_mode": source_mode,
        "source_type": source_type,
        "ingestion_time": None if ingestion_time is None else parse_timestamp(ingestion_time).isoformat() + "Z",
        "prediction_time": None if prediction_time is None else parse_timestamp(prediction_time).isoformat() + "Z",
        "event_id": event_id,
        "status": status,
        "stay_id": int(stay_id),
        "window_id": int(window_id),
        "charttime": event_time,
        "window_start": None if window_start is None else parse_timestamp(window_start).isoformat() + "Z",
        "window_end": None if window_end is None else parse_timestamp(window_end).isoformat() + "Z",
        "sequence_length": int(inference["sequence_length"]),
        "prediction": int(inference["prediction"]),
        "threshold": float(inference["threshold"]),
        "xgb_raw_probability": float(inference["xgb_raw_probability"]),
        "xgb_probability": float(inference["xgb_probability"]),
        "lstm_raw_probability": float(inference["lstm_raw_probability"]),
        "lstm_probability": float(inference["lstm_probability"]),
        "ensemble_probability": float(inference["ensemble_probability"]),
        "model_version": str(inference["model_version"]),
        "latency_ms": float(inference.get("latency_ms", 0.0)),
        "features": np.asarray(result["features"]).tolist(),
        "shap_status": shap_result.get("status") if shap_result else "not_available",
        "top_features": top_features_of(shap_result),
        "data_quality": {"missing_temporal_features": int(np.sum(np.asarray(result["missing_mask"], dtype=np.float32) > 0)), "total_temporal_features": int(len(result["missing_mask"]))}
    })
def publish_insufficient_history(producer, stay_id, window_id, charttime, result, sequence, source_mode, source_type, prediction_time, ingestion_time, window_start, window_end):
    latency_ms = None if ingestion_time is None else max(0.0, (prediction_time - parse_timestamp(ingestion_time)).total_seconds() * 1000.0)
    message = safe_json({"schema_version": SCHEMA_VERSION, "event_time": parse_timestamp(charttime).isoformat() + "Z", "status": "INSUFFICIENT_HISTORY", "stay_id": int(stay_id), "window_id": int(window_id), "charttime": parse_timestamp(charttime).isoformat() + "Z", "window_start": window_start.isoformat() + "Z", "window_end": parse_timestamp(window_end).isoformat() + "Z", "sequence_length": int(sequence["sequence_length"]), "required_sequence_length": int(getattr(C, "MAX_SEQ_LEN", 120)), "source_mode": source_mode, "source_type": source_type, "ingestion_time": None if ingestion_time is None else parse_timestamp(ingestion_time).isoformat() + "Z", "prediction_time": prediction_time.isoformat() + "Z", "latency_ms": latency_ms, "data_quality": {"status": "insufficient_history", "missing_temporal_features": int(np.sum(np.asarray(result["missing_mask"], dtype=np.float32) > 0)), "total_temporal_features": int(len(result["missing_mask"]))}, "features": np.asarray(result["features"]).tolist()})
    producer.send(RISK_TOPIC, key=stay_id, value=message).get(timeout=10)
    print(f"INSUFFICIENT_HISTORY: stay_id={stay_id} window_id={window_id} sequence_length={sequence['sequence_length']}/{getattr(C, 'MAX_SEQ_LEN', 120)}")
def publish_result(producer, stay_id, window_id, charttime, status, result, inference_result, shap_result, window_start=None, window_end=None, source_mode=None, source_type=None, ingestion_time=None, prediction_time=None, event_id=None):
    risk_message = make_risk_message(stay_id, window_id, charttime, status, result, inference_result, shap_result, window_start, window_end, source_mode, source_type, ingestion_time, prediction_time, event_id)
    risk_future = producer.send(RISK_TOPIC, key=stay_id, value=risk_message)
    risk_future.get(timeout=10)
    if shap_result is not None:
        explanation_message = safe_json(shap_result)
        explanation_message["schema_version"] = SCHEMA_VERSION
        explanation_message["event_time"] = parse_timestamp(charttime).isoformat() + "Z"
        explanation_message["stay_id"] = int(stay_id)
        explanation_message["window_id"] = int(window_id)
        explanation_message["window_status"] = status
        explanation_message["window_start"] = None if window_start is None else parse_timestamp(window_start).isoformat() + "Z"
        explanation_message["window_end"] = None if window_end is None else parse_timestamp(window_end).isoformat() + "Z"
        explanation_message["source_mode"] = source_mode
        explanation_message["source_type"] = source_type
        explanation_message["ingestion_time"] = None if ingestion_time is None else parse_timestamp(ingestion_time).isoformat() + "Z"
        explanation_message["prediction_time"] = None if prediction_time is None else parse_timestamp(prediction_time).isoformat() + "Z"
        try:
            explanation_future = producer.send(EXPLANATION_TOPIC, key=stay_id, value=explanation_message)
            explanation_future.get(timeout=10)
        except Exception as exc:
            logger.exception("explanation publish failed stay_id=%s window_id=%s status=%s", stay_id, window_id, status)
            send_dlq(producer, {"type": "explanation_publish_failed", "stay_id": int(stay_id), "window_id": int(window_id), "status": status, "error": str(exc)})
    print(f"{status}: stay_id={stay_id} window_id={window_id} ensemble={float(inference_result['ensemble_probability']):.6f} prediction={int(inference_result['prediction'])} xgb={float(inference_result['xgb_probability']):.6f} lstm={float(inference_result['lstm_probability']):.6f}")
    if shap_result is not None and shap_result.get("status") == "ok":
        top = top_features_of(shap_result)[:3]
        print(f"SHAP {status}: {[(x.get('feature'), round(float(x.get('shap_value', 0.0)), 4)) for x in top]}")
def prune_window_cache(stay_id, window_cache, keep):
    cached = window_cache.get(stay_id)
    if not cached:
        return
    if len(cached) > keep:
        for old_id in sorted(cached)[:-keep]:
            del cached[old_id]
def window_bounds(intime, window_id):
    start = parse_timestamp(intime) + timedelta(minutes=int(window_id) * C.WINDOW_MINUTES)
    return start, start + timedelta(minutes=C.WINDOW_MINUTES)
def clear_live_window(live_chart_batches, stay_id, window_start, window_end):
    batches = live_chart_batches.get(stay_id)
    if not batches:
        return
    for charttime in list(batches):
        if window_start <= charttime < window_end:
            del batches[charttime]
def attempt_final(stay_id, window_id, preprocessor, producer, inference, shap_engine, static_data, window_cache, finalized, final_attempts, final_retry_after, window_last_charttime, source_modes, window_metadata, live_state_manager, live_chart_batches, finalized_risk, final_payloads, final_risk_sent, final_explanation_sent, last_finalized_window, error_counts):
    key = (stay_id, int(window_id))
    if key in finalized:
        return "ok"
    if kafka_breaker_active():
        final_retry_after[key] = time.monotonic() + KAFKA_BREAKER_COOLDOWN_SECONDS
        return "retry"
    last_finalized = last_finalized_window.get(stay_id)
    if last_finalized is not None and int(window_id) <= int(last_finalized):
        finalized.add(key)
        final_attempts.pop(key, None)
        final_retry_after.pop(key, None)
        final_payloads.pop(key, None)
        final_risk_sent.pop(key, None)
        final_explanation_sent.pop(key, None)
        error_counts["finals_skipped_out_of_order"] += 1
        send_dlq(producer, {"type": "final_skipped_out_of_order", "stay_id": stay_id, "window_id": int(window_id), "last_finalized_window": int(last_finalized)})
        final_id = f"{stay_id}:{int(window_id)}:FINAL"
        try:
            failed_time = parse_timestamp(window_last_charttime.get(stay_id, {}).get(int(window_id), window_bounds(parse_timestamp(static_data[stay_id]["intime"]), int(window_id))[1]))
            failed_message = safe_json({"schema_version": SCHEMA_VERSION, "event_time": failed_time.isoformat() + "Z", "status": "FINAL_FAILED", "stay_id": int(stay_id), "window_id": int(window_id), "charttime": failed_time.isoformat() + "Z", "event_id": final_id, "final_id": final_id, "reason": "out_of_order_final_skip", "last_finalized_window": int(last_finalized)})
            producer.send(RISK_TOPIC, key=stay_id, value=failed_message).get(timeout=10)
        except Exception:
            logger.exception("out-of-order FINAL_FAILED publish failed stay_id=%s window_id=%s", stay_id, window_id)
        try:
            intime = parse_timestamp(static_data[stay_id]["intime"])
            window_start, window_end = window_bounds(intime, int(window_id))
            live_state_manager.clear_window(stay_id, int(window_id))
            clear_live_window(live_chart_batches, stay_id, window_start, window_end)
        except Exception:
            logger.exception("out-of-order FINAL cleanup failed stay_id=%s window_id=%s", stay_id, window_id)
        return "ok"
    if time.monotonic() < final_retry_after.get(key, 0.0):
        return "retry"
    intime = parse_timestamp(static_data[stay_id]["intime"])
    window_start, window_end = window_bounds(intime, int(window_id))
    final_charttime = window_last_charttime.get(stay_id, {}).get(int(window_id), window_end)
    payload = final_payloads.get(key)
    if payload is None:
        try:
            final_result = preprocessor.get_final_window(stay_id, int(window_id), static_data[stay_id])
            window_cache[stay_id][int(window_id)] = final_result
            metadata = window_metadata[stay_id].get(int(window_id), {})
            final_payloads[key] = {
                "result": final_result,
                "charttime": parse_timestamp(final_charttime),
                "window_source_mode": metadata.get("source_mode", source_modes.get(stay_id)),
                "window_source_type": metadata.get("source_type"),
                "window_ingestion_time": metadata.get("ingestion_time")
            }
            payload = final_payloads[key]
        except Exception as exc:
            if is_broker_down(exc) or kafka_breaker_active():
                if is_broker_down(exc): trip_kafka_breaker(f"FINAL preprocessing dependency unavailable: {exc}")
                final_retry_after[key] = time.monotonic() + KAFKA_BREAKER_COOLDOWN_SECONDS
                return "retry"
            final_attempts[key] += 1
            final_retry_after[key] = time.monotonic() + min(300.0, FINAL_RETRY_BASE_SECONDS * (2 ** final_attempts[key]))
            logger.exception("FINAL preprocessing failed stay_id=%s window_id=%s attempt=%s", stay_id, window_id, final_attempts[key])
            send_dlq(producer, {"type": "final_failed", "stay_id": stay_id, "window_id": int(window_id), "charttime": parse_timestamp(final_charttime).isoformat() + "Z", "attempt": final_attempts[key], "max_attempts": MAX_FINAL_ATTEMPTS, "error": str(exc)})
            if final_attempts[key] < MAX_FINAL_ATTEMPTS:
                return "retry"
            finalized.add(key)
            final_attempts.pop(key, None)
            final_retry_after.pop(key, None)
            final_payloads.pop(key, None)
            try:
                live_state_manager.clear_window(stay_id, int(window_id))
                clear_live_window(live_chart_batches, stay_id, window_start, window_end)
            except Exception:
                logger.exception("FINAL_FAILED cleanup failed stay_id=%s window_id=%s", stay_id, window_id)
                send_dlq(producer, {"type": "final_failed_cleanup_failed", "stay_id": stay_id, "window_id": int(window_id)})
            error_counts["finals_abandoned"] += 1
            final_id = f"{stay_id}:{int(window_id)}:FINAL"
            final_failed_message = safe_json({"schema_version": SCHEMA_VERSION, "event_time": parse_timestamp(final_charttime).isoformat() + "Z", "status": "FINAL_FAILED", "stay_id": stay_id, "window_id": int(window_id), "charttime": parse_timestamp(final_charttime).isoformat() + "Z", "event_id": final_id, "final_id": final_id, "error": str(exc), "attempts": MAX_FINAL_ATTEMPTS})
            try:
                producer.send(RISK_TOPIC, key=stay_id, value=final_failed_message).get(timeout=10)
            except Exception:
                logger.exception("FINAL_FAILED publish failed stay_id=%s window_id=%s", stay_id, window_id)
            return "abandoned"
    final_charttime = payload["charttime"] if payload is not None else parse_timestamp(final_charttime)
    window_source_mode = payload.get("window_source_mode", source_modes.get(stay_id)) if payload is not None else source_modes.get(stay_id)
    window_source_type = payload.get("window_source_type") if payload is not None else None
    window_ingestion_time = payload.get("window_ingestion_time") if payload is not None else None
    if payload is not None and "risk" not in payload:
        try:
            final_result = payload["result"]
            window_cache[stay_id][int(window_id)] = final_result
            if "inference" not in payload:
                final_sequence = build_sequence(preprocessor, stay_id, window_cache, int(window_id))
                prediction_time = datetime.now(timezone.utc).replace(tzinfo=None)
                final_inference, final_shap = run_scoring(stay_id, int(window_id), final_charttime, final_result, final_sequence, inference, shap_engine, "FINAL")
                payload["inference"] = final_inference
                payload["shap"] = final_shap
                payload["prediction_time"] = prediction_time
            else:
                final_inference = payload["inference"]
                final_shap = payload["shap"]
                prediction_time = payload["prediction_time"]
            final_id = f"{stay_id}:{int(window_id)}:FINAL"
            final_risk_message = make_risk_message(stay_id, int(window_id), final_charttime, "FINAL", final_result, final_inference, final_shap, window_start, window_end, window_source_mode, window_source_type, window_ingestion_time, prediction_time, final_id)
            final_risk_message["final_id"] = final_id
            final_explanation_message = None
            if final_shap is not None:
                final_explanation_message = safe_json(final_shap)
                final_explanation_message["schema_version"] = SCHEMA_VERSION
                final_explanation_message["event_time"] = parse_timestamp(final_charttime).isoformat() + "Z"
                final_explanation_message["stay_id"] = int(stay_id)
                final_explanation_message["window_id"] = int(window_id)
                final_explanation_message["window_status"] = "FINAL"
                final_explanation_message["window_start"] = window_start.isoformat() + "Z"
                final_explanation_message["window_end"] = window_end.isoformat() + "Z"
                final_explanation_message["source_mode"] = window_source_mode
                final_explanation_message["source_type"] = window_source_type
                final_explanation_message["ingestion_time"] = None if window_ingestion_time is None else parse_timestamp(window_ingestion_time).isoformat() + "Z"
                final_explanation_message["prediction_time"] = prediction_time.isoformat() + "Z"
                final_explanation_message["event_id"] = final_id
                final_explanation_message["final_id"] = final_id
            payload["risk"] = safe_json(final_risk_message)
            payload["explanation"] = final_explanation_message
            payload["window_id"] = int(window_id)
            payload["ensemble_probability"] = float(final_inference["ensemble_probability"])
            payload["xgb_probability"] = float(final_inference["xgb_probability"])
            payload["lstm_probability"] = float(final_inference["lstm_probability"])
        except Exception as exc:
            if is_broker_down(exc) or kafka_breaker_active():
                if is_broker_down(exc): trip_kafka_breaker(f"FINAL scoring dependency unavailable: {exc}")
                final_retry_after[key] = time.monotonic() + KAFKA_BREAKER_COOLDOWN_SECONDS
                return "retry"
            final_attempts[key] += 1
            final_retry_after[key] = time.monotonic() + min(300.0, FINAL_RETRY_BASE_SECONDS * (2 ** final_attempts[key]))
            logger.exception("FINAL scoring failed stay_id=%s window_id=%s attempt=%s", stay_id, window_id, final_attempts[key])
            send_dlq(producer, {"type": "final_failed", "stay_id": stay_id, "window_id": int(window_id), "charttime": parse_timestamp(final_charttime).isoformat() + "Z", "attempt": final_attempts[key], "max_attempts": MAX_FINAL_ATTEMPTS, "error": str(exc)})
            if final_attempts[key] < MAX_FINAL_ATTEMPTS:
                return "retry"
            finalized.add(key)
            final_attempts.pop(key, None)
            final_retry_after.pop(key, None)
            final_payloads.pop(key, None)
            try:
                live_state_manager.clear_window(stay_id, int(window_id))
                clear_live_window(live_chart_batches, stay_id, window_start, window_end)
            except Exception:
                logger.exception("FINAL_FAILED cleanup failed stay_id=%s window_id=%s", stay_id, window_id)
                send_dlq(producer, {"type": "final_failed_cleanup_failed", "stay_id": stay_id, "window_id": int(window_id)})
            error_counts["finals_abandoned"] += 1
            final_id = f"{stay_id}:{int(window_id)}:FINAL"
            final_failed_message = safe_json({"schema_version": SCHEMA_VERSION, "event_time": parse_timestamp(final_charttime).isoformat() + "Z", "status": "FINAL_FAILED", "stay_id": stay_id, "window_id": int(window_id), "charttime": parse_timestamp(final_charttime).isoformat() + "Z", "event_id": final_id, "final_id": final_id, "error": str(exc), "attempts": MAX_FINAL_ATTEMPTS})
            try:
                producer.send(RISK_TOPIC, key=stay_id, value=final_failed_message).get(timeout=10)
            except Exception:
                logger.exception("FINAL_FAILED publish failed stay_id=%s window_id=%s", stay_id, window_id)
            return "abandoned"
    payload = final_payloads[key]
    if not final_risk_sent.get(key, False):
        try:
            persist_risk_event_sql(payload["risk"])
            producer.send(RISK_TOPIC, key=stay_id, value=payload["risk"]).get(timeout=3)
            final_risk_sent[key] = True
        except Exception as exc:
            if is_broker_down(exc):
                trip_kafka_breaker(f"FINAL risk publish failed: {exc}")
                final_retry_after[key] = time.monotonic() + KAFKA_BREAKER_COOLDOWN_SECONDS
                return "retry"
            final_attempts[key] += 1
            logger.exception("FINAL risk publish failed stay_id=%s window_id=%s attempt=%s", stay_id, window_id, final_attempts[key])
            send_dlq(producer, {"type": "final_risk_publish_failed", "stay_id": stay_id, "window_id": int(window_id), "error": str(exc), "attempt": final_attempts[key], "max_attempts": MAX_FINAL_ATTEMPTS})
            if final_attempts[key] < MAX_FINAL_ATTEMPTS:
                final_retry_after[key] = time.monotonic() + min(300.0, FINAL_RETRY_BASE_SECONDS * (2 ** final_attempts[key]))
                return "retry"
            finalized.add(key)
            final_retry_after.pop(key, None)
            final_attempts.pop(key, None)
            final_payloads.pop(key, None)
            final_risk_sent.pop(key, None)
            final_explanation_sent.pop(key, None)
            try:
                live_state_manager.clear_window(stay_id, int(window_id))
                clear_live_window(live_chart_batches, stay_id, window_start, window_end)
            except Exception:
                logger.exception("FINAL_FAILED cleanup failed stay_id=%s window_id=%s", stay_id, window_id)
                send_dlq(producer, {"type": "final_failed_cleanup_failed", "stay_id": stay_id, "window_id": int(window_id)})
            error_counts["finals_abandoned"] += 1
            final_id = f"{stay_id}:{int(window_id)}:FINAL"
            final_failed_message = safe_json({"schema_version": SCHEMA_VERSION, "event_time": parse_timestamp(final_charttime).isoformat() + "Z", "status": "FINAL_FAILED", "stay_id": stay_id, "window_id": int(window_id), "charttime": parse_timestamp(final_charttime).isoformat() + "Z", "event_id": final_id, "final_id": final_id, "error": str(exc), "attempts": MAX_FINAL_ATTEMPTS})
            try:
                producer.send(RISK_TOPIC, key=stay_id, value=final_failed_message).get(timeout=10)
            except Exception:
                logger.exception("FINAL_FAILED publish failed stay_id=%s window_id=%s", stay_id, window_id)
            return "abandoned"
    if final_risk_sent.get(key, False):
        discard_pending_live_explanations(stay_id, int(window_id))
    if payload["explanation"] is not None and not final_explanation_sent.get(key, False):
        try:
            producer.send(EXPLANATION_TOPIC, key=stay_id, value=payload["explanation"]).get(timeout=10)
            final_explanation_sent[key] = True
        except Exception as exc:
            if is_broker_down(exc):
                trip_kafka_breaker(f"FINAL explanation publish failed: {exc}")
                final_retry_after[key] = time.monotonic() + KAFKA_BREAKER_COOLDOWN_SECONDS
                return "retry"
            final_attempts[key] += 1
            logger.exception("FINAL explanation publish failed stay_id=%s window_id=%s attempt=%s", stay_id, window_id, final_attempts[key])
            send_dlq(producer, {"type": "final_explanation_publish_failed", "stay_id": stay_id, "window_id": int(window_id), "error": str(exc), "attempt": final_attempts[key], "max_attempts": MAX_FINAL_ATTEMPTS})
            final_explanation_sent[key] = False
            if final_attempts[key] < MAX_FINAL_ATTEMPTS:
                final_retry_after[key] = time.monotonic() + min(300.0, FINAL_RETRY_BASE_SECONDS * (2 ** final_attempts[key]))
                return "retry"
            final_retry_after[key] = time.monotonic() + min(
                300.0,
                FINAL_RETRY_BASE_SECONDS * (2 ** MAX_FINAL_ATTEMPTS),
            )
            logger.error(
                "FINAL explanation still pending after retry budget "
                "stay_id=%s window_id=%s final_id=%s",
                stay_id,
                window_id,
                f"{stay_id}:{int(window_id)}:FINAL",
            )
            return "retry"
    if finalized_risk.get(stay_id) is None or int(window_id) >= int(finalized_risk[stay_id]["window_id"]):
        finalized_risk[stay_id] = {
            "window_id": int(window_id),
            "charttime": payload["charttime"],
            "ensemble_probability": payload["ensemble_probability"],
            "xgb_probability": payload["xgb_probability"],
            "lstm_probability": payload["lstm_probability"]
        }
    try:
        live_state_manager.clear_window(stay_id, int(window_id))
        clear_live_window(live_chart_batches, stay_id, window_start, window_end)
    except Exception as exc:
        logger.exception("FINAL state cleanup failed stay_id=%s window_id=%s", stay_id, window_id)
        send_dlq(producer, {"type": "final_state_cleanup_failed", "stay_id": stay_id, "window_id": int(window_id), "error": str(exc)})
        finalized.add(key)
        discard_pending_live_explanations(stay_id, int(window_id))
        if last_finalized_window.get(stay_id) is None or int(window_id) > int(last_finalized_window[stay_id]):
            last_finalized_window[stay_id] = int(window_id)
        final_attempts.pop(key, None)
        final_retry_after.pop(key, None)
        final_payloads.pop(key, None)
        final_risk_sent.pop(key, None)
        final_explanation_sent.pop(key, None)
        return "ok"
    finalized.add(key)
    discard_pending_live_explanations(stay_id, int(window_id))
    if last_finalized_window.get(stay_id) is None or int(window_id) > int(last_finalized_window[stay_id]):
        last_finalized_window[stay_id] = int(window_id)
    final_attempts.pop(key, None)
    final_retry_after.pop(key, None)
    final_payloads.pop(key, None)
    final_risk_sent.pop(key, None)
    final_explanation_sent.pop(key, None)
    return "ok"

def retry_due_finals(preprocessor, producer, inference, shap_engine, static_data, window_cache, finalized, final_attempts, final_retry_after, window_last_charttime, source_modes, window_metadata, live_state_manager, live_chart_batches, finalized_risk, final_payloads, final_risk_sent, final_explanation_sent, last_finalized_window, error_counts):
    now = time.monotonic()
    due = sorted((key for key, retry_at in final_retry_after.items() if retry_at <= now and key not in finalized), key=lambda item: (item[0], item[1]))
    for stay_id, window_id in due:
        older_pending = any(other_stay == stay_id and other_window < window_id and (other_stay, other_window) not in finalized for other_stay, other_window in final_retry_after)
        if older_pending:
            continue
        attempt_final(stay_id, window_id, preprocessor, producer, inference, shap_engine, static_data, window_cache, finalized, final_attempts, final_retry_after, window_last_charttime, source_modes, window_metadata, live_state_manager, live_chart_batches, finalized_risk, final_payloads, final_risk_sent, final_explanation_sent, last_finalized_window, error_counts)

def process_observation(preprocessor, producer, inference, shap_engine, static_data, window_cache, last_seen, finalized, final_attempts, final_retry_after, window_last_charttime, max_event_time, source_modes, processed_event_ids, processed_event_order, processed_fallback_keys, processed_fallback_order, window_metadata, live_state_module, live_state_manager, live_observations, live_observation_times, live_chart_batches, finalized_risk, final_payloads, final_risk_sent, final_explanation_sent, last_finalized_window, observation, error_counts):
    stay_id = int(observation["stay_id"])
    if stay_id not in static_data:
        raise RuntimeError(f"Missing static data for stay {stay_id}")
    source_mode = str(observation.get("source_mode", "unknown"))
    previous_source_mode = source_modes.get(stay_id)
    if previous_source_mode is not None and previous_source_mode != source_mode:
        raise RuntimeError(f"Mixed source_mode for stay {stay_id}: {previous_source_mode} -> {source_mode}")
    source_modes[stay_id] = source_mode
    charttime = parse_timestamp(observation["charttime"])
    intime = parse_timestamp(static_data[stay_id]["intime"])
    ingestion_time = parse_timestamp(observation["ingestion_time"])
    if charttime < intime:
        raise RuntimeError(f"Observation occurs before ICU intime for stay {stay_id}")
    if source_mode == "realtime" and (ingestion_time - charttime).total_seconds() > ALLOWED_LATENESS_SECONDS:
        error_counts["late_observations"] += 1
        logger.warning("Late realtime observation dropped stay_id=%s charttime=%s ingestion_time=%s", stay_id, charttime.isoformat(), ingestion_time.isoformat())
        send_dlq(producer, {"type": "late_realtime_observation", "stay_id": stay_id, "charttime": charttime.isoformat() + "Z", "ingestion_time": ingestion_time.isoformat() + "Z", "allowed_lateness_seconds": ALLOWED_LATENESS_SECONDS, "event_id": observation.get("event_id")})
        return True
    window_id = int(preprocessor.get_window_id(charttime, intime))
    last_finalized = last_finalized_window.get(stay_id)
    if (stay_id, window_id) in finalized or (stay_id, window_id) in final_payloads or (stay_id, window_id) in final_attempts or (last_finalized is not None and window_id <= int(last_finalized)):
        error_counts["late_after_finalization"] += 1
        logger.warning("Late observation arrived after finalization: stay_id=%s window_id=%s charttime=%s", stay_id, window_id, charttime.isoformat())
        send_dlq(producer, {"type": "late_after_finalization", "stay_id": stay_id, "window_id": window_id, "charttime": charttime.isoformat() + "Z", "vital_sign": observation["vital_sign"], "event_id": observation.get("event_id"), "last_finalized_window": None if last_finalized is None else int(last_finalized)})
        return True
    dedupe_key = observation_dedupe_key(observation)
    fallback_key = observation_fallback_key(observation)
    if dedupe_key in processed_event_ids[stay_id] or fallback_key in processed_fallback_keys[stay_id]:
        error_counts["duplicate_observations"] += 1
        logger.warning("Duplicate observation ignored stay_id=%s vital=%s charttime=%s event_id=%s", stay_id, observation["vital_sign"], charttime.isoformat(), observation.get("event_id"))
        return True
    preprocessor.add_observation(stay_id, charttime, intime, observation["vital_sign"], float(observation["value"]))
    processed_event_ids[stay_id][dedupe_key] = None
    processed_event_order[stay_id].append(dedupe_key)
    processed_fallback_keys[stay_id][fallback_key] = None
    processed_fallback_order[stay_id].append(fallback_key)
    if len(processed_event_order[stay_id]) > MAX_EVENT_IDS_PER_STAY:
        old_key = processed_event_order[stay_id].popleft()
        processed_event_ids[stay_id].pop(old_key, None)
    if len(processed_fallback_order[stay_id]) > MAX_EVENT_IDS_PER_STAY:
        old_key = processed_fallback_order[stay_id].popleft()
        processed_fallback_keys[stay_id].pop(old_key, None)
    seen = time.monotonic()
    previous_seen = last_seen.get(stay_id)
    last_seen[stay_id] = (seen, charttime if previous_seen is None else max(previous_seen[1], charttime))
    previous_max_event_time = max_event_time.get(stay_id)
    if previous_max_event_time is None or charttime > previous_max_event_time:
        max_event_time[stay_id] = charttime
    window_last_charttime.setdefault(stay_id, {})[window_id] = max(charttime, window_last_charttime.get(stay_id, {}).get(window_id, charttime))
    result = preprocessor.get_partial_window(stay_id, window_id, static_data[stay_id])
    cache = window_cache[stay_id]
    window_metadata[stay_id][window_id] = {"source_mode": source_mode, "source_type": str(observation.get("source_type", "unknown")), "ingestion_time": ingestion_time, "event_id": str(observation.get("event_id", ""))}
    cache[window_id] = result
    vital_sign = observation["vital_sign"]
    value = float(observation["value"])
    previous_charttime = live_observation_times[stay_id].get(vital_sign)
    if previous_charttime is None or charttime >= previous_charttime:
        live_observations[stay_id][vital_sign] = value
        live_observation_times[stay_id][vital_sign] = charttime
    live_chart_batches[stay_id].setdefault(charttime, set()).add(vital_sign)
    if live_chart_batches[stay_id][charttime] >= LIVE_UPDATE_VITALS:
        try:
            sequence = build_sequence(preprocessor, stay_id, window_cache, window_id)
            window_start, window_end = window_bounds(intime, window_id)
            published = publish_live_update(producer, stay_id, window_id, charttime, result, sequence, inference, shap_engine, live_state_module, live_state_manager, live_observations[stay_id], finalized_risk, source_mode, str(observation.get("source_type", "unknown")), ingestion_time, str(observation.get("event_id", "")), window_start, window_end, error_counts=error_counts)
            if not published and kafka_breaker_active():
                error_counts["live_updates_skipped_breaker"] += 1
        except Exception as exc:
            error_counts["live_update_failures"] += 1
            logger.exception("LIVE UPDATE failed stay_id=%s window_id=%s charttime=%s", stay_id, window_id, charttime.isoformat())
            send_dlq(producer, {"type": "live_update_failed", "stay_id": stay_id, "window_id": window_id, "charttime": charttime.isoformat() + "Z", "error": str(exc), "event_id": observation.get("event_id")})
        del live_chart_batches[stay_id][charttime]
    watermark = max_event_time[stay_id] - timedelta(seconds=WATERMARK_SECONDS)
    keep = int(getattr(C, "MAX_SEQ_LEN", 120)) + 1
    eligible_windows = []
    for candidate_window_id in sorted(cache):
        if (stay_id, int(candidate_window_id)) in finalized:
            continue
        _, candidate_window_end = window_bounds(intime, int(candidate_window_id))
        if candidate_window_end <= watermark:
            eligible_windows.append(int(candidate_window_id))
    for candidate_window_id in eligible_windows:
        status = attempt_final(stay_id, candidate_window_id, preprocessor, producer, inference, shap_engine, static_data, window_cache, finalized, final_attempts, final_retry_after, window_last_charttime, source_modes, window_metadata, live_state_manager, live_chart_batches, finalized_risk, final_payloads, final_risk_sent, final_explanation_sent, last_finalized_window, error_counts)
        if status == "retry":
            break
    prune_window_cache(stay_id, window_cache, keep)
    return True

def refresh_due_live_updates(preprocessor, producer, inference, shap_engine, static_data, window_cache, live_state_module, live_state_manager, live_observation_map, live_observation_times, window_last_charttime, window_metadata, finalized, final_attempts, final_payloads, finalized_risk, error_counts=None):
    if DISABLE_LIVE_REFRESH or LIVE_UPDATE_REFRESH_SECONDS == 0 or kafka_breaker_active():
        return
    deadline = time.monotonic() + LIVE_REFRESH_SWEEP_BUDGET_SECONDS
    for raw_stay_id in live_state_manager.active_stays():
        if time.monotonic() >= deadline:
            break
        if kafka_breaker_active():
            break
        stay_id = int(raw_stay_id)
        now = time.monotonic()
        last_attempt = _last_live_update_monotonic.get(stay_id)
        if last_attempt is None or now - last_attempt < LIVE_UPDATE_REFRESH_SECONDS:
            continue
        if stay_id not in static_data:
            continue
        previous_state = live_state_manager.get(stay_id)
        if previous_state is None:
            continue
        window_id = int(previous_state.window_id)
        key = (stay_id, window_id)
        if key in finalized or key in final_attempts or key in final_payloads:
            continue
        metadata = window_metadata.get(stay_id, {}).get(window_id, {})
        if str(metadata.get("source_mode", "")).lower() != "realtime":
            continue
        _last_live_update_monotonic[stay_id] = now
        try:
            result = preprocessor.get_partial_window(stay_id, window_id, static_data[stay_id])
            window_cache[stay_id][window_id] = result
            sequence = build_sequence(preprocessor, stay_id, window_cache, window_id)
            charttime = window_last_charttime.get(stay_id, {}).get(window_id, parse_timestamp(previous_state.charttime))
            intime = parse_timestamp(static_data[stay_id]["intime"])
            window_start, window_end = window_bounds(intime, window_id)
            ingestion_time = metadata.get("ingestion_time")
            event_id = f"refresh:{stay_id}:{window_id}:{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}"
            live_observations = live_observation_map.get(stay_id, {})
            source_mode = metadata.get("source_mode")
            source_type = metadata.get("source_type", "refresh")
            published = publish_live_update(producer, stay_id, window_id, charttime, result, sequence, inference, shap_engine, live_state_module, live_state_manager, live_observations, finalized_risk, source_mode, source_type, ingestion_time, event_id, window_start, window_end, refresh=True, error_counts=error_counts)
            if not published:
                # Deliberate state rejection or breaker refusal is not an exception;
                # retain the normal refresh interval to avoid repeated inference.
                _last_live_update_monotonic[stay_id] = time.monotonic()
        except Exception as exc:
            _last_live_update_monotonic[stay_id] = time.monotonic() - max(
                0.0, LIVE_UPDATE_REFRESH_SECONDS - LIVE_REFRESH_RETRY_SECONDS
            )
            if error_counts is not None:
                error_counts["live_update_failures"] += 1
                error_counts["live_refresh_failures"] += 1
            logger.exception("periodic live UPDATE refresh failed stay_id=%s window_id=%s", stay_id, window_id)


def clear_preprocessor_stay(preprocessor, stay_id):
    for attr in ("active_windows", "locf_state"):
        state = getattr(preprocessor, attr, None)
        if isinstance(state, dict):
            state.pop(stay_id, None)
def flush_discharge_finals(stay_id, preprocessor, producer, inference, shap_engine, static_data, window_cache, finalized, final_attempts, final_retry_after, window_last_charttime, source_modes, window_metadata, live_state_manager, live_chart_batches, finalized_risk, final_payloads, final_risk_sent, final_explanation_sent, last_finalized_window, error_counts, hard_deadline=False):
    try:
        pending_windows = set(int(window_id) for window_id in window_cache.get(stay_id, {}) if (stay_id, int(window_id)) not in finalized)
        active_windows = getattr(preprocessor, "active_windows", {})
        if not isinstance(active_windows, dict):
            raise RuntimeError("preprocessor.active_windows is not a dict")
        stay_windows = active_windows.get(stay_id, {})
        if not isinstance(stay_windows, dict):
            raise RuntimeError(f"preprocessor.active_windows[{stay_id}] is not a dict")
        pending_windows.update(int(window_id) for window_id in stay_windows if (stay_id, int(window_id)) not in finalized)
        pending_windows.update(int(window_id) for current_stay, window_id in final_payloads if current_stay == stay_id and (stay_id, int(window_id)) not in finalized)
        pending_windows.update(int(window_id) for current_stay, window_id in final_attempts if current_stay == stay_id and (stay_id, int(window_id)) not in finalized)
        for window_id in sorted(pending_windows):
            key = (stay_id, window_id)
            # Respect retry scheduling after the first discharge-flush attempt.
            # A hard deadline may still force an explicit FINAL_FAILED below.
            if final_retry_after.get(key, 0.0) > time.monotonic() and not hard_deadline:
                return False
            try:
                status = attempt_final(stay_id, window_id, preprocessor, producer, inference, shap_engine, static_data, window_cache, finalized, final_attempts, final_retry_after, window_last_charttime, source_modes, window_metadata, live_state_manager, live_chart_batches, finalized_risk, final_payloads, final_risk_sent, final_explanation_sent, last_finalized_window, error_counts)
                if status == "retry":
                    # Discharge gets bounded extra attempts for either the risk or
                    # explanation. Do not consume FINAL attempts while the breaker is open.
                    for _retry_index in range(2):
                        if kafka_breaker_active():
                            break
                        # Do not erase exponential backoff; a short pause is not
                        # permission to retry before the scheduled retry time.
                        if final_retry_after.get(key, 0.0) > time.monotonic():
                            break
                        time.sleep(0.1)
                if status == "retry" and final_risk_sent.get(key, False) and not final_explanation_sent.get(key, False):
                    error_counts["discharge_final_explanation_pending"] += 1
                    logger.error(
                        "FINAL risk published but explanation remains pending; "
                        "preserving retry state stay_id=%s window_id=%s hard_deadline=%s",
                        stay_id, window_id, hard_deadline,
                    )
                if status == "retry":
                    error_counts["discharge_final_retries"] += 1
                    if not final_risk_sent.get(key, False) and not kafka_breaker_active() and hard_deadline:
                        final_id = f"{stay_id}:{window_id}:FINAL"
                        failed_charttime = final_payloads.get(key, {}).get("charttime")
                        if failed_charttime is None:
                            failed_charttime = window_last_charttime.get(stay_id, {}).get(window_id)
                        if failed_charttime is None:
                            failed_charttime = window_bounds(parse_timestamp(static_data[stay_id]["intime"]), window_id)[1]
                        failed_charttime = parse_timestamp(failed_charttime)
                        failed_message = safe_json({"schema_version": SCHEMA_VERSION, "event_time": failed_charttime.isoformat() + "Z", "status": "FINAL_FAILED", "stay_id": stay_id, "window_id": window_id, "charttime": failed_charttime.isoformat() + "Z", "event_id": final_id, "final_id": final_id, "error": "discharge_flush_incomplete", "reason": "discharge_flush_incomplete"})
                        try:
                            producer.send(RISK_TOPIC, key=stay_id, value=failed_message).get(timeout=10)
                            finalized.add(key)
                            if last_finalized_window.get(stay_id) is None or window_id > int(last_finalized_window[stay_id]):
                                last_finalized_window[stay_id] = window_id
                            final_attempts.pop(key, None)
                            final_retry_after.pop(key, None)
                            final_payloads.pop(key, None)
                            final_risk_sent.pop(key, None)
                            final_explanation_sent.pop(key, None)
                            try:
                                live_state_manager.clear_window(stay_id, window_id)
                                intime = parse_timestamp(static_data[stay_id]["intime"])
                                _, window_end = window_bounds(intime, window_id)
                                clear_live_window(live_chart_batches, stay_id, window_bounds(intime, window_id)[0], window_end)
                            except Exception:
                                logger.exception("DISCHARGE FINAL_FAILED cleanup failed stay_id=%s window_id=%s", stay_id, window_id)
                        except Exception:
                            logger.exception("DISCHARGE FINAL_FAILED publish failed stay_id=%s window_id=%s", stay_id, window_id)
                            error_counts["discharge_final_failed_publish"] += 1
                    # Do not finalize a later window while this earlier one remains unresolved.
                    if key not in finalized:
                        break
            except Exception as exc:
                error_counts["discharge_final_failures"] += 1
                logger.exception("DISCHARGE FINAL failed stay_id=%s window_id=%s", stay_id, window_id)
                send_dlq(producer, {"type": "discharge_final_failed", "stay_id": stay_id, "window_id": window_id, "error": str(exc)})
                # Preserve strict FINAL ordering: never continue to a later window
                # after an unexpected failure on this one.
                return False
    except Exception as exc:
        error_counts["discharge_flush_errors"] += 1
        logger.exception("DISCHARGE flush failed stay_id=%s", stay_id)
        send_dlq(producer, {"type": "discharge_flush_failed", "stay_id": stay_id, "error": str(exc)})
        return False
    unresolved = [key for key in set(final_payloads) | set(final_attempts) | set(final_risk_sent) | set(final_explanation_sent) if key[0] == stay_id and key not in finalized]
    active_windows = getattr(preprocessor, "active_windows", {})
    active = active_windows.get(stay_id, {}) if isinstance(active_windows, dict) else {}
    unresolved_ids = {key[1] for key in unresolved}
    unresolved_ids.update(int(w) for w in window_cache.get(stay_id, {}) if (stay_id, int(w)) not in finalized)
    if isinstance(active, dict):
        unresolved_ids.update(int(w) for w in active if (stay_id, int(w)) not in finalized)
    return not unresolved_ids

def evict_stay(stay_id, preprocessor, shap_engine, static_data, window_cache, last_seen, finalized, final_attempts, final_retry_after, window_last_charttime, max_event_time, live_state_manager=None, live_observations=None, live_observation_times=None, live_chart_batches=None, source_modes=None, processed_event_ids=None, processed_fallback_keys=None, processed_event_order=None, processed_fallback_order=None, window_metadata=None, finalized_risk=None, final_payloads=None, final_risk_sent=None, final_explanation_sent=None, last_finalized_window=None):
    _last_live_update_monotonic.pop(int(stay_id), None)
    _last_live_update_window.pop(int(stay_id), None)
    for queue_id in [queue_id for queue_id, item in _pending_live_explanations.items() if item["stay_id"] == int(stay_id)]:
        _pending_live_explanations.pop(queue_id, None)
    clear_preprocessor_stay(preprocessor, stay_id)
    window_cache.pop(stay_id, None)
    last_seen.pop(stay_id, None)
    window_last_charttime.pop(stay_id, None)
    max_event_time.pop(stay_id, None)
    finalized.difference_update({item for item in finalized if item[0] == stay_id})
    for key in [key for key in final_attempts if key[0] == stay_id]:
        final_attempts.pop(key, None)
    for key in [key for key in final_retry_after if key[0] == stay_id]:
        final_retry_after.pop(key, None)
    shap_engine.reset(stay_id)
    if live_state_manager is not None:
        live_state_manager.reset(stay_id)
    if live_observations is not None:
        live_observations.pop(stay_id, None)
    if live_observation_times is not None:
        live_observation_times.pop(stay_id, None)
    if live_chart_batches is not None:
        live_chart_batches.pop(stay_id, None)
    if source_modes is not None:
        source_modes.pop(stay_id, None)
    if processed_event_ids is not None:
        processed_event_ids.pop(stay_id, None)
    if processed_fallback_keys is not None:
        processed_fallback_keys.pop(stay_id, None)
    if processed_event_order is not None:
        processed_event_order.pop(stay_id, None)
    if processed_fallback_order is not None:
        processed_fallback_order.pop(stay_id, None)
    if window_metadata is not None:
        window_metadata.pop(stay_id, None)
    if finalized_risk is not None:
        finalized_risk.pop(stay_id, None)
    if final_payloads is not None:
        for key in [key for key in final_payloads if key[0] == stay_id]:
            final_payloads.pop(key, None)
    if final_risk_sent is not None:
        for key in [key for key in final_risk_sent if key[0] == stay_id]:
            final_risk_sent.pop(key, None)
    if final_explanation_sent is not None:
        for key in [key for key in final_explanation_sent if key[0] == stay_id]:
            final_explanation_sent.pop(key, None)
    # Keep last_finalized_window as a tombstone after eviction so late observations
    # cannot recreate an already finalized window and emit a duplicate FINAL.
def expire_pending(pending_observations, pending_last_seen, pending_total, observation_consumer, observation_offset_tracker, error_counts=None):
    now = time.monotonic()
    for stay_id, seen in list(pending_last_seen.items()):
        if now - seen >= PENDING_TTL_SECONDS:
            pending = pending_observations.pop(stay_id, None)
            if pending is not None:
                pending_total[0] -= len(pending)
                dropped = len(pending)
                for _, partition, offset in pending:
                    mark_observation_offset_done(observation_offset_tracker, OBSERVATION_TOPIC, partition, offset)
                commit_ready_observation_offsets(observation_consumer, observation_offset_tracker)
                if error_counts is not None:
                    error_counts["pending_observation_ttl_drops"] += dropped
                logger.error("expired %s buffered observations waiting for static data stay_id=%s", dropped, stay_id)
            pending_last_seen.pop(stay_id, None)
    while pending_total[0] > PENDING_MAX_TOTAL:
        oldest = min(pending_last_seen, key=pending_last_seen.get, default=None)
        if oldest is None:
            break
        pending = pending_observations.pop(oldest, None)
        if pending is not None:
            pending_total[0] -= len(pending)
            dropped = len(pending)
            for _, partition, offset in pending:
                mark_observation_offset_done(observation_offset_tracker, OBSERVATION_TOPIC, partition, offset)
            commit_ready_observation_offsets(observation_consumer, observation_offset_tracker)
            if error_counts is not None:
                error_counts["pending_observation_global_cap_drops"] += dropped
            logger.error("dropped %s buffered observations due to global cap stay_id=%s", dropped, oldest)
        pending_last_seen.pop(oldest, None)
def finalize_idle_stays(preprocessor, producer, inference, shap_engine, static_data, window_cache, last_seen, finalized, final_attempts, final_retry_after, idle_sweep_cursor, error_counts, window_last_charttime, max_event_time, live_state_manager, live_observations, live_observation_times, live_chart_batches, finalized_risk, final_payloads, final_risk_sent, final_explanation_sent, last_finalized_window, source_modes, processed_event_ids, processed_event_order, processed_fallback_keys, processed_fallback_order, window_metadata):
    deadline = time.monotonic() + IDLE_SWEEP_BUDGET_SECONDS
    now = time.monotonic()
    stays = list(last_seen.items())
    if stays:
        start_index = idle_sweep_cursor[0] % len(stays)
        stays = stays[start_index:] + stays[:start_index]
    else:
        start_index = 0
    processed_stays = 0
    for stay_id, state in stays:
        if time.monotonic() > deadline:
            break
        processed_stays += 1
        age = now - state[0]
        candidate_windows = [int(window_id) for window_id in sorted(window_cache.get(stay_id, {})) if (stay_id, int(window_id)) not in finalized]
        if age >= IDLE_FINALIZE_SECONDS:
            for window_id in candidate_windows:
                if time.monotonic() > deadline:
                    break
                if final_retry_after.get((stay_id, window_id), 0.0) > time.monotonic():
                    break
                status = attempt_final(stay_id, window_id, preprocessor, producer, inference, shap_engine, static_data, window_cache, finalized, final_attempts, final_retry_after, window_last_charttime, source_modes, window_metadata, live_state_manager, live_chart_batches, finalized_risk, final_payloads, final_risk_sent, final_explanation_sent, last_finalized_window, error_counts)
                if status == "retry":
                    break
        remaining = [int(window_id) for window_id in window_cache.get(stay_id, {}) if (stay_id, int(window_id)) not in finalized]
        if age >= EVICT_AFTER_SECONDS and not remaining:
            try:
                evict_stay(stay_id, preprocessor, shap_engine, static_data, window_cache, last_seen, finalized, final_attempts, final_retry_after, window_last_charttime, max_event_time, live_state_manager, live_observations, live_observation_times, live_chart_batches, source_modes, processed_event_ids, processed_fallback_keys, processed_event_order, processed_fallback_order, window_metadata, finalized_risk, final_payloads, final_risk_sent, final_explanation_sent, last_finalized_window=last_finalized_window)
            except Exception:
                logger.exception("stay eviction failed stay_id=%s", stay_id)
    if stays:
        idle_sweep_cursor[0] = (start_index + processed_stays) % len(stays)

def send_dlq(producer, record):
    if not ENABLE_DLQ:
        return False
    if kafka_breaker_active():
        logger.warning("DLQ send skipped while Kafka breaker is open; type=%s", record.get("type") if isinstance(record, dict) else type(record).__name__)
        return False
    try:
        payload = record if isinstance(record, dict) else {"raw_record": record.decode("utf-8", errors="replace") if isinstance(record, (bytes, bytearray)) else str(record)}
        producer.send(DLQ_TOPIC, key=payload.get("stay_id") if isinstance(payload, dict) else None, value=safe_json(payload)).get(timeout=0.25)
        return True
    except Exception as exc:
        logger.warning("failed to publish DLQ record (does not trip global breaker): %s", exc)
        return False
def check_shap_self_test(result):
    if not isinstance(result, dict) or result.get("status") != "ok":
        raise RuntimeError(f"SHAP startup self-test failed or returned no explicit status=ok: {result}")
def safe_next_offset(pending_observations, pending_total, partition, record_offset):
    next_offset = record_offset + 1
    if pending_total[0]:
        lowest = min((offset for queue in pending_observations.values() for _, part, offset in queue if part == partition), default=None)
        if lowest is not None:
            next_offset = min(next_offset, lowest)
    return next_offset
def commit_observation_offset(consumer, committed_offsets, topic, partition, next_offset):
    previous = committed_offsets.get(partition, -1)
    if next_offset <= previous:
        return
    try:
        try:
            metadata = OffsetAndMetadata(next_offset, "", -1)
        except TypeError:
            metadata = OffsetAndMetadata(next_offset, "")
        consumer.commit({TopicPartition(topic, partition): metadata})
    except Exception:
        logger.exception("Observation offset commit FAILED topic=%s partition=%s next_offset=%s; offset remains unacknowledged in local bookkeeping", topic, partition, next_offset)
        return False
    committed_offsets[partition] = next_offset
    return True

def mark_observation_offset_seen(tracker, consumer, topic, partition, offset):
    """Register an observation before processing so later records cannot skip it."""
    key = (topic, int(partition))
    state = tracker.get(key)

    if state is None:
        tp = TopicPartition(topic, int(partition))
        try:
            committed = consumer.committed(tp)
        except Exception:
            committed = None

        next_offset = int(committed) if committed is not None else int(offset)
        state = {"next": next_offset, "status": {}}
        tracker[key] = state

    offset = int(offset)
    if offset >= state["next"]:
        state["status"].setdefault(offset, False)

    return key


def mark_observation_offset_done(tracker, topic, partition, offset):
    """Mark one observation terminal; commit only contiguous completed records."""
    key = (topic, int(partition))
    state = tracker.get(key)
    if state is None:
        return

    offset = int(offset)
    if offset >= state["next"]:
        state["status"][offset] = True


def commit_ready_observation_offsets(consumer, tracker):
    """Advance each observation partition only through completed records."""
    for (topic, partition), state in list(tracker.items()):
        next_offset = state["next"]
        advanced = next_offset

        for offset in sorted(state["status"]):
            if offset < next_offset:
                continue
            if not state["status"][offset]:
                break

            # Allow offset gaps while never crossing an observed unfinished record.
            advanced = offset + 1

        if advanced == next_offset:
            continue

        tp = TopicPartition(topic, partition)
        try:
            try:
                metadata = OffsetAndMetadata(advanced, "", -1)
            except TypeError:
                metadata = OffsetAndMetadata(advanced, "")

            consumer.commit({tp: metadata})
        except Exception:
            logger.exception(
                "Observation contiguous offset commit failed "
                "topic=%s partition=%s next_offset=%s",
                topic, partition, advanced,
            )
            continue

        state["next"] = advanced
        state["status"] = {
            offset: complete
            for offset, complete in state["status"].items()
            if offset >= advanced
        }

def mark_lifecycle_offset_seen(tracker, consumer, topic, partition, offset):
    """Register a lifecycle record before processing it, preserving partition order."""
    key = (topic, int(partition))
    state = tracker.get(key)
    if state is None:
        tp = TopicPartition(topic, int(partition))
        try:
            committed = consumer.committed(tp)
        except Exception:
            committed = None
        # Kafka commits the *next* offset to consume. If no commit exists, start
        # from this first delivered offset (auto_offset_reset governs the start).
        next_offset = int(committed) if committed is not None else int(offset)
        state = {"next": next_offset, "status": {}}
        tracker[key] = state
    state["status"][int(offset)] = False
    return key


def mark_lifecycle_offset_done(tracker, topic, partition, offset):
    key = (topic, int(partition))
    state = tracker.get(key)
    if state is None:
        return
    state["status"][int(offset)] = True


def commit_ready_lifecycle_offsets(consumer, tracker):
    """Commit through observed, completed records in partition order; tolerate offset gaps."""
    for (topic, partition), state in list(tracker.items()):
        next_offset = state["next"]
        advanced = next_offset
        for offset in sorted(state["status"]):
            if offset < next_offset:
                continue
            if not state["status"][offset]:
                break
            # Kafka offsets can have gaps (for example, compacted topics); the
            # next observed record is the next safe advancement boundary.
            advanced = offset + 1
        if advanced == next_offset:
            continue
        tp = TopicPartition(topic, partition)
        try:
            try:
                metadata = OffsetAndMetadata(advanced, "", -1)
            except TypeError:
                metadata = OffsetAndMetadata(advanced, "")
            consumer.commit(offsets={tp: metadata})
        except Exception:
            logger.exception("Lifecycle contiguous offset commit failed topic=%s partition=%s next_offset=%s", topic, partition, advanced)
            continue
        state["next"] = advanced
        state["status"] = {offset: complete for offset, complete in state["status"].items() if offset >= advanced}


def main():
    global _shutdown
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)
    RealtimePreprocessor = load_module("realtime_preprocessing", "01_realtime_preprocessing.py").RealtimePreprocessor
    RealtimeInference = load_module("realtime_inference", "02_realtime_inference.py").RealtimeInference
    RealtimeSHAP = load_module("realtime_shap", "10_realtime_shap.py").RealtimeSHAP
    live_state_module = load_module("live_update_state", "13_live_update_state.py")
    preprocessor = RealtimePreprocessor()
    inference = None
    producer = None
    static_consumer = None
    observation_consumer = None
    lifecycle_consumer = None
    try:
        preprocessor.start()
        configured_xgb_weight = getattr(C, "XGB_WEIGHT", None)
        configured_lstm_weight = getattr(C, "LSTM_WEIGHT", None)
        if not hasattr(C, "XGB_WEIGHT") or not hasattr(C, "LSTM_WEIGHT") or not hasattr(C, "ENSEMBLE_THRESHOLD"):
            raise RuntimeError("Frozen ensemble config missing XGB_WEIGHT, LSTM_WEIGHT, or ENSEMBLE_THRESHOLD")
        configured_threshold = C.ENSEMBLE_THRESHOLD
        if PENDING_MAX_PER_STAY <= 0 or PENDING_MAX_TOTAL <= 0 or MAX_EVENT_IDS_PER_STAY <= 0:
            raise RuntimeError("Pending/event-id capacity limits must be positive")
        try:
            xgb_w, lstm_w, threshold = map(float, (configured_xgb_weight, configured_lstm_weight, configured_threshold))
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Frozen ensemble weights and threshold must be numeric and non-null") from exc
        if not all(math.isfinite(value) for value in (xgb_w, lstm_w, threshold)):
            raise RuntimeError("Frozen ensemble weights and threshold must be finite")
        if abs(xgb_w - FROZEN_XGB_WEIGHT) >= 1e-9:
            raise RuntimeError(f"Frozen XGB ensemble weight drift: {configured_xgb_weight}")
        if abs(lstm_w - FROZEN_LSTM_WEIGHT) >= 1e-9:
            raise RuntimeError(f"Frozen LSTM ensemble weight drift: {configured_lstm_weight}")
        if abs(xgb_w + lstm_w - 1.0) >= 1e-9:
            raise RuntimeError("Frozen ensemble weights must sum to 1.0")
        if abs(threshold - FROZEN_ENSEMBLE_THRESHOLD) >= 1e-9:
            raise RuntimeError(f"Frozen ensemble threshold drift: {configured_threshold}")
        inference = RealtimeInference()
        inference.start()
        shap_engine = RealtimeSHAP(inference.artifacts["xgb_model"], feature_names=inference.artifacts["feature_names"], top_k=10, max_gap_windows=6)
        shap_self_test = shap_engine.startup_self_test()
        check_shap_self_test(shap_self_test)
        live_state_manager = live_state_module.LiveUpdateStateManager(expected_n_features=C.FEATURE_COUNT)
        print(f"SHAP STARTUP SELF-TEST: {shap_self_test}")
        producer = create_producer()
        static_consumer = create_static_consumer()
        observation_consumer = create_consumer(OBSERVATION_TOPIC, OBSERVATION_GROUP_ID, "earliest")
        lifecycle_consumer = create_consumer(LIFECYCLE_TOPIC, "icu_realtime_lifecycle_v1", "earliest", auto_commit=False)
        static_data = {}
        # Bounded in-memory tombstones. This bounds memory but is not a durable
        # replacement for persisted lifecycle state across process restarts.
        discharged_stays = OrderedDict()
        pending_discharges = {}
        lifecycle_offset_tracker = {}
        pending_observations = defaultdict(deque)
        window_cache = defaultdict(dict)
        last_seen = {}
        finalized = set()
        final_attempts = defaultdict(int)
        final_retry_after = {}
        idle_sweep_cursor = [0]
        window_last_charttime = {}
        max_event_time = {}
        source_modes = {}
        processed_event_ids = defaultdict(dict)
        processed_event_order = defaultdict(deque)
        processed_fallback_keys = defaultdict(dict)
        processed_fallback_order = defaultdict(deque)
        window_metadata = defaultdict(dict)
        finalized_risk = {}
        final_payloads = {}
        final_risk_sent = {}
        final_explanation_sent = {}
        last_finalized_window = OrderedDict()
        live_observations = defaultdict(dict)
        live_observation_times = defaultdict(dict)
        live_chart_batches = defaultdict(dict)
        _last_live_update_monotonic.clear()
        _pending_live_explanations.clear()
        pending_last_seen = {}
        pending_total = [0]
        committed_offsets = {}  # Retained for compatibility with legacy diagnostics only.
        observation_offset_tracker = {}
        error_counts = defaultdict(int)
        last_idle_sweep = time.monotonic()
        print("KAFKA REAL-TIME INFERENCE STARTED")
        print(f"Kafka: {BOOTSTRAP_SERVERS}")
        print(f"Risk topic: {RISK_TOPIC}")
        print(f"Explanation topic: {EXPLANATION_TOPIC}")
        print(f"Lifecycle topic: {LIFECYCLE_TOPIC}")
        print(f"DLQ: {'enabled' if ENABLE_DLQ else 'disabled'}")
        while not _shutdown:
            static_records = static_consumer.poll(timeout_ms=100)
            for _, records in static_records.items():
                for record in records:
                    data = None
                    try:
                        data = json.loads(record.value.decode("utf-8"))
                        validate_static(data)
                        stay_id = int(data["stay_id"])
                        data["intime"] = parse_timestamp(data["intime"])
                        static_data[stay_id] = data
                        print(f"STATIC RECEIVED: stay_id={stay_id}")
                        buffered = pending_observations.pop(stay_id, deque())
                        pending_total[0] -= len(buffered)
                        pending_last_seen.pop(stay_id, None)
                        while buffered:
                            observation, partition, offset = buffered.popleft()
                            try:
                                process_observation(preprocessor, producer, inference, shap_engine, static_data, window_cache, last_seen, finalized, final_attempts, final_retry_after, window_last_charttime, max_event_time, source_modes, processed_event_ids, processed_event_order, processed_fallback_keys, processed_fallback_order, window_metadata, live_state_module, live_state_manager, live_observations, live_observation_times, live_chart_batches, finalized_risk, final_payloads, final_risk_sent, final_explanation_sent, last_finalized_window, observation, error_counts)
                            except Exception:
                                logger.exception("failed buffered record stay_id=%s", stay_id)
                                send_dlq(producer, {"type": "buffered_observation_failed", "stay_id": stay_id, "error": "buffered observation processing failed", "record": observation})
                            finally:
                                mark_observation_offset_done(observation_offset_tracker, OBSERVATION_TOPIC, partition, offset)
                                commit_ready_observation_offsets(observation_consumer, observation_offset_tracker)
                    except Exception:
                        error_counts["static_errors"] += 1
                        logger.exception("failed static record")
                        send_dlq(producer, {"type": "static_failed", "stay_id": data.get("stay_id") if isinstance(data, dict) else None, "error": "static record processing failed", "record": record.value.decode("utf-8", errors="replace")})
            retry_due_finals(preprocessor, producer, inference, shap_engine, static_data, window_cache, finalized, final_attempts, final_retry_after, window_last_charttime, source_modes, window_metadata, live_state_manager, live_chart_batches, finalized_risk, final_payloads, final_risk_sent, final_explanation_sent, last_finalized_window, error_counts)
            retry_pending_live_explanations(producer, error_counts)
            refresh_due_live_updates(preprocessor, producer, inference, shap_engine, static_data, window_cache, live_state_module, live_state_manager, live_observations, live_observation_times, window_last_charttime, window_metadata, finalized, final_attempts, final_payloads, finalized_risk, error_counts)
            if time.monotonic() - last_idle_sweep >= IDLE_SWEEP_SECONDS:
                expire_pending(pending_observations, pending_last_seen, pending_total, observation_consumer, observation_offset_tracker, error_counts)
                if not DISABLE_IDLE_FINALIZATION:
                    finalize_idle_stays(preprocessor, producer, inference, shap_engine, static_data, window_cache, last_seen, finalized, final_attempts, final_retry_after, idle_sweep_cursor, error_counts, window_last_charttime, max_event_time, live_state_manager, live_observations, live_observation_times, live_chart_batches, finalized_risk, final_payloads, final_risk_sent, final_explanation_sent, last_finalized_window, source_modes, processed_event_ids, processed_event_order, processed_fallback_keys, processed_fallback_order, window_metadata)
                last_idle_sweep = time.monotonic()
            lifecycle_records = lifecycle_consumer.poll(timeout_ms=100)
            for _, records in lifecycle_records.items():
                for record in records:
                    data = None
                    mark_lifecycle_offset_seen(lifecycle_offset_tracker, lifecycle_consumer, record.topic, record.partition, record.offset)
                    try:
                        data = json.loads(record.value.decode("utf-8"))
                        event_type = str(data.get("event_type", "")).upper()
                        stay_id = int(data["stay_id"])
                        if event_type != "DISCHARGE":
                            logger.warning(
                                "Ignoring unsupported lifecycle event "
                                "stay_id=%s event_type=%s",
                                stay_id,
                                event_type,
                            )
                            mark_lifecycle_offset_done(lifecycle_offset_tracker, record.topic, record.partition, record.offset)
                            continue
                        now_mono = time.monotonic()
                        pending_discharges.setdefault(stay_id, {
                            "due": now_mono + DISCHARGE_GRACE_SECONDS,
                            "deadline": now_mono + DISCHARGE_GRACE_SECONDS + DISCHARGE_HARD_DEADLINE_SECONDS,
                            "data": data,
                            "lifecycle_records": [],
                        })
                        # Track each event offset individually; partition commits are
                        # advanced only through the contiguous-completion tracker.
                        lifecycle_record = (record.topic, int(record.partition), int(record.offset))
                        if lifecycle_record not in pending_discharges[stay_id]["lifecycle_records"]:
                            pending_discharges[stay_id]["lifecycle_records"].append(lifecycle_record)
                        logger.info("DISCHARGE scheduled stay_id=%s grace_seconds=%s", stay_id, DISCHARGE_GRACE_SECONDS)
                    except Exception:
                        error_counts["lifecycle_errors"] += 1
                        logger.exception(
                            "failed lifecycle record stay_id=%s",
                            None if data is None else data.get("stay_id"),
                        )
                        send_dlq(
                            producer,
                            {
                                "type": "lifecycle_failed",
                                "stay_id": (
                                    None
                                    if data is None
                                    else data.get("stay_id")
                                ),
                                "error": "lifecycle record processing failed",
                                "record": (
                                    data
                                    if data is not None
                                    else record.value.decode(
                                        "utf-8",
                                        errors="replace",
                                    )
                                ),
                            },
                        )
                        mark_lifecycle_offset_done(lifecycle_offset_tracker, record.topic, record.partition, record.offset)
            observation_records = observation_consumer.poll(timeout_ms=100)
            for _, records in observation_records.items():
                for record in records:
                    data = None
                    mark_observation_offset_seen(observation_offset_tracker, observation_consumer, record.topic, record.partition, record.offset)
                    try:
                        data = json.loads(record.value.decode("utf-8"))
                        validate_observation(data)
                        normalize_observation_vital(data)
                        validate_observation_values(data)
                        stay_id = int(data["stay_id"])
                        if stay_id in discharged_stays:
                            logger.info(
                                "Ignoring observation for discharged stay_id=%s",
                                stay_id,
                            )
                            mark_observation_offset_done(observation_offset_tracker, record.topic, record.partition, record.offset)
                            commit_ready_observation_offsets(observation_consumer, observation_offset_tracker)
                            continue
                        if stay_id not in static_data:
                            pending = pending_observations[stay_id]
                            if len(pending) >= PENDING_MAX_PER_STAY:
                                _, dropped_partition, dropped_offset = pending.popleft()
                                mark_observation_offset_done(observation_offset_tracker, OBSERVATION_TOPIC, dropped_partition, dropped_offset)
                                commit_ready_observation_offsets(observation_consumer, observation_offset_tracker)
                                pending_total[0] -= 1
                                error_counts["pending_observation_per_stay_cap_drops"] += 1
                                logger.error("pending observation cap reached; oldest buffered record dropped stay_id=%s cap=%s", stay_id, PENDING_MAX_PER_STAY)
                            pending.append((data, record.partition, record.offset))
                            pending_total[0] += 1
                            pending_last_seen[stay_id] = time.monotonic()
                            print(f"OBSERVATION BUFFERED: stay_id={stay_id}")
                            continue
                        process_observation(preprocessor, producer, inference, shap_engine, static_data, window_cache, last_seen, finalized, final_attempts, final_retry_after, window_last_charttime, max_event_time, source_modes, processed_event_ids, processed_event_order, processed_fallback_keys, processed_fallback_order, window_metadata, live_state_module, live_state_manager, live_observations, live_observation_times, live_chart_batches, finalized_risk, final_payloads, final_risk_sent, final_explanation_sent, last_finalized_window, data, error_counts)
                        mark_observation_offset_done(observation_offset_tracker, record.topic, record.partition, record.offset)
                        commit_ready_observation_offsets(observation_consumer, observation_offset_tracker)
                    except Exception as exc:
                        error_counts["observation_errors"] += 1
                        if isinstance(data, dict) and "vital_sign" in data and "value" in data:
                            error_counts["invalid_observations"] += 1
                        logger.exception("failed record stay_id=%s", None if data is None else data.get("stay_id"))
                        send_dlq(producer, {"type": "observation_failed", "stay_id": None if data is None else data.get("stay_id"), "error": "observation record processing failed", "record": data if data is not None else record.value.decode("utf-8", errors="replace")})
                        mark_observation_offset_done(observation_offset_tracker, record.topic, record.partition, record.offset)
                        commit_ready_observation_offsets(observation_consumer, observation_offset_tracker)
            due_discharges = [stay_id for stay_id, state in pending_discharges.items() if time.monotonic() >= state["due"]]
            for stay_id in due_discharges:
                discharge_state = pending_discharges[stay_id]
                deadline_reached = time.monotonic() >= discharge_state["deadline"]
                flush_complete = stay_id not in static_data
                if stay_id in static_data:
                    try:
                        flush_complete = flush_discharge_finals(
                            stay_id, preprocessor, producer, inference, shap_engine, static_data,
                            window_cache, finalized, final_attempts, final_retry_after,
                            window_last_charttime, source_modes, window_metadata, live_state_manager,
                            live_chart_batches, finalized_risk, final_payloads, final_risk_sent,
                            final_explanation_sent, last_finalized_window, error_counts,
                            hard_deadline=deadline_reached,
                        )
                    except Exception:
                        error_counts["discharge_flush_errors"] += 1
                        logger.exception("DISCHARGE flush failed stay_id=%s", stay_id)
                        flush_complete = False
                if not flush_complete and not deadline_reached:
                    discharge_state["due"] = time.monotonic() + min(5.0, KAFKA_BREAKER_COOLDOWN_SECONDS if kafka_breaker_active() else 1.0)
                    logger.warning("DISCHARGE retained for FINAL retry stay_id=%s deadline_in=%.1fs", stay_id, max(0.0, discharge_state["deadline"] - time.monotonic()))
                    continue
                if not flush_complete and deadline_reached and kafka_breaker_active():
                    discharge_state["due"] = time.monotonic() + KAFKA_BREAKER_COOLDOWN_SECONDS
                    logger.critical("DISCHARGE hard deadline passed while Kafka breaker is open; retaining stay until broker recovers stay_id=%s", stay_id)
                    continue
                if not flush_complete and deadline_reached:
                    error_counts["discharge_hard_deadline_exhausted"] += 1
                    logger.critical(
                        "DISCHARGE deadline reached with unresolved FINALs; retaining stay for retry stay_id=%s",
                        stay_id,
                    )
                    send_dlq(producer, {
                        "type": "discharge_hard_deadline_exhausted",
                        "stay_id": stay_id,
                        "reason": "FINAL flush remains incomplete; stay retained for retry",
                    })
                    discharge_state["due"] = time.monotonic() + (
                        KAFKA_BREAKER_COOLDOWN_SECONDS if kafka_breaker_active() else 5.0
                    )
                    continue
                # Acknowledge lifecycle records only after FINAL handling completed or
                # the configured hard-deadline abandonment was explicitly recorded.
                lifecycle_records = discharge_state.get("lifecycle_records", [])
                for topic_name, partition_id, lifecycle_offset in lifecycle_records:
                    mark_lifecycle_offset_done(lifecycle_offset_tracker, topic_name, partition_id, lifecycle_offset)
                commit_ready_lifecycle_offsets(lifecycle_consumer, lifecycle_offset_tracker)
                pending_discharges.pop(stay_id, None)
                discharged_stays[stay_id] = time.monotonic()
                discharged_stays.move_to_end(stay_id)
                while len(discharged_stays) > MAX_DISCHARGED_TOMBSTONES:
                    expired_stay_id, _ = discharged_stays.popitem(last=False)
                    logger.warning("Evicted oldest discharged-stay tombstone stay_id=%s; durable lifecycle dedupe is not enabled", expired_stay_id)
                # Keep the latest finalized-window tombstone bounded too. It is an
                # in-process guard only; durable dedupe requires persisted state.
                last_finalized_window.move_to_end(stay_id) if stay_id in last_finalized_window else None
                while len(last_finalized_window) > MAX_DISCHARGED_TOMBSTONES:
                    old_stay, _ = last_finalized_window.popitem(last=False)
                    logger.warning("Evicted finalized-window tombstone stay_id=%s", old_stay)
                static_data.pop(stay_id, None)
                pending = pending_observations.pop(stay_id, deque())
                pending_total[0] -= len(pending)
                for _, partition, offset in pending:
                    mark_observation_offset_done(observation_offset_tracker, OBSERVATION_TOPIC, partition, offset)
                commit_ready_observation_offsets(observation_consumer, observation_offset_tracker)
                pending_last_seen.pop(stay_id, None)
                try:
                    evict_stay(stay_id, preprocessor, shap_engine, static_data, window_cache, last_seen,
                        finalized, final_attempts, final_retry_after, window_last_charttime, max_event_time,
                        live_state_manager, live_observations, live_observation_times, live_chart_batches,
                        source_modes, processed_event_ids, processed_fallback_keys, processed_event_order,
                        processed_fallback_order, window_metadata, finalized_risk, final_payloads,
                        final_risk_sent, final_explanation_sent, last_finalized_window)
                except Exception:
                    error_counts["lifecycle_errors"] += 1
                    logger.exception("DISCHARGE eviction failed stay_id=%s", stay_id)
                logger.info("DISCHARGED: stay_id=%s realtime monitoring stopped", stay_id)
                print(f"DISCHARGED: stay_id={stay_id} realtime monitoring stopped")
    finally:
        if _shutdown:
            logger.info("Shutdown requested; closing realtime resources")
        close_actions = [
            ("static_consumer", static_consumer),
            ("observation_consumer", observation_consumer),
            ("lifecycle_consumer", lifecycle_consumer),
            ("producer", producer),
            ("inference", inference),
            ("preprocessor", preprocessor)
        ]
        for name, resource in close_actions:
            if resource is None:
                continue
            try:
                if name == "producer":
                    resource.flush(timeout=10)
                    resource.close()
                elif name == "inference":
                    stop = getattr(resource, "stop", None)
                    if callable(stop):
                        stop()
                elif name == "preprocessor":
                    resource.stop()
                else:
                    resource.close()
            except Exception:
                logger.exception("failed to close %s", name)
        print(f"ERROR COUNTS: {dict(error_counts) if 'error_counts' in locals() else {}}")
        print("KAFKA REAL-TIME INFERENCE STOPPED")
if __name__ == "__main__":
    main()