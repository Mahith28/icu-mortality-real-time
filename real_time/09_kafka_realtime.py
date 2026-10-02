import os
import sys
import json
import importlib.util
import signal
import time
from datetime import datetime, timezone, timedelta
import logging
from collections import defaultdict, deque
import numpy as np
from kafka import KafkaConsumer, KafkaProducer, TopicPartition
PROJECT_ROOT = "/home/mahith/BDA_PROJECT"
REAL_TIME_DIR = os.path.join(PROJECT_ROOT, "Model", "real_time")
sys.path.insert(0, REAL_TIME_DIR)
import config as C
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
IDLE_FINALIZE_SECONDS = float(os.environ.get("ICU_IDLE_FINALIZE_SECONDS", "900"))
IDLE_SWEEP_SECONDS = float(os.environ.get("ICU_IDLE_SWEEP_SECONDS", "60"))
EVICT_AFTER_SECONDS = float(os.environ.get("ICU_EVICT_AFTER_SECONDS", "7200"))
PENDING_TTL_SECONDS = float(os.environ.get("ICU_PENDING_TTL_SECONDS", "3600"))
PENDING_MAX_TOTAL = int(os.environ.get("ICU_PENDING_MAX_TOTAL", "100000"))
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
    logger.info("Shutdown signal received: %s", signum)
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
def create_consumer(topic, group_id, auto_offset_reset, auto_commit=True, max_poll_records=20):
    return KafkaConsumer(topic, bootstrap_servers=BOOTSTRAP_SERVERS, auto_offset_reset=auto_offset_reset, enable_auto_commit=auto_commit, group_id=group_id, max_poll_records=max_poll_records, value_deserializer=None)
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
def validate_static(data):
    required = ["stay_id", "anchor_age", "gender", "race", "insurance", "admission_type", "admission_location", "first_careunit", "intime"]
    missing = [field for field in required if field not in data]
    if missing:
        raise RuntimeError(f"Static message missing fields: {missing}")
def validate_observation(data):
    required = ["stay_id", "charttime", "intime", "vital_sign", "value"]
    missing = [field for field in required if field not in data]
    if missing:
        raise RuntimeError(f"Observation message missing fields: {missing}")
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
def run_scoring(stay_id, window_id, timestamp, result, sequence, inference, shap_engine, window_status):
    features = np.asarray(result["features"], dtype=np.float32)
    inference_result = inference.predict(features, np.asarray(sequence["temporal"], dtype=np.float32), np.asarray(sequence["missing_mask"], dtype=np.float32), np.asarray(sequence["padding_mask"], dtype=np.float32), np.asarray(sequence["static"], dtype=np.float32))
    timestamp_value = parse_timestamp(timestamp).isoformat() + "Z"
    try:
        xgb_raw_margin = inference_result.get("xgb_raw_margin")
        if window_status == "FINAL":
            shap_result = shap_engine.update(stay_id, window_id, timestamp, features, inference_result, window_status="FINAL")
            shap_result["stay_id"] = int(stay_id)
        else:
            explanation = shap_engine.explain_live(features, xgb_raw_probability=float(inference_result["xgb_raw_probability"]), xgb_raw_margin=xgb_raw_margin, verify=True)
            shap_result = {"status": "ok", "stay_id": int(stay_id), "window_id": int(window_id), "timestamp": timestamp_value, "window_status": window_status, "previous_window_id": None, "window_gap": None, "previous_discarded": False, "xgb_raw_probability": float(inference_result["xgb_raw_probability"]), "xgb_raw_margin": None if xgb_raw_margin is None else float(xgb_raw_margin), "xgb_probability": float(inference_result["xgb_probability"]), "lstm_probability": float(inference_result["lstm_probability"]), "ensemble_probability": float(inference_result["ensemble_probability"]), "risk": {"ensemble_probability": float(inference_result["ensemble_probability"]), "ensemble_probability_previous": None, "ensemble_probability_delta": None}, "xgb_component": {"raw_probability": float(inference_result["xgb_raw_probability"]), "raw_margin": None if xgb_raw_margin is None else float(xgb_raw_margin), "probability": float(inference_result["xgb_probability"]), "probability_previous": None, "probability_delta": None, "weight": float(C.XGB_WEIGHT)}, "lstm_component": {"probability": float(inference_result["lstm_probability"]), "probability_previous": None, "probability_delta": None, "weight": float(C.LSTM_WEIGHT)}, "shap": explanation, "changed_features": [], "latency_ms": 0.0}
    except Exception as exc:
        logger.exception("SHAP failed stay_id=%s window_id=%s", stay_id, window_id)
        shap_result = {"status": "error", "stay_id": int(stay_id), "window_id": int(window_id), "timestamp": timestamp_value, "window_status": window_status, "error_type": type(exc).__name__, "error": str(exc), "shap": {"top_features": []}}
    return inference_result, safe_json(shap_result)
def make_risk_message(stay_id, window_id, charttime, status, result, inference, shap_result, window_start=None, window_end=None):
    event_time = parse_timestamp(charttime).isoformat() + "Z"
    return safe_json({
        "schema_version": SCHEMA_VERSION,
        "event_time": event_time,
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
        "top_features": top_features_of(shap_result)
    })
def publish_result(producer, stay_id, window_id, charttime, status, result, inference_result, shap_result, window_start=None, window_end=None):
    risk_message = make_risk_message(stay_id, window_id, charttime, status, result, inference_result, shap_result, window_start, window_end)
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
def process_observation(preprocessor, producer, inference, shap_engine, static_data, window_cache, last_window, last_seen, finalized, window_last_charttime, observation):
    stay_id = int(observation["stay_id"])
    charttime = parse_timestamp(observation["charttime"])
    intime = parse_timestamp(observation["intime"])
    window_id = int(preprocessor.add_observation(stay_id, charttime, intime, observation["vital_sign"], float(observation["value"])))
    seen = time.monotonic()
    previous_seen = last_seen.get(stay_id)
    last_seen[stay_id] = (seen, charttime if previous_seen is None else max(previous_seen[1], charttime))
    window_last_charttime.setdefault(stay_id, {})[window_id] = max(charttime, window_last_charttime.get(stay_id, {}).get(window_id, charttime))
    result = preprocessor.get_partial_window(stay_id, window_id, static_data[stay_id])
    previous_window_id = last_window.get(stay_id)
    keep = int(getattr(C, "MAX_SEQ_LEN", 120)) + 1
    cache = window_cache[stay_id]
    is_late = (previous_window_id is not None and window_id < int(previous_window_id)) or (stay_id, window_id) in finalized
    if is_late and cache and len(cache) >= keep and window_id < min(cache):
        logger.info("dropping too-old late observation stay_id=%s window_id=%s", stay_id, window_id)
        return
    cache[window_id] = result
    if is_late:
        print(f"LATE OBSERVATION: stay_id={stay_id} window_id={window_id} last_window={previous_window_id}")
        try:
            sequence = build_sequence(preprocessor, stay_id, window_cache, window_id)
            inference_result, shap_result = run_scoring(stay_id, window_id, charttime, result, sequence, inference, shap_engine, "LATE")
            window_start, window_end = window_bounds(intime, window_id)
            publish_result(producer, stay_id, window_id, charttime, "LATE", result, inference_result, shap_result, window_start, window_end)
        except Exception as exc:
            logger.exception("LATE processing failed stay_id=%s window_id=%s", stay_id, window_id)
            send_dlq(producer, {"type": "late_failed", "stay_id": stay_id, "window_id": window_id, "charttime": str(charttime), "error": str(exc)})
        finally:
            prune_window_cache(stay_id, window_cache, keep)
        return
    if previous_window_id is not None and window_id > int(previous_window_id) and (stay_id, int(previous_window_id)) not in finalized:
        prev_id = int(previous_window_id)
        final_result = cache.get(prev_id)
        if final_result is None:
            logger.warning("missing previous window stay_id=%s window_id=%s", stay_id, prev_id)
            send_dlq(producer, {"type": "final_missing_window", "stay_id": stay_id, "window_id": prev_id, "trigger_charttime": str(charttime)})
        else:
            try:
                final_sequence = build_sequence(preprocessor, stay_id, window_cache, prev_id)
                final_charttime = window_last_charttime.get(stay_id, {}).get(prev_id, charttime)
                final_inference, final_shap = run_scoring(stay_id, prev_id, final_charttime, final_result, final_sequence, inference, shap_engine, "FINAL")
                window_start, window_end = window_bounds(intime, prev_id)
                publish_result(producer, stay_id, prev_id, final_charttime, "FINAL", final_result, final_inference, final_shap, window_start, window_end)
                finalized.add((stay_id, prev_id))
                last_window[stay_id] = window_id
            except Exception as exc:
                logger.exception("FINAL failed stay_id=%s window_id=%s", stay_id, prev_id)
                send_dlq(producer, {"type": "final_failed", "stay_id": stay_id, "window_id": prev_id, "trigger_charttime": str(charttime), "error": str(exc)})
    try:
        sequence = build_sequence(preprocessor, stay_id, window_cache, window_id)
        inference_result, shap_result = run_scoring(stay_id, window_id, charttime, result, sequence, inference, shap_engine, "LIVE")
        window_start, window_end = window_bounds(intime, window_id)
        publish_result(producer, stay_id, window_id, charttime, "LIVE", result, inference_result, shap_result, window_start, window_end)
        last_window[stay_id] = window_id
    except Exception as exc:
        logger.exception("LIVE processing failed stay_id=%s window_id=%s", stay_id, window_id)
        send_dlq(producer, {"type": "live_failed", "stay_id": stay_id, "window_id": window_id, "charttime": str(charttime), "error": str(exc)})
    finally:
        prune_window_cache(stay_id, window_cache, keep)
def clear_preprocessor_stay(preprocessor, stay_id):
    for attr in ("active_windows", "locf_state"):
        state = getattr(preprocessor, attr, None)
        if isinstance(state, dict):
            state.pop(stay_id, None)
def evict_stay(stay_id, preprocessor, shap_engine, window_cache, last_window, last_seen, finalized, window_last_charttime):
    clear_preprocessor_stay(preprocessor, stay_id)
    window_cache.pop(stay_id, None)
    last_window.pop(stay_id, None)
    last_seen.pop(stay_id, None)
    window_last_charttime.pop(stay_id, None)
    finalized.difference_update({item for item in finalized if item[0] == stay_id})
    shap_engine.reset(stay_id)
def expire_pending(pending_observations, pending_last_seen, pending_total):
    now = time.monotonic()
    for stay_id, seen in list(pending_last_seen.items()):
        if now - seen >= PENDING_TTL_SECONDS:
            pending = pending_observations.pop(stay_id, None)
            if pending is not None:
                pending_total[0] -= len(pending)
            pending_last_seen.pop(stay_id, None)
    while pending_total[0] > PENDING_MAX_TOTAL:
        oldest = min(pending_last_seen, key=pending_last_seen.get, default=None)
        if oldest is None:
            break
        pending = pending_observations.pop(oldest, None)
        if pending is not None:
            pending_total[0] -= len(pending)
        pending_last_seen.pop(oldest, None)
def finalize_idle_stays(preprocessor, producer, inference, shap_engine, static_data, window_cache, last_window, last_seen, finalized, window_last_charttime):
    now = time.monotonic()
    for stay_id, state in list(last_seen.items()):
        seen, last_charttime = state
        window_id = last_window.get(stay_id)
        if window_id is None:
            continue
        key = (stay_id, int(window_id))
        age = now - seen
        if key in finalized:
            if age >= EVICT_AFTER_SECONDS:
                try:
                    evict_stay(stay_id, preprocessor, shap_engine, window_cache, last_window, last_seen, finalized, window_last_charttime)
                except Exception:
                    logger.exception("stay eviction failed stay_id=%s", stay_id)
            continue
        if age < IDLE_FINALIZE_SECONDS:
            continue
        result = window_cache.get(stay_id, {}).get(int(window_id))
        if result is None:
            continue
        timestamp = window_last_charttime.get(stay_id, {}).get(int(window_id), last_charttime)
        intime = static_data.get(stay_id, {}).get("intime")
        if intime is None:
            logger.warning("missing intime for idle FINAL stay_id=%s window_id=%s", stay_id, window_id)
            continue
        try:
            sequence = build_sequence(preprocessor, stay_id, window_cache, int(window_id))
            inference_result, shap_result = run_scoring(stay_id, int(window_id), timestamp, result, sequence, inference, shap_engine, "FINAL")
            window_start, window_end = window_bounds(intime, int(window_id))
            publish_result(producer, stay_id, int(window_id), timestamp, "FINAL", result, inference_result, shap_result, window_start, window_end)
            finalized.add(key)
            print(f"IDLE FINAL: stay_id={stay_id} window_id={window_id}")
        except Exception as exc:
            logger.exception("idle FINAL failed stay_id=%s window_id=%s", stay_id, window_id)
            send_dlq(producer, {"type": "idle_final_failed", "stay_id": int(stay_id), "window_id": int(window_id), "event_time": timestamp.isoformat() + "Z", "error": str(exc)})
        if age >= EVICT_AFTER_SECONDS:
            try:
                evict_stay(stay_id, preprocessor, shap_engine, window_cache, last_window, last_seen, finalized, window_last_charttime)
            except Exception:
                logger.exception("stay eviction failed stay_id=%s", stay_id)
def send_dlq(producer, record):
    if not ENABLE_DLQ:
        return
    try:
        payload = record if isinstance(record, dict) else {"raw_record": record.decode("utf-8", errors="replace") if isinstance(record, (bytes, bytearray)) else str(record)}
        producer.send(DLQ_TOPIC, key=payload.get("stay_id") if isinstance(payload, dict) else None, value=safe_json(payload)).get(timeout=10)
    except Exception:
        logger.exception("failed to publish DLQ record")
def check_shap_self_test(result):
    if isinstance(result, dict) and result.get("status") not in (None, "ok"):
        raise RuntimeError(f"SHAP startup self-test failed: {result}")
def main():
    global _shutdown
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)
    RealtimePreprocessor = load_module("realtime_preprocessing", "01_realtime_preprocessing.py").RealtimePreprocessor
    RealtimeInference = load_module("realtime_inference", "02_realtime_inference.py").RealtimeInference
    RealtimeSHAP = load_module("realtime_shap", "10_realtime_shap.py").RealtimeSHAP
    preprocessor = RealtimePreprocessor()
    inference = None
    producer = None
    static_consumer = None
    observation_consumer = None
    lifecycle_consumer = None
    try:
        preprocessor.start()
        inference = RealtimeInference()
        inference.start()
        shap_engine = RealtimeSHAP(inference.artifacts["xgb_model"], feature_names=inference.artifacts["feature_names"], top_k=10, max_gap_windows=6)
        shap_self_test = shap_engine.startup_self_test()
        check_shap_self_test(shap_self_test)
        print(f"SHAP STARTUP SELF-TEST: {shap_self_test}")
        producer = create_producer()
        static_consumer = create_static_consumer()
        observation_consumer = create_consumer(OBSERVATION_TOPIC, OBSERVATION_GROUP_ID, "latest")
        lifecycle_consumer = create_consumer(
            LIFECYCLE_TOPIC,
            "icu_realtime_lifecycle_v1",
            "earliest",
        )
        static_data = {}
        discharged_stays = set()
        pending_observations = defaultdict(deque)
        window_cache = defaultdict(dict)
        last_window = {}
        last_seen = {}
        finalized = set()
        window_last_charttime = {}
        pending_last_seen = {}
        pending_total = [0]
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
                            observation = buffered.popleft()
                            try:
                                process_observation(preprocessor, producer, inference, shap_engine, static_data, window_cache, last_window, last_seen, finalized, window_last_charttime, observation)
                            except Exception:
                                logger.exception("failed buffered record stay_id=%s", stay_id)
                                send_dlq(producer, {"type": "buffered_observation_failed", "stay_id": stay_id, "error": "buffered observation processing failed", "record": observation})
                    except Exception:
                        logger.exception("failed static record")
                        send_dlq(producer, {"type": "static_failed", "stay_id": data.get("stay_id") if isinstance(data, dict) else None, "error": "static record processing failed", "record": record.value.decode("utf-8", errors="replace")})
            if time.monotonic() - last_idle_sweep >= IDLE_SWEEP_SECONDS:
                expire_pending(pending_observations, pending_last_seen, pending_total)
                finalize_idle_stays(preprocessor, producer, inference, shap_engine, static_data, window_cache, last_window, last_seen, finalized, window_last_charttime)
                last_idle_sweep = time.monotonic()
            lifecycle_records = lifecycle_consumer.poll(timeout_ms=100)

            for _, records in lifecycle_records.items():
                for record in records:
                    data = None
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
                            continue

                        discharged_stays.add(stay_id)

                        # Prevent future static/observation processing.
                        static_data.pop(stay_id, None)
                        pending = pending_observations.pop(stay_id, deque())
                        pending_total[0] -= len(pending)
                        pending_last_seen.pop(stay_id, None)

                        # Remove realtime scoring state.
                        evict_stay(
                            stay_id,
                            preprocessor,
                            shap_engine,
                            window_cache,
                            last_window,
                            last_seen,
                            finalized,
                            window_last_charttime,
                        )

                        logger.info(
                            "DISCHARGED: stay_id=%s realtime monitoring stopped",
                            stay_id,
                        )
                        print(
                            f"DISCHARGED: stay_id={stay_id} "
                            f"realtime monitoring stopped"
                        )

                    except Exception:
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

            observation_records = observation_consumer.poll(timeout_ms=100)
            for _, records in observation_records.items():
                for record in records:
                    data = None
                    try:
                        data = json.loads(record.value.decode("utf-8"))
                        validate_observation(data)
                        stay_id = int(data["stay_id"])

                        if stay_id in discharged_stays:
                            logger.info(
                                "Ignoring observation for discharged stay_id=%s",
                                stay_id,
                            )
                            continue

                        if stay_id not in static_data:
                            pending = pending_observations[stay_id]
                            if len(pending) >= PENDING_MAX_PER_STAY:
                                pending.popleft()
                                pending_total[0] -= 1
                                logger.warning("pending observation cap reached stay_id=%s cap=%s", stay_id, PENDING_MAX_PER_STAY)
                            pending.append(data)
                            pending_total[0] += 1
                            pending_last_seen[stay_id] = time.monotonic()
                            print(f"OBSERVATION BUFFERED: stay_id={stay_id}")
                            continue
                        process_observation(preprocessor, producer, inference, shap_engine, static_data, window_cache, last_window, last_seen, finalized, window_last_charttime, data)
                    except Exception:
                        logger.exception("failed record stay_id=%s", None if data is None else data.get("stay_id"))
                        send_dlq(producer, {"type": "observation_failed", "stay_id": None if data is None else data.get("stay_id"), "error": "observation record processing failed", "record": data if data is not None else record.value.decode("utf-8", errors="replace")})
    finally:
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
        print("KAFKA REAL-TIME INFERENCE STOPPED")
if __name__ == "__main__":
    main()