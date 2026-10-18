import json
import logging
import threading
import time
from datetime import datetime
from kafka import KafkaConsumer
from kafka.structs import OffsetAndMetadata, TopicPartition
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from backend.app.core.config import settings
from backend.app.db.database import SessionLocal
from backend.app.models.patient_stay import PatientStay
from backend.app.models.risk_event import RiskEvent
from backend.app.models.risk_explanation import RiskExplanation
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("vitalis.kafka")
def parse_datetime(value):
    if not value:
        return None
    try:
        value = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        return dt
    except Exception:
        return None
def find_patient_id(db, stay_id):
    stay = db.scalar(
        select(PatientStay).where(
            PatientStay.stay_id == stay_id
        )
    )
    if not stay:
        return None
    return stay.patient_id
def save_risk_event(message):
    if not isinstance(message, dict):
        raise ValueError("Risk message must be a JSON object")

    stay_id = message.get("stay_id")
    if stay_id is None:
        raise ValueError("Risk message is missing stay_id")

    db = SessionLocal()
    try:
        patient_id = find_patient_id(db, int(stay_id))
        if patient_id is None:
            raise RuntimeError(
                f"No VITALIS patient mapping found for stay_id={stay_id}"
            )

        event = RiskEvent(
            patient_id=patient_id,
            stay_id=int(stay_id),
            window_id=int(message.get("window_id", 0)),
            status=message.get("status", "UNKNOWN"),
            event_time=parse_datetime(message.get("event_time")),
            window_start=parse_datetime(message.get("window_start")),
            window_end=parse_datetime(message.get("window_end")),
            sequence_length=message.get("sequence_length"),
            prediction=message.get("prediction"),
            threshold=message.get("threshold"),
            xgb_raw_probability=message.get("xgb_raw_probability"),
            xgb_probability=message.get("xgb_probability"),
            lstm_raw_probability=message.get("lstm_raw_probability"),
            lstm_probability=message.get("lstm_probability"),
            ensemble_probability=message.get("ensemble_probability"),
            model_version=message.get("model_version"),
            latency_ms=message.get("latency_ms"),
            shap_status=message.get("shap_status"),
            top_features=message.get("top_features"),
        )
        db.add(event)
        db.commit()

        logger.info(
            "Saved risk event | patient_id=%s | stay_id=%s | "
            "window=%s | status=%s",
            patient_id,
            stay_id,
            message.get("window_id"),
            message.get("status"),
        )
        return True
    except Exception:
        db.rollback()
        logger.exception("Failed to save risk event | stay_id=%s", stay_id)
        raise
    finally:
        db.close()


def run_consumer(stop_event=None):
    consumer = KafkaConsumer(
        settings.risk_topic,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        value_deserializer=lambda value: json.loads(value.decode("utf-8")),
        auto_offset_reset="latest",
        enable_auto_commit=False,
        group_id="vitalis-risk-consumer",
        consumer_timeout_ms=1000,
    )

    logger.info(
        "VITALIS Kafka consumer started | topic=%s | broker=%s",
        settings.risk_topic,
        settings.kafka_bootstrap_servers,
    )

    try:
        while stop_event is None or not stop_event.is_set():
            for record in consumer:
                if stop_event is not None and stop_event.is_set():
                    break

                logger.info(
                    "Received risk message | topic=%s | partition=%s | offset=%s",
                    record.topic,
                    record.partition,
                    record.offset,
                )

                for attempt in range(1, 4):
                    try:
                        save_risk_event(record.value)
                        consumer.commit({
                            TopicPartition(record.topic, record.partition):
                                OffsetAndMetadata(
                                    offset=record.offset + 1,
                                    metadata="",
                                    leader_epoch=getattr(record, "leader_epoch", -1),
                                )
                        })
                        break
                    except Exception:
                        logger.exception(
                            "Risk save/commit attempt %s/3 failed | "
                            "topic=%s partition=%s offset=%s",
                            attempt,
                            record.topic,
                            record.partition,
                            record.offset,
                        )
                        if attempt == 3:
                            logger.error(
                                "Stopping risk consumer with offset uncommitted | "
                                "topic=%s partition=%s offset=%s",
                                record.topic,
                                record.partition,
                                record.offset,
                            )
                            raise
                        time.sleep(attempt * 2)

    except KeyboardInterrupt:
        logger.info("VITALIS risk consumer stopped")
    finally:
        consumer.close()


def save_risk_explanation(message):
    # Invalid/non-ok explanations are intentionally ignored by the consumer.
    if not isinstance(message, dict) or message.get("status") != "ok":
        logger.warning("Ignoring invalid or non-ok SHAP explanation message")
        return True

    # Support both the nested SHAP schema and the live root-level schema.
    nested_shap = message.get("shap")
    if isinstance(nested_shap, dict):
        shap = nested_shap
    elif isinstance(message.get("shap_values"), list):
        shap = message
    else:
        logger.warning("Explanation message has no SHAP payload")
        return True

    values = shap.get("shap_values")
    top_features = shap.get("top_features")
    expected_value = shap.get("expected_value")
    model_margin = shap.get("model_margin")
    stay_id = message.get("stay_id")
    window_id = message.get("window_id")

    if stay_id is None or window_id is None:
        logger.warning("Explanation message missing stay_id/window_id")
        return True
    if not isinstance(values, list) or not values:
        logger.warning(
            "Explanation missing full SHAP vector | stay_id=%s window_id=%s",
            stay_id, window_id,
        )
        return True
    if expected_value is None or model_margin is None:
        logger.warning(
            "Explanation missing baseline/model margin | stay_id=%s window_id=%s",
            stay_id, window_id,
        )
        return True

    try:
        stay_id = int(stay_id)
        window_id = int(window_id)
        values = [float(value) for value in values]
        expected_value = float(expected_value)
        model_margin = float(model_margin)
        additivity_error = shap.get("additivity_error")
        additivity_error = (
            None if additivity_error is None else float(additivity_error)
        )
    except (TypeError, ValueError):
        logger.warning("Invalid numeric SHAP data for stay_id=%s", stay_id)
        return True

    db = SessionLocal()
    try:
        patient_id = find_patient_id(db, stay_id)
        if patient_id is None:
            raise RuntimeError(
                f"No VITALIS patient mapping for explanation stay_id={stay_id}"
            )

        row = {
            "patient_id": patient_id,
            "stay_id": stay_id,
            "window_id": window_id,
            "event_time": parse_datetime(
                message.get("event_time") or message.get("timestamp")
            ),
            "status": message.get("status"),
            "component": shap.get("component"),
            "output": shap.get("output"),
            "expected_value": expected_value,
            "model_margin": model_margin,
            "additivity_error": additivity_error,
            "shap_values": values,
            "top_features": top_features if isinstance(top_features, list) else [],
            "explanation_payload": message,
        }

        statement = pg_insert(RiskExplanation).values(**row)
        statement = statement.on_conflict_do_update(
            constraint="uq_risk_explanation_stay_window",
            set_={
                key: value
                for key, value in row.items()
                if key not in ("stay_id", "window_id")
            },
        )
        db.execute(statement)
        db.commit()
        logger.info(
            "Saved full SHAP explanation | patient_id=%s stay_id=%s window=%s features=%s",
            patient_id, stay_id, window_id, len(values),
        )
        return True
    except Exception:
        db.rollback()
        logger.exception(
            "Failed to save SHAP explanation | stay_id=%s window=%s",
            stay_id, window_id,
        )
        raise
    finally:
        db.close()


def run_explanation_consumer(stop_event=None):
    consumer = KafkaConsumer(
        settings.explanation_topic,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        value_deserializer=lambda value: json.loads(value.decode("utf-8")),
        auto_offset_reset="latest",
        enable_auto_commit=False,
        group_id="vitalis-explanation-consumer",
        consumer_timeout_ms=1000,
    )
    logger.info(
        "VITALIS explanation consumer started | topic=%s | broker=%s",
        settings.explanation_topic,
        settings.kafka_bootstrap_servers,
    )
    try:
        while stop_event is None or not stop_event.is_set():
            for record in consumer:
                if stop_event is not None and stop_event.is_set():
                    break
                logger.info(
                    "Received explanation | topic=%s | partition=%s | offset=%s",
                    record.topic,
                    record.partition,
                    record.offset,
                )
                # Retry this record before stopping; never commit a failed save.
                for attempt in range(1, 4):
                    try:
                        # Invalid messages are logged and acknowledged by the save function.
                        save_risk_explanation(record.value)
                        consumer.commit({
                            TopicPartition(record.topic, record.partition):
                                OffsetAndMetadata(
                                    offset=record.offset + 1,
                                    metadata="",
                                    leader_epoch=getattr(record, "leader_epoch", -1),
                                )
                        })
                        break
                    except Exception:
                        logger.exception(
                            "SHAP save/commit attempt %s/3 failed | topic=%s partition=%s offset=%s",
                            attempt, record.topic, record.partition, record.offset,
                        )
                        if attempt == 3:
                            logger.error(
                                "Stopping explanation consumer with offset uncommitted "
                                "| topic=%s partition=%s offset=%s",
                                record.topic, record.partition, record.offset,
                            )
                            raise
                        time.sleep(attempt * 2)
    except KeyboardInterrupt:
        logger.info("VITALIS explanation consumer stopped")
    finally:
        consumer.close()


if __name__ == "__main__":
    run_consumer()
