import json
import logging
import threading
from datetime import datetime
from kafka import KafkaConsumer
from sqlalchemy import select
from backend.app.core.config import settings
from backend.app.db.database import SessionLocal
from backend.app.models.patient_stay import PatientStay
from backend.app.models.risk_event import RiskEvent
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
    stay_id = message.get("stay_id")
    if stay_id is None:
        logger.warning("Received risk message without stay_id")
        return
    db = SessionLocal()
    try:
        patient_id = find_patient_id(db, int(stay_id))
        if patient_id is None:
            logger.warning(
                "No VITALIS patient mapping found for stay_id=%s",
                stay_id,
            )
            return
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
            "window=%s | status=%s | ensemble=%.6f",
            patient_id,
            stay_id,
            message.get("window_id"),
            message.get("status"),
            float(message.get("ensemble_probability", 0)),
        )
    except Exception:
        db.rollback()
        logger.exception("Failed to save risk event")
    finally:
        db.close()
def run_consumer():
    consumer = KafkaConsumer(
        settings.risk_topic,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        value_deserializer=lambda value: json.loads(
            value.decode("utf-8")
        ),
        auto_offset_reset="latest",
        enable_auto_commit=True,
        group_id="vitalis-risk-consumer",
    )
    logger.info(
        "VITALIS Kafka consumer started | topic=%s | broker=%s",
        settings.risk_topic,
        settings.kafka_bootstrap_servers,
    )
    try:
        for record in consumer:
            logger.info(
                "Received Kafka message | topic=%s | partition=%s | offset=%s",
                record.topic,
                record.partition,
                record.offset,
            )
            save_risk_event(record.value)
    except KeyboardInterrupt:
        logger.info("Kafka consumer stopped")
    finally:
        consumer.close()
if __name__ == "__main__":
    run_consumer()
