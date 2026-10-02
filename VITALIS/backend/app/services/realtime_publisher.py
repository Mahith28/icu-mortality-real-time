import json
import logging
from datetime import datetime, timezone
from kafka import KafkaProducer
from backend.app.core.config import settings
from backend.app.models.patient import Patient
from backend.app.models.patient_stay import PatientStay
logger = logging.getLogger("vitalis.realtime")
class RealtimePublisher:
    def __init__(self):
        self.producer = KafkaProducer(
            bootstrap_servers=settings.kafka_bootstrap_servers,
            value_serializer=lambda value: json.dumps(
                value,
                separators=(",", ":"),
            ).encode("utf-8"),
            retries=5,
            acks="all",
        )
    def publish_static(
        self,
        patient: Patient,
        stay: PatientStay,
    ):
        if not patient.race:
            raise ValueError(
                "Patient race is required for realtime registration."
            )
        if not patient.insurance:
            raise ValueError(
                "Patient insurance is required for realtime registration."
            )
        if not patient.admission_location:
            raise ValueError(
                "Patient admission_location is required for realtime registration."
            )
        if not patient.age:
            raise ValueError(
                "Patient age is required for realtime registration."
            )
        if not patient.gender:
            raise ValueError(
                "Patient gender is required for realtime registration."
            )
        if not patient.admission_type:
            raise ValueError(
                "Patient admission_type is required for realtime registration."
            )
        if not patient.icu_unit:
            raise ValueError(
                "Patient ICU unit is required for realtime registration."
            )
        intime = stay.intime or datetime.now(timezone.utc)
        if intime.tzinfo is None:
            intime = intime.replace(tzinfo=timezone.utc)
        message = {
            "stay_id": int(stay.stay_id),
            "anchor_age": int(patient.age),
            "gender": patient.gender,
            "race": patient.race,
            "insurance": patient.insurance,
            "admission_type": patient.admission_type,
            "admission_location": patient.admission_location,
            "first_careunit": patient.icu_unit,
            "intime": intime.isoformat().replace(
                "+00:00",
                "Z",
            ),
        }
        future = self.producer.send(
            settings.realtime_static_topic,
            value=message,
            key=str(stay.stay_id).encode("utf-8"),
        )
        future.get(timeout=10)
        logger.info(
            "Realtime static registration published: stay_id=%s",
            stay.stay_id,
        )
        return message
    def close(self):
        self.producer.flush()
        self.producer.close()
publisher = RealtimePublisher()
