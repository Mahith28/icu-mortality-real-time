import json
from datetime import datetime, timezone

from kafka import KafkaProducer

from backend.app.core.config import settings


class RealtimeKafkaProducer:
    def __init__(self):
        self.producer = KafkaProducer(
            bootstrap_servers=settings.kafka_bootstrap_servers,
            value_serializer=lambda value: json.dumps(value).encode("utf-8"),
            acks="all",
            retries=5,
            request_timeout_ms=10000,
            delivery_timeout_ms=30000,
        )

    def publish_static_patient(
        self,
        *,
        stay_id: int,
        anchor_age: int,
        gender: str,
        race: str | None,
        insurance: str | None,
        admission_type: str,
        admission_location: str | None,
        first_careunit: str,
        intime: datetime,
    ) -> None:
        if intime.tzinfo is None:
            intime = intime.replace(tzinfo=timezone.utc)

        message = {
            "stay_id": int(stay_id),
            "anchor_age": int(anchor_age),
            "gender": gender,
            "race": race or "",
            "insurance": insurance or "",
            "admission_type": admission_type,
            "admission_location": admission_location or "",
            "first_careunit": first_careunit,
            "intime": intime.astimezone(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
        }

        future = self.producer.send(
            settings.realtime_static_topic,
            value=message,
        )

        future.get(timeout=10)

    def publish_discharge(
        self,
        *,
        stay_id: int,
        event_time: datetime,
    ) -> None:
        if event_time.tzinfo is None:
            event_time = event_time.replace(tzinfo=timezone.utc)

        message = {
            "event_type": "DISCHARGE",
            "stay_id": int(stay_id),
            "event_time": (
                event_time.astimezone(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            ),
        }

        future = self.producer.send(
            settings.realtime_lifecycle_topic,
            value=message,
        )

        future.get(timeout=10)

    def close(self) -> None:
        self.producer.flush()
        self.producer.close()


realtime_kafka = RealtimeKafkaProducer()
