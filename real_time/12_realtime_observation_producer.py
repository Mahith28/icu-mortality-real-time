import sys
import os
import time
import json
import signal
import random
from datetime import datetime, timedelta, timezone
from kafka import KafkaProducer
from sqlalchemy import select
PROJECT_ROOT = "/home/mahith/BDA_PROJECT"
VITALIS_ROOT = os.path.join(PROJECT_ROOT, "VITALIS")
sys.path.insert(0, VITALIS_ROOT)
from backend.app.core.config import settings
from backend.app.db.database import SessionLocal
from backend.app.models.patient_stay import PatientStay
KAFKA_BOOTSTRAP = settings.kafka_bootstrap_servers
OBSERVATION_TOPIC = "icu_observations"
REAL_SECONDS_PER_MINUTE = 1.0
VITALS = [
    ("heart_rate", 82.0, 4.0, 45.0, 160.0),
    ("sbp", 118.0, 7.0, 70.0, 190.0),
    ("dbp", 72.0, 5.0, 40.0, 120.0),
    ("resp_rate", 18.0, 2.0, 8.0, 45.0),
    ("spo2", 97.0, 1.0, 80.0, 100.0),
    ("temperature", 37.0, 0.2, 34.0, 41.0),
]
_shutdown = False
def shutdown(signum, frame):
    global _shutdown
    _shutdown = True
    print("\nStopping observation producer")

def parse_db_datetime(value):
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
def get_active_stay():
    db = SessionLocal()
    try:
        stmt = (
            select(PatientStay)
            .where(PatientStay.outtime.is_(None))
            .order_by(PatientStay.created_at.desc())
            .limit(1)
        )
        stay = db.execute(stmt).scalars().first()
        if stay is None:
            return None
        return {
            "stay_id": int(stay.stay_id),
            "intime": parse_db_datetime(stay.intime),
        }
    finally:
        db.close()
def create_producer():
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        acks="all",
        retries=5,
        linger_ms=5,
        value_serializer=lambda value: json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8"),
        key_serializer=lambda key: str(key).encode("utf-8"),
    )
def iso_timestamp(value):
    return (
        value.astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )
def generate_value(base, variation, minimum, maximum):
    value = random.gauss(base, variation)
    return max(minimum, min(maximum, value))
def publish_observation(
    producer,
    stay_id,
    intime,
    charttime,
    vital_sign,
    value,
):
    message = {
        "stay_id": int(stay_id),
        "charttime": iso_timestamp(charttime),
        "intime": iso_timestamp(intime),
        "vital_sign": vital_sign,
        "value": round(float(value), 3),
    }
    future = producer.send(
        OBSERVATION_TOPIC,
        key=str(stay_id),
        value=message,
    )
    future.get(timeout=10)
    print(
        f"[OBSERVATION] "
        f"stay_id={stay_id} "
        f"time={iso_timestamp(charttime)} "
        f"{vital_sign}={message['value']}"
    )
def main():
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    print("VITALIS REAL-TIME ICU OBSERVATION PRODUCER")
    print(f"Kafka: {KAFKA_BOOTSTRAP}")
    print(f"Topic: {OBSERVATION_TOPIC}")
    print()
    producer = create_producer()
    active_stay = None
    try:
        while not _shutdown:
            current_stay = get_active_stay()
            if current_stay is None:
                print(
                    "[WAITING] No active VITALIS stay found. "
                    "Create/admit a patient first."
                )
                time.sleep(3)
                continue
            if (
                active_stay is None
                or current_stay["stay_id"] != active_stay["stay_id"]
            ):
                active_stay = current_stay
                print()
                print(
                    f"[ACTIVE STAY] "
                    f"stay_id={active_stay['stay_id']}"
                )
                print(
                    f"[ACTIVE STAY] "
                    f"intime={iso_timestamp(active_stay['intime'])}"
                )
                print()
            stay_id = active_stay["stay_id"]
            intime = active_stay["intime"]
            charttime = intime
            print(
                "[STREAM] Starting simulated ICU observations"
            )
            while not _shutdown:
                refreshed_stay = get_active_stay()

                if (
                    refreshed_stay is None
                    or refreshed_stay["stay_id"] != stay_id
                ):
                    print(
                        f"[STAY CHANGE] stay_id={stay_id} "
                        "is no longer the active stay."
                    )
                    active_stay = None
                    break
                for (
                    vital_sign,
                    base,
                    variation,
                    minimum,
                    maximum,
                ) in VITALS:
                    value = generate_value(
                        base,
                        variation,
                        minimum,
                        maximum,
                    )
                    publish_observation(
                        producer=producer,
                        stay_id=stay_id,
                        intime=intime,
                        charttime=charttime,
                        vital_sign=vital_sign,
                        value=value,
                    )
                charttime += timedelta(minutes=1)
                time.sleep(REAL_SECONDS_PER_MINUTE)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nFlushing Kafka producer")
        producer.flush(timeout=10)
        producer.close()
        print("Observation producer stopped.")
if __name__ == "__main__":
    main()