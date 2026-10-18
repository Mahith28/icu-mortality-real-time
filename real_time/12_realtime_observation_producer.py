import sys
import os
import time
import json
import signal
import random
import uuid
import threading
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
TIME_MODE = os.getenv("VITALIS_TIME_MODE", "realtime").lower()
OBSERVATION_INTERVAL_SECONDS = float(
    os.getenv("VITALIS_OBSERVATION_INTERVAL_SECONDS", "30")
)
if TIME_MODE not in {"realtime", "simulation"}:
    raise RuntimeError(
        "VITALIS_TIME_MODE must be 'realtime' or 'simulation'"
    )
VITALS = [
    ("heart_rate", 82.0, 4.0, 45.0, 160.0),
    ("sbp", 118.0, 7.0, 70.0, 190.0),
    ("dbp", 72.0, 5.0, 40.0, 120.0),
    ("resp_rate", 18.0, 2.0, 8.0, 45.0),
    ("spo2", 97.0, 1.0, 80.0, 100.0),
    ("temperature", 37.0, 0.2, 34.0, 41.0),
]
OBSERVATION_SCHEMA_VERSION = "1.0"
_shutdown = False
_shutdown_event = threading.Event()
def shutdown(signum, frame):
    global _shutdown
    _shutdown = True
    _shutdown_event.set()
    print("\nStopping observation producer")
def parse_db_datetime(value):
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
def get_active_stays():
    db = SessionLocal()
    try:
        stmt = (
            select(PatientStay)
            .where(PatientStay.outtime.is_(None))
            .order_by(PatientStay.created_at.asc())
        )
        stays = db.execute(stmt).scalars().all()
        return [
            {
                "stay_id": int(stay.stay_id),
                "intime": parse_db_datetime(stay.intime),
            }
            for stay in stays
        ]
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
class ObservationClock:
    def __init__(self, mode, observation_interval_seconds):
        self.mode = mode
        self.observation_interval = timedelta(
            seconds=observation_interval_seconds
        )
        self.simulated_charttime = None
    def start(self, intime):
        if self.mode == "simulation":
            self.simulated_charttime = intime
    def now(self):
        if self.mode == "realtime":
            return datetime.now(timezone.utc)
        if self.simulated_charttime is None:
            raise RuntimeError("Simulation clock has not been started")
        return self.simulated_charttime
    def advance(self):
        if self.mode == "simulation":
            self.simulated_charttime += self.observation_interval
def publish_observation(
    producer,
    stay_id,
    intime,
    charttime,
    vital_sign,
    value,
):
    ingestion_time = datetime.now(timezone.utc)
    message = {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "event_id": str(uuid.uuid4()),
        "stay_id": int(stay_id),
        "charttime": iso_timestamp(charttime),
        "intime": iso_timestamp(intime),
        "ingestion_time": iso_timestamp(ingestion_time),
        "source_mode": TIME_MODE,
        "source_type": (
            "simulation_synthetic"
            if TIME_MODE == "simulation"
            else "realtime_synthetic_demo"
        ),
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
    print("VITALIS ICU OBSERVATION PRODUCER")
    print(f"Time mode: {TIME_MODE}")
    print(f"Observation interval: {OBSERVATION_INTERVAL_SECONDS} seconds")
    print(f"Kafka: {KAFKA_BOOTSTRAP}")
    print(f"Topic: {OBSERVATION_TOPIC}")
    print()
    producer = create_producer()
    patient_states = {}
    try:
        while not _shutdown:
            active_stays = get_active_stays()
            active_stay_ids = {
                stay["stay_id"] for stay in active_stays
            }
            if not active_stays:
                print(
                    "[WAITING] No active VITALIS stays found. "
                    "Create/admit a patient first."
                )
                _shutdown_event.wait(3)
                continue
            for stay in active_stays:
                stay_id = stay["stay_id"]
                intime = stay["intime"]
                if stay_id not in patient_states:
                    clock = ObservationClock(
                        mode=TIME_MODE,
                        observation_interval_seconds=(
                            OBSERVATION_INTERVAL_SECONDS
                        ),
                    )
                    clock.start(intime)
                    patient_states[stay_id] = {
                        "stay_id": stay_id,
                        "intime": intime,
                        "clock": clock,
                        "next_tick": time.monotonic(),
                    }
                    print()
                    print(
                        f"[ACTIVE STAY] "
                        f"stay_id={stay_id}"
                    )
                    print(
                        f"[ACTIVE STAY] "
                        f"intime={iso_timestamp(intime)}"
                    )
                    print(
                        f"[STREAM] Starting {TIME_MODE} ICU observations "
                        f"for stay_id={stay_id}"
                    )
                    print()
            removed_stays = set(patient_states) - active_stay_ids
            for stay_id in removed_stays:
                print(
                    f"[STAY CHANGE] "
                    f"stay_id={stay_id} is no longer active."
                )
                del patient_states[stay_id]
            now = time.monotonic()
            for stay_id in sorted(patient_states):
                state = patient_states[stay_id]
                if now < state["next_tick"]:
                    continue
                clock = state["clock"]
                intime = state["intime"]
                charttime = clock.now()
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
                if TIME_MODE == "simulation":
                    clock.advance()
                state["next_tick"] += OBSERVATION_INTERVAL_SECONDS
            next_ticks = [
                state["next_tick"]
                for state in patient_states.values()
            ]
            if next_ticks:
                wait_time = max(
                    0.0,
                    min(next_ticks) - time.monotonic(),
                )
                _shutdown_event.wait(wait_time)
            else:
                _shutdown_event.wait(1)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nFlushing Kafka producer")
        producer.flush()
        producer.close()
        print("Observation producer stopped.")
if __name__ == "__main__":
    main()