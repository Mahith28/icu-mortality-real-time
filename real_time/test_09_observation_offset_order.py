import importlib.util
from pathlib import Path

PATH = Path(__file__).with_name("09_kafka_realtime.py")
SPEC = importlib.util.spec_from_file_location("kafka_realtime_offset_test", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Consumer:
    def __init__(self):
        self.commits = []

    def commit(self, offsets):
        self.commits.append(offsets)


def test_safe_next_offset_respects_pending_records():
    pending = {
        1001: [( {"stay_id": 1001}, 0, 5)],
        1002: [( {"stay_id": 1002}, 0, 7)],
    }
    pending_total = [2]

    assert MODULE.safe_next_offset(pending, pending_total, 0, 8) == 5
    assert MODULE.safe_next_offset(pending, pending_total, 1, 8) == 9


def test_observation_commit_helper_does_not_commit_backwards():
    consumer = Consumer()
    committed = {}

    assert MODULE.commit_observation_offset(
        consumer, committed, "icu_observations", 0, 5
    ) is True

    MODULE.commit_observation_offset(
        consumer, committed, "icu_observations", 0, 4
    )

    assert committed[0] == 5
    assert len(consumer.commits) == 1

def test_later_record_must_not_commit_past_unfinished_record():
    # Offset 5 is still being processed; offset 6 completes first.
    # The next safe commit must remain at 5, not advance to 7.
    completed_offsets = {5: False, 6: True}
    next_offset = 5

    for offset in sorted(completed_offsets):
        if not completed_offsets[offset]:
            break
        next_offset = offset + 1

    assert next_offset == 5, (
        f"offset advanced past unfinished record: {next_offset}"
    )

class TrackerConsumer:
    def __init__(self, fail_commits=False):
        self.commits = []
        self.fail_commits = fail_commits

    def commit(self, offsets):
        if self.fail_commits:
            raise RuntimeError("intentional commit failure")
        self.commits.append(offsets)


def test_production_tracker_blocks_unfinished_offset():
    consumer = TrackerConsumer()
    tracker = {}
    topic = "icu_observations"
    partition = 0

    MODULE.mark_observation_offset_seen(
        tracker, consumer, topic, partition, 10
    )
    MODULE.mark_observation_offset_seen(
        tracker, consumer, topic, partition, 11
    )

    MODULE.mark_observation_offset_done(tracker, topic, partition, 11)
    MODULE.commit_ready_observation_offsets(consumer, tracker)

    # Offset 10 is unfinished: do not commit offset 12.
    assert consumer.commits == []
    assert tracker[(topic, partition)]["next"] == 10

    MODULE.mark_observation_offset_done(tracker, topic, partition, 10)
    MODULE.commit_ready_observation_offsets(consumer, tracker)

    # Both records are now done: next offset can advance through 11.
    assert len(consumer.commits) == 1
    assert tracker[(topic, partition)]["next"] == 12


def test_production_tracker_retains_state_after_commit_failure():
    consumer = TrackerConsumer(fail_commits=True)
    tracker = {}
    topic = "icu_observations"
    partition = 0

    MODULE.mark_observation_offset_seen(
        tracker, consumer, topic, partition, 20
    )
    MODULE.mark_observation_offset_done(tracker, topic, partition, 20)

    MODULE.commit_ready_observation_offsets(consumer, tracker)

    # Failed commit must not advance the tracker's next offset.
    assert tracker[(topic, partition)]["next"] == 20

    consumer.fail_commits = False
    MODULE.commit_ready_observation_offsets(consumer, tracker)

    assert len(consumer.commits) == 1
    assert tracker[(topic, partition)]["next"] == 21

if __name__ == "__main__":
    test_safe_next_offset_respects_pending_records()
    test_observation_commit_helper_does_not_commit_backwards()
    test_later_record_must_not_commit_past_unfinished_record()
    test_production_tracker_blocks_unfinished_offset()
    test_production_tracker_retains_state_after_commit_failure()
    print("OBSERVATION OFFSET BASELINE TESTS PASSED")
