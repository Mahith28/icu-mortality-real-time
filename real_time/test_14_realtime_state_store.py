"""Standalone tests for 14_realtime_state_store.py; no Kafka required."""
import importlib.util
import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("14_realtime_state_store.py")
SPEC = importlib.util.spec_from_file_location("realtime_state_store", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_snapshot_and_offsets_survive_reopen():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "state.db"
        payload = {
            "schema_version": 1,
            "active_windows": {"1001": {"12": [{"value": 82.0}]}},
            "locf_state": {"1001": [82.0, 70.0]},
            "pending_finals": {"1001:12": {"risk_sent": False}},
        }
        with MODULE.RealtimeStateStore(db) as store:
            store.save_snapshot(payload, {("icu_observations", 0): 13})

        with MODULE.RealtimeStateStore(db) as reopened:
            loaded, offsets, updated_at = reopened.load()
            assert loaded == payload
            assert offsets == {("icu_observations", 0): 13}
            assert isinstance(updated_at, float)


def test_invalid_payload_rejected_before_write():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "state.db"
        with MODULE.RealtimeStateStore(db) as store:
            store.save_snapshot({"revision": 1}, {("icu_observations", 0): 5})
            try:
                store.save_snapshot({"bad": float("nan")}, {("icu_observations", 0): 99})
            except MODULE.StateStoreError:
                pass
            else:
                raise AssertionError("non-finite JSON payload should be rejected")
            assert store.load()[:2] == ({"revision": 1}, {("icu_observations", 0): 5})


def test_missing_snapshot_is_none():
    with tempfile.TemporaryDirectory() as tmp:
        with MODULE.RealtimeStateStore(Path(tmp) / "state.db") as store:
            assert store.load() == (None, {}, None)


def test_tilde_path_is_expanded():
    with tempfile.TemporaryDirectory() as tmp:
        old_home = os.environ.get("HOME")
        old_userprofile = os.environ.get("USERPROFILE")
        os.environ["HOME"] = tmp
        os.environ["USERPROFILE"] = tmp
        try:
            with MODULE.RealtimeStateStore("~/state/rt.db") as store:
                assert store.path == str((Path(tmp) / "state" / "rt.db").resolve())
                store.save_snapshot({"ok": True})
                assert store.load_snapshot() == {"ok": True}
        finally:
            if old_home is None:
                os.environ.pop("HOME", None)
            else:
                os.environ["HOME"] = old_home
            if old_userprofile is None:
                os.environ.pop("USERPROFILE", None)
            else:
                os.environ["USERPROFILE"] = old_userprofile


def test_invalid_offsets_raise_state_store_error():
    with tempfile.TemporaryDirectory() as tmp:
        with MODULE.RealtimeStateStore(Path(tmp) / "state.db") as store:
            for offsets in [
                {("icu_observations", 1.9): 4},
                {("icu_observations", 0): True},
                {("icu_observations", -1): 4},
                {("icu_observations", 0): 1 << 80},
                {("icu_observations", "bad"): 4},
            ]:
                try:
                    store.save_snapshot({"ok": True}, offsets)
                except MODULE.StateStoreError:
                    pass
                else:
                    raise AssertionError(f"invalid offsets accepted: {offsets!r}")


def test_use_after_close_raises_state_store_error():
    with tempfile.TemporaryDirectory() as tmp:
        store = MODULE.RealtimeStateStore(Path(tmp) / "state.db")
        store.close()
        for action in (store.load, lambda: store.save_snapshot({"x": 1})):
            try:
                action()
            except MODULE.StateStoreError as exc:
                assert "closed" in str(exc)
            else:
                raise AssertionError("use after close should raise StateStoreError")


def test_memory_database_supported():
    with MODULE.RealtimeStateStore(":memory:") as store:
        store.save_snapshot({"memory": True})
        assert store.load_snapshot() == {"memory": True}


def test_path_preparation_failure_is_wrapped():
    with tempfile.TemporaryDirectory() as tmp:
        parent_file = Path(tmp) / "not_a_directory"
        parent_file.write_text("x", encoding="utf-8")
        try:
            MODULE.RealtimeStateStore(parent_file / "state.db")
        except MODULE.StateStoreError as exc:
            assert "initialize" in str(exc).lower()
        else:
            raise AssertionError("invalid parent path should raise StateStoreError")


def test_offset_regression_is_rejected_atomically():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "state.db"
        with MODULE.RealtimeStateStore(db) as store:
            store.save_snapshot({"revision": 1}, {("icu_observations", 0): 20})
            try:
                store.save_snapshot({"revision": 2}, {("icu_observations", 0): 19})
            except MODULE.StateStoreError as exc:
                assert "regression" in str(exc).lower()
            else:
                raise AssertionError("lower Kafka offset should be rejected")
            loaded, offsets, _ = store.load()
            assert loaded == {"revision": 1}
            assert offsets == {("icu_observations", 0): 20}


def test_newer_database_schema_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "state.db"
        import sqlite3
        conn = sqlite3.connect(db)
        conn.execute("PRAGMA user_version=999")
        conn.close()
        try:
            MODULE.RealtimeStateStore(db)
        except MODULE.StateStoreError as exc:
            assert "schema version" in str(exc).lower()
        else:
            raise AssertionError("newer database schema should be rejected")


def test_corrupt_offset_row_raises_state_store_error():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "state.db"
        with MODULE.RealtimeStateStore(db) as store:
            store.save_snapshot({"ok": True}, {("icu_observations", 0): 10})
        import sqlite3
        conn = sqlite3.connect(db)
        conn.execute("UPDATE kafka_offsets SET partition_id='bad'")
        conn.commit()
        conn.close()
        with MODULE.RealtimeStateStore(db) as store:
            try:
                store.load()
            except MODULE.StateStoreError as exc:
                assert "offset" in str(exc).lower()
            else:
                raise AssertionError("corrupt offset row should raise StateStoreError")



def _tamper_db(db: Path, sql: str, params=()):
    conn = sqlite3.connect(db)
    try:
        conn.execute(sql, params)
        conn.commit()
    finally:
        conn.close()


def test_multi_partition_failure_rolls_back_all():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "state.db"
        with MODULE.RealtimeStateStore(db) as store:
            store.save_snapshot({"revision": 1}, {("t", 0): 10, ("t", 1): 10})
            try:
                store.save_snapshot({"revision": 2}, {("t", 0): 20, ("t", 1): 5})
            except MODULE.StateStoreError as exc:
                assert "regression" in str(exc).lower()
            else:
                raise AssertionError("expected second partition regression")
            snap, offsets, _ = store.load()
            assert snap == {"revision": 1}
            assert offsets == {("t", 0): 10, ("t", 1): 10}


def test_legacy_unversioned_database_is_migrated():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "legacy.db"
        conn = sqlite3.connect(db)
        conn.executescript("""
            CREATE TABLE runtime_snapshot (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                schema_version INTEGER NOT NULL,
                updated_at REAL NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE kafka_offsets (
                topic TEXT NOT NULL,
                partition_id INTEGER NOT NULL,
                next_offset INTEGER NOT NULL,
                PRIMARY KEY (topic, partition_id)
            );
            INSERT INTO runtime_snapshot VALUES (1, 1, 123.0, '{"legacy":true}');
            INSERT INTO kafka_offsets VALUES ('t', 0, 8);
            PRAGMA user_version=0;
        """)
        conn.close()
        with MODULE.RealtimeStateStore(db) as store:
            snap, offsets, updated = store.load()
            assert snap == {"legacy": True}
            assert offsets == {("t", 0): 8}
            assert updated == 123.0
        conn = sqlite3.connect(db)
        try:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == MODULE.STATE_SCHEMA_VERSION
        finally:
            conn.close()


def test_duplicate_json_keys_rejected_on_load_and_mixed_keys_on_save():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "state.db"
        with MODULE.RealtimeStateStore(db) as store:
            try:
                store.save_snapshot({1: "a", "1": "b"})
            except MODULE.StateStoreError:
                pass
            else:
                raise AssertionError("mixed keys that collide in JSON must be rejected")
            store.save_snapshot({"ok": True})
        _tamper_db(db, "UPDATE runtime_snapshot SET payload_json=?", ('{"1":"a","1":"b"}',))
        with MODULE.RealtimeStateStore(db) as store:
            try:
                store.load()
            except MODULE.StateStoreError as exc:
                assert "invalid json" in str(exc).lower()
            else:
                raise AssertionError("duplicate JSON keys must be rejected on load")


def test_allow_rewind_requires_explicit_opt_in():
    with tempfile.TemporaryDirectory() as tmp:
        with MODULE.RealtimeStateStore(Path(tmp) / "state.db") as store:
            store.save_snapshot({"revision": 1}, {("t", 0): 20})
            try:
                store.save_snapshot({"revision": 2}, {("t", 0): 10})
            except MODULE.StateStoreError:
                pass
            else:
                raise AssertionError("rewind must be rejected by default")
            snap, offsets, _ = store.load()
            assert snap == {"revision": 1}
            assert offsets == {("t", 0): 20}

            store.save_snapshot(
                {"revision": 3}, {("t", 0): 10}, allow_rewind=True
            )
            snap, offsets, _ = store.load()
            assert snap == {"revision": 3}
            assert offsets == {("t", 0): 10}


def test_equal_offsets_and_no_offset_save_are_allowed():
    with tempfile.TemporaryDirectory() as tmp:
        with MODULE.RealtimeStateStore(Path(tmp) / "state.db") as store:
            store.save_snapshot({"revision": 1}, {("t", 0): 10})
            store.save_snapshot({"revision": 2}, {("t", 0): 10})
            store.save_snapshot({"revision": 3}, None)
            snap, offsets, _ = store.load()
            assert snap == {"revision": 3}
            assert offsets == {("t", 0): 10}


def test_crash_inside_save_snapshot_before_commit_preserves_old_state():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "state.db"
        with MODULE.RealtimeStateStore(db) as store:
            store.save_snapshot({"revision": 1}, {("t", 0): 10})

        module_path = Path(MODULE.__file__).resolve()
        child = r'''import importlib.util, os, sys
module_path, db_path = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("state_store_child", module_path)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
s = M.RealtimeStateStore(db_path)
def crash_before_offset_insert(sql):
    if "INSERT INTO kafka_offsets" in sql:
        os._exit(17)
s._connection().set_trace_callback(crash_before_offset_insert)
s.save_snapshot({"revision": 2}, {("t", 0): 20})
os._exit(19)  # Reaching here means the injection did not fire.
'''
        result = subprocess.run(
            [sys.executable, "-c", child, str(module_path), str(db)], check=False
        )
        assert result.returncode == 17
        with MODULE.RealtimeStateStore(db) as store:
            snap, offsets, _ = store.load()
            assert snap == {"revision": 1}
            assert offsets == {("t", 0): 10}


def test_crash_after_commit_preserves_new_snapshot_and_offsets():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "state.db"
        with MODULE.RealtimeStateStore(db) as store:
            store.save_snapshot({"revision": 1}, {("t", 0): 10})
        child = r'''import os, sqlite3, sys
c = sqlite3.connect(sys.argv[1])
with c:
    c.execute("UPDATE runtime_snapshot SET payload_json=?, updated_at=? WHERE singleton=1", ('{"revision":2}', 2.0))
    c.execute("UPDATE kafka_offsets SET next_offset=20 WHERE topic='t' AND partition_id=0")
c.close()
os._exit(18)
'''
        result = subprocess.run([sys.executable, "-c", child, str(db)], check=False)
        assert result.returncode == 18
        with MODULE.RealtimeStateStore(db) as store:
            snap, offsets, _ = store.load()
            assert snap == {"revision": 2}
            assert offsets == {("t", 0): 20}


if __name__ == "__main__":
    tests = [value for name, value in globals().items() if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
    print(f"REALTIME STATE STORE TESTS PASSED ({len(tests)} tests)")
