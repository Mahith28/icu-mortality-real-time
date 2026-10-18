"""SQLite-backed snapshots for the realtime inference service.

This module deliberately does not import inference code. It provides a small,
versioned persistence primitive. Recovery callers should use ``load()`` to
read the runtime snapshot and Kafka next offsets from one consistent SQLite
read transaction. Kafka offsets must only be committed after the matching
snapshot and offsets have been durably saved here.

Snapshots use JSON semantics: tuples become lists and dictionary keys become
strings. Callers must normalize runtime-specific objects before saving.
The store is intended for single-process, single-thread ownership; do not share
one instance across threads or run multiple writers against the same database.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Mapping, NamedTuple

STATE_SCHEMA_VERSION = 1
_SQLITE_INT_MAX = (1 << 63) - 1


class StateStoreError(RuntimeError):
    """Raised when persisted state is invalid or cannot be safely used."""


class LoadedState(NamedTuple):
    snapshot: dict[str, Any] | None
    offsets: dict[tuple[str, int], int]
    updated_at: float | None


class RealtimeStateStore:
    """Store versioned JSON snapshots and offsets in one SQLite database."""

    def __init__(self, path: str | os.PathLike[str]):
        self._conn: sqlite3.Connection | None = None
        try:
            raw_path = os.fspath(path)
            if raw_path == ":memory:":
                self.path = raw_path
            else:
                resolved = Path(raw_path).expanduser().resolve()
                resolved.parent.mkdir(parents=True, exist_ok=True)
                self.path = str(resolved)

            self._conn = sqlite3.connect(self.path, timeout=10.0)
            journal_row = self._conn.execute("PRAGMA journal_mode=WAL").fetchone()
            journal_mode = str(journal_row[0]).lower() if journal_row else ""
            if self.path != ":memory:" and journal_mode != "wal":
                raise StateStoreError(
                    f"SQLite WAL mode unavailable (active mode: {journal_mode or 'unknown'})"
                )
            self._conn.execute("PRAGMA synchronous=FULL")
            self._conn.execute("PRAGMA foreign_keys=ON")

            user_version = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
            if user_version not in (0, STATE_SCHEMA_VERSION):
                raise StateStoreError(
                    f"Unsupported database schema version {user_version}; "
                    f"expected {STATE_SCHEMA_VERSION}"
                )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runtime_snapshot (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    schema_version INTEGER NOT NULL,
                    updated_at REAL NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS kafka_offsets (
                    topic TEXT NOT NULL,
                    partition_id INTEGER NOT NULL,
                    next_offset INTEGER NOT NULL,
                    PRIMARY KEY (topic, partition_id)
                )
                """
            )
            if user_version == 0:
                # Migrate the previous unversioned schema only when any stored
                # snapshot already declares the schema version we understand.
                snapshot_row = self._conn.execute(
                    "SELECT schema_version FROM runtime_snapshot WHERE singleton=1"
                ).fetchone()
                if snapshot_row is not None and snapshot_row[0] != STATE_SCHEMA_VERSION:
                    raise StateStoreError(
                        "Existing unversioned state database requires an explicit migration"
                    )
                self._conn.execute(f"PRAGMA user_version={STATE_SCHEMA_VERSION}")
            self._conn.commit()
        except StateStoreError:
            self._close_after_init_failure()
            raise
        except (sqlite3.Error, OSError, TypeError, ValueError) as exc:
            self._close_after_init_failure()
            raise StateStoreError("Failed to initialize runtime state database") from exc

    def _close_after_init_failure(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass
            self._conn = None

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise StateStoreError("store is closed")
        return self._conn

    @staticmethod
    def _encode(payload: Mapping[str, Any]) -> str:
        if not isinstance(payload, Mapping):
            raise StateStoreError("Snapshot payload must be a mapping")
        try:
            # Materialize generic mappings into a JSON-serializable plain dict.
            # Keep sort_keys=True: mixed key types such as 1 and "1" must
            # fail during save rather than create ambiguous JSON object keys.
            return json.dumps(
                dict(payload), allow_nan=False, separators=(",", ":"), sort_keys=True
            )
        except (TypeError, ValueError, RecursionError) as exc:
            raise StateStoreError(
                "Snapshot must contain JSON-safe values; normalize runtime "
                "objects explicitly before saving."
            ) from exc

    @staticmethod
    def _validate_offsets(
        next_offsets: Mapping[tuple[str, int], int] | None,
    ) -> dict[tuple[str, int], int]:
        if next_offsets is None:
            return {}
        if not isinstance(next_offsets, Mapping):
            raise StateStoreError("Kafka offsets must be a mapping")
        validated: dict[tuple[str, int], int] = {}
        for key, value in next_offsets.items():
            if (
                not isinstance(key, tuple)
                or len(key) != 2
                or not isinstance(key[0], str)
                or not key[0]
                or isinstance(key[1], bool)
                or not isinstance(key[1], int)
                or isinstance(value, bool)
                or not isinstance(value, int)
                or key[1] < 0
                or value < 0
                or key[1] > _SQLITE_INT_MAX
                or value > _SQLITE_INT_MAX
            ):
                raise StateStoreError(f"Invalid Kafka offset entry: {key!r}={value!r}")
            validated[(key[0], key[1])] = value
        return validated

    def save_snapshot(
        self,
        payload: Mapping[str, Any],
        next_offsets: Mapping[tuple[str, int], int] | None = None,
        *,
        allow_rewind: bool = False,
    ) -> None:
        """Atomically save a snapshot and optional Kafka next offsets.

        ``next_offsets`` uses ``(topic, partition) -> next offset`` semantics.
        Callers must only include offsets whose associated work is represented
        as complete in this snapshot. This method intentionally does not
        commit Kafka consumer offsets; the caller coordinates that step.

        Offset regressions are rejected by default. Set ``allow_rewind=True``
        only for an intentional recovery operation (for example, after a topic
        is recreated or an operator explicitly requests replay). This updates
        only the offsets supplied in this call; it does not reset other rows.
        """
        encoded = self._encode(payload)
        offsets = self._validate_offsets(next_offsets)
        conn = self._connection()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO runtime_snapshot
                        (singleton, schema_version, updated_at, payload_json)
                    VALUES (1, ?, ?, ?)
                    ON CONFLICT(singleton) DO UPDATE SET
                        schema_version=excluded.schema_version,
                        updated_at=excluded.updated_at,
                        payload_json=excluded.payload_json
                    """,
                    (STATE_SCHEMA_VERSION, time.time(), encoded),
                )
                for (topic, partition), next_offset in offsets.items():
                    if allow_rewind:
                        conn.execute(
                            """
                            INSERT INTO kafka_offsets(topic, partition_id, next_offset)
                            VALUES (?, ?, ?)
                            ON CONFLICT(topic, partition_id) DO UPDATE SET
                                next_offset=excluded.next_offset
                            """,
                            (topic, partition, next_offset),
                        )
                    else:
                        conn.execute(
                            """
                            INSERT INTO kafka_offsets(topic, partition_id, next_offset)
                            VALUES (?, ?, ?)
                            ON CONFLICT(topic, partition_id) DO UPDATE SET
                                next_offset=excluded.next_offset
                            WHERE excluded.next_offset >= kafka_offsets.next_offset
                            """,
                            (topic, partition, next_offset),
                        )
                        if conn.execute("SELECT changes()").fetchone()[0] == 0:
                            current = conn.execute(
                                "SELECT next_offset FROM kafka_offsets "
                                "WHERE topic=? AND partition_id=?",
                                (topic, partition),
                            ).fetchone()
                            if current is None or current[0] != next_offset:
                                raise StateStoreError(
                                    f"Kafka offset regression rejected for {topic}[{partition}]: "
                                    f"requested {next_offset}, stored "
                                    f"{current[0] if current else 'missing'}"
                                )
        except StateStoreError:
            raise
        except sqlite3.Error as exc:
            raise StateStoreError("Failed to atomically persist runtime state") from exc

    def load(self) -> LoadedState:
        """Read snapshot, offsets, and snapshot timestamp consistently.

        The explicit read transaction prevents another connection's commit
        from splitting the snapshot and offset reads across two revisions.
        """
        conn = self._connection()
        try:
            conn.execute("BEGIN")
            snapshot_row = conn.execute(
                "SELECT schema_version, updated_at, payload_json "
                "FROM runtime_snapshot WHERE singleton=1"
            ).fetchone()
            offset_rows = conn.execute(
                "SELECT topic, partition_id, next_offset FROM kafka_offsets"
            ).fetchall()
            conn.commit()
        except sqlite3.Error as exc:
            try:
                conn.rollback()
            except sqlite3.Error:
                pass
            raise StateStoreError("Failed to load runtime state") from exc

        payload: dict[str, Any] | None = None
        updated_at: float | None = None
        if snapshot_row is not None:
            version, updated_at, encoded = snapshot_row
            if version != STATE_SCHEMA_VERSION:
                raise StateStoreError(
                    f"Unsupported runtime state schema version {version}; "
                    f"expected {STATE_SCHEMA_VERSION}"
                )
            def reject_duplicate_keys(pairs):
                result = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError(f"duplicate JSON object key: {key!r}")
                    result[key] = value
                return result

            try:
                decoded = json.loads(encoded, object_pairs_hook=reject_duplicate_keys)
            except (TypeError, ValueError, json.JSONDecodeError, RecursionError) as exc:
                raise StateStoreError("Persisted runtime snapshot is invalid JSON") from exc
            if not isinstance(decoded, dict):
                raise StateStoreError("Persisted runtime snapshot must be a JSON object")
            payload = decoded

        offsets: dict[tuple[str, int], int] = {}
        for topic, partition, offset in offset_rows:
            if (
                not isinstance(topic, str)
                or not topic
                or isinstance(partition, bool)
                or not isinstance(partition, int)
                or isinstance(offset, bool)
                or not isinstance(offset, int)
                or partition < 0
                or offset < 0
                or partition > _SQLITE_INT_MAX
                or offset > _SQLITE_INT_MAX
            ):
                raise StateStoreError("Persisted Kafka offset entry is invalid")
            offsets[(topic, partition)] = offset
        return LoadedState(payload, offsets, updated_at)

    def load_snapshot(self) -> dict[str, Any] | None:
        """Load a snapshot. Prefer ``load()`` during coordinated recovery."""
        return self.load()[0]

    def load_offsets(self) -> dict[tuple[str, int], int]:
        """Load offsets. Prefer ``load()`` during coordinated recovery."""
        return self.load()[1]

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error as exc:
                raise StateStoreError("Failed to close runtime state database") from exc
            finally:
                self._conn = None

    def __enter__(self) -> "RealtimeStateStore":
        self._connection()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
            return
        try:
            self.close()
        except StateStoreError:
            # Preserve the exception raised by the with-body.
            pass
