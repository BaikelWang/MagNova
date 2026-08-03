from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from pydantic import Field

from .models import (
    CandidateMaterial,
    JobSnapshot,
    JobStatus,
    MaterialRequest,
    StrictModel,
    WorkflowEvent,
    WorkflowStage,
)
from .tooling import (
    MagNovaToolAdapter,
    PermanentMagNovaError,
    TransientMagNovaError,
)


def utc_now() -> datetime:
    return datetime.now(UTC)


class StoredTask(StrictModel):
    task_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    request: MaterialRequest
    status: JobStatus
    stage: WorkflowStage = WorkflowStage.QUEUED
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    workflow_trace: list[WorkflowEvent] = Field(default_factory=list)
    candidates: list[CandidateMaterial] = Field(default_factory=list)
    error_message: str | None = None
    transient_error_count: int = Field(default=0, ge=0)
    created_at: datetime
    updated_at: datetime

    @property
    def terminal(self) -> bool:
        return self.status in {JobStatus.SUCCEEDED, JobStatus.FAILED}


class SQLiteTaskStore:
    """Persistent task repository with additive schema migrations."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_tasks (
                    task_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL DEFAULT 'queued',
                    progress REAL NOT NULL DEFAULT 0,
                    workflow_trace_json TEXT NOT NULL DEFAULT '[]',
                    candidates_json TEXT NOT NULL,
                    error_message TEXT,
                    transient_error_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(agent_tasks)"
                ).fetchall()
            }
            migrations = {
                "stage": (
                    "ALTER TABLE agent_tasks "
                    "ADD COLUMN stage TEXT NOT NULL DEFAULT 'queued'"
                ),
                "progress": (
                    "ALTER TABLE agent_tasks "
                    "ADD COLUMN progress REAL NOT NULL DEFAULT 0"
                ),
                "workflow_trace_json": (
                    "ALTER TABLE agent_tasks "
                    "ADD COLUMN workflow_trace_json TEXT NOT NULL DEFAULT '[]'"
                ),
            }
            for column, statement in migrations.items():
                if column not in columns:
                    connection.execute(statement)
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_agent_tasks_session_updated
                ON agent_tasks(session_id, updated_at DESC)
                """
            )

    @staticmethod
    def _from_row(row: sqlite3.Row) -> StoredTask:
        return StoredTask(
            task_id=row["task_id"],
            session_id=row["session_id"],
            request=MaterialRequest.model_validate_json(row["request_json"]),
            status=JobStatus(row["status"]),
            stage=WorkflowStage(row["stage"]),
            progress=row["progress"],
            workflow_trace=[
                WorkflowEvent.model_validate(item)
                for item in json.loads(row["workflow_trace_json"])
            ],
            candidates=[
                CandidateMaterial.model_validate(item)
                for item in json.loads(row["candidates_json"])
            ],
            error_message=row["error_message"],
            transient_error_count=row["transient_error_count"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def create(
        self,
        session_id: str,
        request: MaterialRequest,
        snapshot: JobSnapshot,
    ) -> StoredTask:
        now = utc_now()
        with self._connect() as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO agent_tasks (
                        task_id, session_id, request_json, status,
                        stage, progress, workflow_trace_json,
                        candidates_json, error_message,
                        transient_error_count, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                    """,
                    (
                        snapshot.task_id,
                        session_id,
                        request.model_dump_json(),
                        snapshot.status.value,
                        snapshot.stage.value,
                        snapshot.progress,
                        json.dumps(
                            [
                                item.model_dump(mode="json")
                                for item in snapshot.workflow_trace
                            ],
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            [
                                item.model_dump(mode="json")
                                for item in snapshot.candidates
                            ],
                            ensure_ascii=False,
                        ),
                        snapshot.error_message,
                        now.isoformat(),
                        now.isoformat(),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(
                    f"duplicate MagNova task_id: {snapshot.task_id}"
                ) from exc
        return self.get(snapshot.task_id, session_id)

    def get(
        self,
        task_id: str,
        session_id: str | None = None,
    ) -> StoredTask:
        sql = "SELECT * FROM agent_tasks WHERE task_id = ?"
        parameters: tuple[str, ...] = (task_id,)
        if session_id is not None:
            sql += " AND session_id = ?"
            parameters = (task_id, session_id)
        with self._connect() as connection:
            row = connection.execute(sql, parameters).fetchone()
        if row is None:
            raise KeyError(f"unknown task: {task_id}")
        return self._from_row(row)

    def latest_for_session(self, session_id: str) -> StoredTask | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM agent_tasks
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()
        return None if row is None else self._from_row(row)

    def apply_snapshot(
        self,
        task_id: str,
        snapshot: JobSnapshot,
    ) -> StoredTask:
        if snapshot.task_id != task_id:
            raise ValueError("backend returned a mismatched task_id")
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE agent_tasks
                SET status = ?, stage = ?, progress = ?,
                    workflow_trace_json = ?, candidates_json = ?,
                    error_message = ?, transient_error_count = 0,
                    updated_at = ?
                WHERE task_id = ?
                """,
                (
                    snapshot.status.value,
                    snapshot.stage.value,
                    snapshot.progress,
                    json.dumps(
                        [
                            item.model_dump(mode="json")
                            for item in snapshot.workflow_trace
                        ],
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        [
                            item.model_dump(mode="json")
                            for item in snapshot.candidates
                        ],
                        ensure_ascii=False,
                    ),
                    snapshot.error_message,
                    utc_now().isoformat(),
                    task_id,
                ),
            )
        if cursor.rowcount != 1:
            raise KeyError(f"unknown task: {task_id}")
        return self.get(task_id)

    def record_transient_error(
        self,
        task_id: str,
        message: str,
    ) -> StoredTask:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE agent_tasks
                SET transient_error_count = transient_error_count + 1,
                    error_message = ?, updated_at = ?
                WHERE task_id = ?
                """,
                (message, utc_now().isoformat(), task_id),
            )
        if cursor.rowcount != 1:
            raise KeyError(f"unknown task: {task_id}")
        return self.get(task_id)

    def mark_failed(self, task_id: str, message: str) -> StoredTask:
        record = self.get(task_id)
        return self.apply_snapshot(
            task_id,
            JobSnapshot(
                task_id=task_id,
                status=JobStatus.FAILED,
                stage=WorkflowStage.COMPLETED,
                progress=1.0,
                workflow_trace=record.workflow_trace,
                error_message=message,
            ),
        )


class MagNovaTaskManager:
    """Non-blocking coordinator backed by a persistent task store."""

    def __init__(
        self,
        adapter: MagNovaToolAdapter,
        store: SQLiteTaskStore,
    ) -> None:
        self.adapter = adapter
        self.store = store

    def submit(
        self,
        request: MaterialRequest,
        session_id: str,
    ) -> StoredTask:
        snapshot = self.adapter.submit_job(request)
        return self.store.create(session_id, request, snapshot)

    def refresh(self, task_id: str, session_id: str) -> StoredTask:
        record = self.store.get(task_id, session_id)
        if record.terminal:
            return record

        age = (utc_now() - record.created_at).total_seconds()
        if age >= self.adapter.policy.timeout_seconds:
            return self.store.mark_failed(
                task_id,
                "client polling deadline exceeded",
            )

        try:
            snapshot = self.adapter.poll_job(task_id)
        except TransientMagNovaError as exc:
            record = self.store.record_transient_error(task_id, str(exc))
            if (
                record.transient_error_count
                > self.adapter.policy.max_transient_errors
            ):
                return self.store.mark_failed(
                    task_id,
                    "too many transient MagNova polling errors",
                )
            return record
        except PermanentMagNovaError as exc:
            return self.store.mark_failed(task_id, str(exc))
        except Exception as exc:
            return self.store.mark_failed(
                task_id,
                f"unexpected MagNova polling error: {exc}",
            )
        return self.store.apply_snapshot(task_id, snapshot)

    def latest_for_session(self, session_id: str) -> StoredTask | None:
        return self.store.latest_for_session(session_id)
