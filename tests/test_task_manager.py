from __future__ import annotations

from datetime import timedelta

import pytest

from magnova_agent import (
    InMemoryMagNovaBackend,
    MagNovaAgent,
    MagNovaTaskManager,
    MagNovaToolAdapter,
    MultiTurnMagNovaAgent,
    PollingPolicy,
    RuleBasedInterpreter,
    SQLiteTaskStore,
)
from magnova_agent.models import AgentStatus, JobSnapshot, JobStatus
from magnova_agent.task_manager import utc_now
from magnova_agent.tooling import TransientMagNovaError


def complete_request():
    return RuleBasedInterpreter().interpret(
        "请生成磁矩密度大于 0.2 μB/Å³ 的材料"
    )


def policy(**overrides) -> PollingPolicy:
    values = {
        "timeout_seconds": 30.0,
        "initial_interval_seconds": 0.0,
        "max_interval_seconds": 0.0,
        "backoff_multiplier": 1.0,
        "max_transient_errors": 2,
    }
    values.update(overrides)
    return PollingPolicy(**values)


class MustNotRunTool:
    def submit(self, request):
        raise AssertionError("prepare must not call the backend tool")


def test_prepare_validates_without_calling_tool() -> None:
    agent = MagNovaAgent(
        interpreter=RuleBasedInterpreter(),
        tool=MustNotRunTool(),
    )

    outcome = agent.prepare("生成带隙为 1.5 eV 的材料")

    assert outcome.status == AgentStatus.READY
    assert outcome.tool_call is not None
    assert outcome.task_id is None


def test_multiturn_prepare_fills_slots_without_blocking() -> None:
    agent = MultiTurnMagNovaAgent(
        MagNovaAgent(
            interpreter=RuleBasedInterpreter(),
            tool=MustNotRunTool(),
        )
    )

    first = agent.prepare("设计磁矩密度大于 0.2 的材料", "session")
    second = agent.prepare("单位是 μB/Å³", "session")

    assert first.status == AgentStatus.NEEDS_CLARIFICATION
    assert second.status == AgentStatus.READY
    assert second.request.target_properties[0].unit == "μB/Å³"


def test_sqlite_store_survives_new_store_instance(tmp_path) -> None:
    database = tmp_path / "tasks.sqlite3"
    backend = InMemoryMagNovaBackend()
    snapshot = backend.submit_job(complete_request())
    first_store = SQLiteTaskStore(database)
    first_store.create("session-a", complete_request(), snapshot)

    reopened_store = SQLiteTaskStore(database)
    restored = reopened_store.get(snapshot.task_id, "session-a")

    assert restored.status == JobStatus.QUEUED
    assert restored.request == complete_request()


def test_task_manager_is_non_blocking_and_persists_transitions(tmp_path) -> None:
    backend = InMemoryMagNovaBackend(polls_before_success=2)
    adapter = MagNovaToolAdapter(backend, policy=policy())
    store = SQLiteTaskStore(tmp_path / "tasks.sqlite3")
    manager = MagNovaTaskManager(adapter, store)

    submitted = manager.submit(complete_request(), "session-a")
    running = manager.refresh(submitted.task_id, "session-a")
    succeeded = manager.refresh(submitted.task_id, "session-a")

    assert submitted.status == JobStatus.QUEUED
    assert running.status == JobStatus.RUNNING
    assert succeeded.status == JobStatus.SUCCEEDED
    assert len(succeeded.candidates) == 3
    assert SQLiteTaskStore(store.path).get(submitted.task_id) == succeeded


def test_same_request_in_two_sessions_gets_distinct_task_ids(tmp_path) -> None:
    backend = InMemoryMagNovaBackend()
    manager = MagNovaTaskManager(
        MagNovaToolAdapter(backend, policy=policy()),
        SQLiteTaskStore(tmp_path / "tasks.sqlite3"),
    )

    first = manager.submit(complete_request(), "session-a")
    second = manager.submit(complete_request(), "session-b")

    assert first.task_id != second.task_id
    assert manager.latest_for_session("session-a").task_id == first.task_id
    assert manager.latest_for_session("session-b").task_id == second.task_id
    with pytest.raises(KeyError):
        manager.store.get(first.task_id, "session-b")


class AlwaysFlakyBackend:
    def submit_job(self, request):
        return JobSnapshot(task_id="flaky-task", status=JobStatus.QUEUED)

    def get_job(self, task_id):
        raise TransientMagNovaError("temporary overload")


def test_transient_error_budget_is_persisted(tmp_path) -> None:
    manager = MagNovaTaskManager(
        MagNovaToolAdapter(
            AlwaysFlakyBackend(),
            policy=policy(max_transient_errors=1),
        ),
        SQLiteTaskStore(tmp_path / "tasks.sqlite3"),
    )
    submitted = manager.submit(complete_request(), "session-a")

    first = manager.refresh(submitted.task_id, "session-a")
    second = manager.refresh(submitted.task_id, "session-a")

    assert first.status == JobStatus.QUEUED
    assert first.transient_error_count == 1
    assert second.status == JobStatus.FAILED
    assert "too many transient" in second.error_message


def test_expired_task_is_marked_failed_without_polling(tmp_path) -> None:
    backend = InMemoryMagNovaBackend()
    store = SQLiteTaskStore(tmp_path / "tasks.sqlite3")
    manager = MagNovaTaskManager(
        MagNovaToolAdapter(backend, policy=policy(timeout_seconds=0)),
        store,
    )
    submitted = manager.submit(complete_request(), "session-a")

    expired = manager.refresh(submitted.task_id, "session-a")

    assert expired.status == JobStatus.FAILED
    assert expired.created_at <= utc_now() + timedelta(seconds=1)
    assert "deadline" in expired.error_message
