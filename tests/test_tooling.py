from __future__ import annotations

import pytest

from magnova_agent import MagNovaAgent, RuleBasedInterpreter
from magnova_agent.models import JobSnapshot, JobStatus
from magnova_agent.tooling import (
    InMemoryMagNovaBackend,
    MagNovaJobFailed,
    MagNovaJobTimeout,
    MagNovaToolAdapter,
    PollingPolicy,
    TransientMagNovaError,
    generate_mock_candidates,
)


def complete_request():
    return RuleBasedInterpreter().interpret(
        "请生成磁矩密度大于 0.2 μB/Å³ 的材料"
    )


def instant_policy(**overrides) -> PollingPolicy:
    values = {
        "timeout_seconds": 1.0,
        "initial_interval_seconds": 0.0,
        "max_interval_seconds": 0.0,
        "backoff_multiplier": 1.0,
        "max_transient_errors": 2,
    }
    values.update(overrides)
    return PollingPolicy(**values)


def test_adapter_submits_polls_and_returns_candidates() -> None:
    adapter = MagNovaToolAdapter(
        InMemoryMagNovaBackend(polls_before_success=2),
        policy=instant_policy(),
    )

    result = adapter.submit(complete_request())

    assert result.task_id.startswith("mock-async-")
    assert result.backend_status == JobStatus.SUCCEEDED
    assert len(result.candidates) == 3


def test_agent_exposes_backend_task_id() -> None:
    adapter = MagNovaToolAdapter(
        InMemoryMagNovaBackend(polls_before_success=1),
        policy=instant_policy(),
    )
    agent = MagNovaAgent(
        interpreter=RuleBasedInterpreter(),
        tool=adapter,
    )

    outcome = agent.run("生成带隙为 1.5 eV 的材料")

    assert outcome.task_id is not None
    assert outcome.task_id.startswith("mock-async-")
    assert outcome.backend_status == JobStatus.SUCCEEDED
    assert len(outcome.candidates) == 3


class FlakyBackend:
    def __init__(self) -> None:
        self.request = None
        self.poll_count = 0

    def submit_job(self, request):
        self.request = request
        return JobSnapshot(task_id="flaky-1", status=JobStatus.QUEUED)

    def get_job(self, task_id):
        self.poll_count += 1
        if self.poll_count == 1:
            raise TransientMagNovaError("temporary overload")
        return JobSnapshot(
            task_id=task_id,
            status=JobStatus.SUCCEEDED,
            candidates=generate_mock_candidates(self.request),
        )


def test_adapter_retries_transient_poll_error() -> None:
    backend = FlakyBackend()
    adapter = MagNovaToolAdapter(backend, policy=instant_policy())

    result = adapter.submit(complete_request())

    assert backend.poll_count == 2
    assert result.backend_status == JobStatus.SUCCEEDED


class FailedBackend:
    def submit_job(self, request):
        return JobSnapshot(
            task_id="failed-1",
            status=JobStatus.FAILED,
            error_message="generation crashed",
        )

    def get_job(self, task_id):
        raise AssertionError("terminal submission must not be polled")


def test_adapter_surfaces_terminal_failure() -> None:
    adapter = MagNovaToolAdapter(FailedBackend(), policy=instant_policy())

    with pytest.raises(MagNovaJobFailed, match="generation crashed"):
        adapter.submit(complete_request())


class NeverFinishedBackend:
    def submit_job(self, request):
        return JobSnapshot(task_id="slow-1", status=JobStatus.QUEUED)

    def get_job(self, task_id):
        return JobSnapshot(task_id=task_id, status=JobStatus.RUNNING)


def test_adapter_stops_at_timeout() -> None:
    adapter = MagNovaToolAdapter(
        NeverFinishedBackend(),
        policy=instant_policy(timeout_seconds=0.0),
    )

    with pytest.raises(MagNovaJobTimeout, match="timed out"):
        adapter.submit(complete_request())
