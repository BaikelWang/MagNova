from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import quote
from uuid import uuid4

import requests
from pydantic import ValidationError

from .models import (
    CandidateMaterial,
    Comparator,
    JobSnapshot,
    JobStatus,
    MaterialRequest,
    PropertyConstraint,
    ToolResult,
)


class MagNovaTool(Protocol):
    def submit(self, request: MaterialRequest) -> ToolResult:
        """Submit a validated request and return a normalized tool result."""


class MagNovaBackend(Protocol):
    def submit_job(self, request: MaterialRequest) -> JobSnapshot:
        """Submit work without assuming that it completes immediately."""

    def get_job(self, task_id: str) -> JobSnapshot:
        """Return the latest snapshot for a previously submitted job."""


class MagNovaToolError(RuntimeError):
    """Base error raised at the MagNova tool boundary."""


class TransientMagNovaError(MagNovaToolError):
    """A retryable transport or backend availability error."""


class PermanentMagNovaError(MagNovaToolError):
    """A non-retryable request or backend protocol error."""


class MagNovaJobFailed(MagNovaToolError):
    """The remote job reached a terminal failed state."""


class MagNovaJobTimeout(MagNovaToolError):
    """The remote job did not finish within the configured deadline."""


@dataclass(frozen=True)
class PollingPolicy:
    timeout_seconds: float = 300.0
    initial_interval_seconds: float = 1.0
    max_interval_seconds: float = 10.0
    backoff_multiplier: float = 1.5
    max_transient_errors: int = 3

    def __post_init__(self) -> None:
        if self.timeout_seconds < 0:
            raise ValueError("timeout_seconds must be non-negative")
        if self.initial_interval_seconds < 0:
            raise ValueError("initial_interval_seconds must be non-negative")
        if self.max_interval_seconds < self.initial_interval_seconds:
            raise ValueError(
                "max_interval_seconds must be greater than or equal to "
                "initial_interval_seconds"
            )
        if self.backoff_multiplier < 1:
            raise ValueError("backoff_multiplier must be at least 1")
        if self.max_transient_errors < 0:
            raise ValueError("max_transient_errors must be non-negative")


class MagNovaToolAdapter:
    """Bridge an asynchronous job backend to the Agent's tool interface.

    ``submit_job`` and ``poll_job`` expose the non-blocking primitives. The
    compatibility ``submit`` method waits for completion so the current Agent
    can keep a small synchronous API while MagNova is still being developed.
    """

    def __init__(
        self,
        backend: MagNovaBackend,
        policy: PollingPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.backend = backend
        self.policy = policy or PollingPolicy()
        self._sleep = sleep
        self._monotonic = monotonic

    @staticmethod
    def _ensure_ready(request: MaterialRequest) -> None:
        if not request.ready_for_submission:
            raise ValueError(
                f"request is incomplete: {request.readiness_issues()}"
            )

    def submit_job(self, request: MaterialRequest) -> JobSnapshot:
        self._ensure_ready(request)
        return self.backend.submit_job(request)

    def poll_job(self, task_id: str) -> JobSnapshot:
        return self.backend.get_job(task_id)

    def wait_for_result(
        self,
        task_id: str,
        initial_snapshot: JobSnapshot | None = None,
    ) -> ToolResult:
        started_at = self._monotonic()
        interval = self.policy.initial_interval_seconds
        transient_errors = 0
        snapshot = initial_snapshot or self.poll_job(task_id)

        while True:
            if snapshot.task_id != task_id:
                raise PermanentMagNovaError(
                    "MagNova backend returned a mismatched task_id"
                )
            if snapshot.status == JobStatus.SUCCEEDED:
                return ToolResult(
                    task_id=task_id,
                    backend_status=snapshot.status,
                    candidates=snapshot.candidates,
                )
            if snapshot.status == JobStatus.FAILED:
                detail = snapshot.error_message or "unknown backend error"
                raise MagNovaJobFailed(
                    f"MagNova task {task_id} failed: {detail}"
                )
            if self._monotonic() - started_at >= self.policy.timeout_seconds:
                raise MagNovaJobTimeout(
                    f"MagNova task {task_id} timed out after "
                    f"{self.policy.timeout_seconds:g}s"
                )

            self._sleep(interval)
            try:
                snapshot = self.poll_job(task_id)
                transient_errors = 0
            except TransientMagNovaError:
                transient_errors += 1
                if transient_errors > self.policy.max_transient_errors:
                    raise
            interval = min(
                interval * self.policy.backoff_multiplier,
                self.policy.max_interval_seconds,
            )

    def submit(self, request: MaterialRequest) -> ToolResult:
        snapshot = self.submit_job(request)
        return self.wait_for_result(snapshot.task_id, snapshot)


class HTTPMagNovaBackend:
    """HTTP implementation of the MagNova asynchronous job contract.

    Expected contract:

    - ``POST /v1/tasks`` with a ``MaterialRequest`` JSON body;
    - ``GET /v1/tasks/{task_id}`` for status and eventual candidates;
    - both endpoints return a ``JobSnapshot``-compatible JSON object.
    """

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        request_timeout_seconds: float = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        if not self.base_url:
            raise ValueError("base_url must not be empty")
        self.request_timeout_seconds = request_timeout_seconds
        self.session = session or requests.Session()
        if api_key:
            self.session.headers.update(
                {"Authorization": f"Bearer {api_key}"}
            )

    @staticmethod
    def _raise_for_status(response: requests.Response) -> None:
        if response.status_code == 429 or response.status_code >= 500:
            raise TransientMagNovaError(
                f"MagNova HTTP {response.status_code}"
            )
        if response.status_code >= 400:
            raise PermanentMagNovaError(
                f"MagNova HTTP {response.status_code}: "
                f"{response.text[:300]}"
            )

    def _request(self, method: str, path: str, **kwargs: Any) -> JobSnapshot:
        try:
            response = self.session.request(
                method,
                f"{self.base_url}{path}",
                timeout=self.request_timeout_seconds,
                **kwargs,
            )
            self._raise_for_status(response)
            return JobSnapshot.model_validate(response.json())
        except TransientMagNovaError:
            raise
        except requests.RequestException as exc:
            raise TransientMagNovaError(
                f"MagNova transport error: {exc}"
            ) from exc
        except (ValueError, ValidationError) as exc:
            raise PermanentMagNovaError(
                "MagNova returned an invalid job payload"
            ) from exc

    def submit_job(self, request: MaterialRequest) -> JobSnapshot:
        return self._request(
            "POST",
            "/v1/tasks",
            json=request.model_dump(mode="json"),
        )

    def get_job(self, task_id: str) -> JobSnapshot:
        return self._request("GET", f"/v1/tasks/{quote(task_id, safe='')}")


def _matching_value(prop: PropertyConstraint) -> float:
    defaults = {
        "magnetic_moment_density": 0.28,
        "curie_temperature": 620.0,
        "formation_energy": -0.62,
        "band_gap": 1.5,
    }
    default = defaults.get(prop.name, 1.0)
    margin = max(abs(prop.value or default) * 0.1, 0.05)
    if prop.operator in {Comparator.GT, Comparator.GTE} and prop.value is not None:
        return prop.value + margin
    if prop.operator in {Comparator.LT, Comparator.LTE} and prop.value is not None:
        return prop.value - margin
    if prop.operator == Comparator.EQ and prop.value is not None:
        return prop.value
    if prop.operator == Comparator.RANGE:
        assert prop.lower_bound is not None and prop.upper_bound is not None
        return (prop.lower_bound + prop.upper_bound) / 2
    if prop.operator == Comparator.MAXIMIZE:
        return default + max(abs(default) * 0.1, 0.05)
    if prop.operator == Comparator.MINIMIZE:
        return default - max(abs(default) * 0.1, 0.05)
    return default


def generate_mock_candidates(
    request: MaterialRequest,
) -> list[CandidateMaterial]:
    seed = hashlib.sha256(
        request.model_dump_json().encode("utf-8")
    ).hexdigest()[:8]
    predicted = {
        prop.name: round(_matching_value(prop), 6)
        for prop in request.target_properties
    }
    candidates: list[CandidateMaterial] = []
    for index, formula in enumerate(("FeCoB2", "Fe2NiP", "MnAlC"), start=1):
        candidates.append(
            CandidateMaterial(
                candidate_id=f"mock-{seed}-{index}",
                formula=formula,
                score=round(0.94 - index * 0.06, 3),
                predicted_properties={
                    name: round(value * (1.0 - (index - 1) * 0.015), 6)
                    for name, value in predicted.items()
                },
                rationale=(
                    "Mock candidate for interface and evaluation development; "
                    "not a scientific prediction."
                ),
            )
        )
    return candidates


class MockMagNovaTool:
    """Immediate deterministic backend retained for small unit tests."""

    def submit(self, request: MaterialRequest) -> ToolResult:
        if not request.ready_for_submission:
            raise ValueError(
                f"request is incomplete: {request.readiness_issues()}"
            )
        seed = hashlib.sha256(
            request.model_dump_json().encode("utf-8")
        ).hexdigest()[:8]
        return ToolResult(
            task_id=f"mock-sync-{seed}",
            backend_status=JobStatus.SUCCEEDED,
            candidates=generate_mock_candidates(request),
        )


@dataclass
class InMemoryMagNovaBackend:
    """Stateful fake backend that follows queued → running → succeeded."""

    polls_before_success: int = 2
    _requests: dict[str, MaterialRequest] = field(default_factory=dict)
    _poll_counts: dict[str, int] = field(default_factory=dict)

    def submit_job(self, request: MaterialRequest) -> JobSnapshot:
        task_id = f"mock-async-{uuid4().hex[:12]}"
        self._requests[task_id] = request.model_copy(deep=True)
        self._poll_counts[task_id] = 0
        return JobSnapshot(task_id=task_id, status=JobStatus.QUEUED)

    def get_job(self, task_id: str) -> JobSnapshot:
        if task_id not in self._requests:
            raise PermanentMagNovaError(f"unknown MagNova task: {task_id}")
        self._poll_counts[task_id] += 1
        poll_count = self._poll_counts[task_id]
        if poll_count < self.polls_before_success:
            return JobSnapshot(task_id=task_id, status=JobStatus.RUNNING)
        return JobSnapshot(
            task_id=task_id,
            status=JobStatus.SUCCEEDED,
            candidates=generate_mock_candidates(self._requests[task_id]),
        )
