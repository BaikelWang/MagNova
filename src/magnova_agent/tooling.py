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
    WorkflowStage,
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


class MagNovaNeedsClarification(MagNovaToolError):
    """Backend rejected the request because scientific inputs are incomplete."""

    def __init__(self, message: str, questions: list[str] | None = None) -> None:
        super().__init__(message)
        self.questions = list(questions or [])


class MagNovaBlocked(MagNovaToolError):
    """Backend refused the request (unsupported task/property); never falls back to Mock."""

    def __init__(self, message: str, block_reason: str | None = None) -> None:
        super().__init__(message)
        self.block_reason = block_reason


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
    """Immediate deterministic backend retained for small unit tests / Mock mode.

    Never used as a silent fallback when Real/CLI/HTTP backends fail.
    """

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


def _map_magnova_response_candidates(payload: dict[str, Any]) -> list[CandidateMaterial]:
    """Map SACGT magnova_response_v1 candidates → Agent CandidateMaterial."""
    out: list[CandidateMaterial] = []
    for item in payload.get("candidates") or []:
        probs = item.get("sacgt_proxy_probabilities") or {}
        predicted: dict[str, float] = {}
        for key in (
            "p_is_stable_relaxed",
            "p_bandgap_gt02_relaxed",
            "p_fm_ground_state_relaxed",
            "ensemble_std_fm_relaxed",
            "total_m_fm_init_relaxed",
        ):
            if probs.get(key) is not None:
                predicted[key] = float(probs[key])
        if item.get("mattersim_total_energy") is not None:
            predicted["mattersim_total_energy"] = float(item["mattersim_total_energy"])
        score = item.get("agent_score")
        if score is None:
            score = float(item.get("rank_score") or 0.0)
            score = max(0.0, min(1.0, score))
        rationale = (
            f"REAL MagNova AI screening | queue={item.get('queue')} | "
            f"rank_score={item.get('rank_score')} | "
            f"claim_level={item.get('claim_level')} | "
            f"NOT DFT/experimental verification; "
            f"totalM is not FM ground-state evidence."
        )
        out.append(
            CandidateMaterial(
                candidate_id=str(item["candidate_id"]),
                formula=str(item.get("formula") or "UNKNOWN"),
                score=float(score),
                predicted_properties=predicted,
                rationale=rationale,
            )
        )
    return out


class CliMagNovaBackend:
    """Thin adapter: Agent MaterialRequest → SACGT ``run_magnova_request_v1.py``.

    Does not import SACGT scientific cores into the Agent venv. Failures raise
    typed errors and never invent Mock candidates.
    """

    def __init__(
        self,
        *,
        sacgt_python: str = "python",
        cli_script: str = "run_magnova_request.py",
        output_root: str = "runtime/magnova_runs",
        mode: str = "from_artifacts",
        request_timeout_seconds: float = 86400.0,
        working_directory: str | None = None,
    ) -> None:
        self.sacgt_python = sacgt_python
        self.cli_script = cli_script
        self.output_root = output_root
        self.mode = mode
        self.request_timeout_seconds = request_timeout_seconds
        self.working_directory = working_directory
        self._cache: dict[str, JobSnapshot] = {}

    def submit_job(self, request: MaterialRequest) -> JobSnapshot:
        import json
        import subprocess
        import tempfile
        from pathlib import Path

        payload = request.model_dump(mode="json")
        # Agent schema has no num_candidates; default 10 unless options injected later
        payload.setdefault("num_candidates", 10)
        payload["schema_version"] = "magnova_request_v1"
        payload["request_id"] = f"agent-{uuid4().hex[:12]}"
        with tempfile.TemporaryDirectory(prefix="magnova_agent_req_") as tmp:
            req_path = Path(tmp) / "request.json"
            req_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            cmd = [
                self.sacgt_python,
                self.cli_script,
                "--request",
                str(req_path),
                "--mode",
                self.mode,
                "--output-root",
                self.output_root,
            ]
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.request_timeout_seconds,
                    cwd=self.working_directory,
                )
            except subprocess.TimeoutExpired as exc:
                raise MagNovaJobTimeout(
                    f"SACGT MagNova CLI timed out after {self.request_timeout_seconds:g}s"
                ) from exc
            except OSError as exc:
                raise TransientMagNovaError(f"SACGT MagNova CLI spawn failed: {exc}") from exc

        stdout = (proc.stdout or "").strip()
        if not stdout:
            raise PermanentMagNovaError(
                f"SACGT MagNova CLI returned empty stdout (rc={proc.returncode}): "
                f"{(proc.stderr or '')[-500:]}"
            )
        try:
            body = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise PermanentMagNovaError(
                "SACGT MagNova CLI returned non-JSON stdout"
            ) from exc

        status = str(body.get("status") or "failed")
        task_id = str(body.get("request_id") or f"cli-{uuid4().hex[:12]}")
        if status == "needs_clarification":
            raise MagNovaNeedsClarification(
                body.get("message") or "needs_clarification",
                questions=list(body.get("clarification_questions") or []),
            )
        if status == "blocked":
            raise MagNovaBlocked(
                body.get("message") or "blocked",
                block_reason=body.get("block_reason"),
            )
        if status == "failed":
            raise MagNovaJobFailed(
                body.get("error") or body.get("message") or "MagNova failed"
            )
        if status not in {"completed", "partial"}:
            raise PermanentMagNovaError(f"unknown MagNova status: {status}")

        # Guard: refuse payloads that look like Mock
        for cand in body.get("candidates") or []:
            cid = str(cand.get("candidate_id") or "")
            if cid.startswith("mock-"):
                raise PermanentMagNovaError(
                    "Refusing Mock candidate_id in Real/CLI MagNova response"
                )
        if body.get("disclaimers", {}).get("mock_candidates_mixed"):
            raise PermanentMagNovaError("Backend claimed mock_candidates_mixed=true")

        snapshot = JobSnapshot(
            task_id=task_id,
            status=JobStatus.SUCCEEDED,
            stage=WorkflowStage.COMPLETED,
            progress=1.0,
            candidates=_map_magnova_response_candidates(body),
            error_message=None if status == "completed" else str(body.get("message")),
        )
        self._cache[task_id] = snapshot
        return snapshot

    def get_job(self, task_id: str) -> JobSnapshot:
        if task_id not in self._cache:
            raise PermanentMagNovaError(f"unknown MagNova CLI task: {task_id}")
        return self._cache[task_id]


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
