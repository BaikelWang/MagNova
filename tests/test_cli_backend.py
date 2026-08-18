from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from magnova_agent.models import (
    CompositionConstraint,
    JobStatus,
    MaterialRequest,
    WorkflowStage,
)
from magnova_agent.tooling import (
    CliMagNovaBackend,
    PermanentMagNovaError,
)


def ready_request() -> MaterialRequest:
    return MaterialRequest(
        composition=CompositionConstraint(exclude_rare_earth=True),
        original_text="generate a rare-earth-free candidate",
    )


def test_cli_backend_maps_real_response_without_mock_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}
    payload = {
        "status": "completed",
        "request_id": "real-001",
        "candidates": [
            {
                "candidate_id": "g1_gen_026",
                "formula": "Mn7HO12",
                "agent_score": 0.61,
                "queue": "exploration",
                "rank_score": 0.61,
                "claim_level": "ai_screening",
                "sacgt_proxy_probabilities": {
                    "p_is_stable_relaxed": 0.90,
                    "p_fm_ground_state_relaxed": 0.61,
                },
            }
        ],
    }

    def fake_run(command: list[str], **kwargs):
        captured["command"] = command
        captured["cwd"] = kwargs["cwd"]
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    backend = CliMagNovaBackend(
        sacgt_python="scientific-python",
        cli_script="private_backend.py",
        output_root="runtime/magnova_runs",
        working_directory="/private/backend",
    )

    snapshot = backend.submit_job(ready_request())

    assert snapshot.status == JobStatus.SUCCEEDED
    assert snapshot.stage == WorkflowStage.COMPLETED
    assert snapshot.task_id == "real-001"
    assert snapshot.candidates[0].candidate_id == "g1_gen_026"
    assert "NOT DFT/experimental verification" in snapshot.candidates[0].rationale
    assert captured["cwd"] == "/private/backend"
    assert captured["command"][0:2] == ["scientific-python", "private_backend.py"]


def test_cli_backend_rejects_mock_candidate_in_real_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "status": "completed",
        "request_id": "real-002",
        "candidates": [{"candidate_id": "mock-001", "formula": "Fe"}],
    }
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        ),
    )

    with pytest.raises(PermanentMagNovaError, match="Refusing Mock"):
        CliMagNovaBackend().submit_job(ready_request())
