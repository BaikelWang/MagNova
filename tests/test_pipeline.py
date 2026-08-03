from __future__ import annotations

import sqlite3

from magnova_agent import (
    MagNovaTaskManager,
    MagNovaToolAdapter,
    MockMaterialGenerator,
    MockPipelineMagNovaBackend,
    MockPropertyPredictor,
    PollingPolicy,
    RuleBasedCandidateScreener,
    RuleBasedInterpreter,
    SQLiteTaskStore,
)
from magnova_agent.models import (
    Comparator,
    CompositionConstraint,
    GeneratedMaterial,
    JobStatus,
    MaterialRequest,
    PredictedMaterial,
    PropertyConstraint,
    WorkflowStage,
)


def request(text: str) -> MaterialRequest:
    return RuleBasedInterpreter().interpret(text)


def policy() -> PollingPolicy:
    return PollingPolicy(
        timeout_seconds=30,
        initial_interval_seconds=0,
        max_interval_seconds=0,
        backoff_multiplier=1,
    )


def test_generator_respects_required_and_excluded_elements() -> None:
    material_request = MaterialRequest(
        original_text="test",
        target_properties=[
            PropertyConstraint(
                name="band_gap",
                operator=Comparator.GT,
                value=1.0,
                unit="eV",
            )
        ],
        composition=CompositionConstraint(
            required_elements=["Fe"],
            excluded_elements=["Ni"],
        ),
    )

    generated = MockMaterialGenerator().generate(material_request)

    assert generated
    assert all("Fe" in item.formula for item in generated)
    assert all("Ni" not in item.formula for item in generated)


def test_predictor_outputs_every_requested_property_and_uncertainty() -> None:
    constraints = [
        PropertyConstraint(
            name="band_gap",
            operator=Comparator.GTE,
            value=1.5,
            unit="eV",
        ),
        PropertyConstraint(
            name="curie_temperature",
            operator=Comparator.GT,
            value=500,
            unit="K",
        ),
    ]
    generated = [
        GeneratedMaterial(
            structure_id="g-1",
            formula="FeCoB2",
            generation_score=0.9,
        )
    ]

    predicted = MockPropertyPredictor().predict(generated, constraints)

    assert set(predicted[0].predicted_properties) == {
        "band_gap",
        "curie_temperature",
    }
    assert set(predicted[0].uncertainties) == {
        "band_gap",
        "curie_temperature",
    }


def test_screener_uses_deterministic_numeric_comparison() -> None:
    material_request = request("生成带隙大于 1.5 eV 的材料")
    predicted = [
        PredictedMaterial(
            structure_id="pass",
            formula="FeCoB2",
            generation_score=0.9,
            predicted_properties={"band_gap": 1.6},
            uncertainties={"band_gap": 0.03},
        ),
        PredictedMaterial(
            structure_id="fail",
            formula="MnAlC",
            generation_score=0.95,
            predicted_properties={"band_gap": 1.4},
            uncertainties={"band_gap": 0.01},
        ),
    ]

    candidates = RuleBasedCandidateScreener().screen(
        predicted,
        material_request,
    )

    assert [item.candidate_id for item in candidates] == ["pass"]


def test_equality_constraint_keeps_at_least_one_mock_candidate() -> None:
    material_request = request("生成带隙为 1.5 eV 的材料")
    generated = MockMaterialGenerator().generate(material_request)
    predicted = MockPropertyPredictor().predict(
        generated,
        material_request.target_properties,
    )

    candidates = RuleBasedCandidateScreener().screen(
        predicted,
        material_request,
    )

    assert candidates
    assert all(item.predicted_properties["band_gap"] == 1.5 for item in candidates)


def test_pipeline_advances_generate_predict_screen() -> None:
    backend = MockPipelineMagNovaBackend()
    material_request = request("生成带隙大于 1.5 eV 的材料")
    submitted = backend.submit_job(material_request)
    generated = backend.get_job(submitted.task_id)
    predicted = backend.get_job(submitted.task_id)
    screened = backend.get_job(submitted.task_id)

    assert submitted.status == JobStatus.QUEUED
    assert submitted.stage == WorkflowStage.QUEUED
    assert generated.stage == WorkflowStage.GENERATE
    assert generated.progress == 0.33
    assert predicted.stage == WorkflowStage.PREDICT
    assert predicted.progress == 0.67
    assert screened.status == JobStatus.SUCCEEDED
    assert screened.stage == WorkflowStage.COMPLETED
    assert screened.progress == 1.0
    assert [event.stage for event in screened.workflow_trace] == [
        WorkflowStage.GENERATE,
        WorkflowStage.PREDICT,
        WorkflowStage.SCREEN,
    ]
    assert screened.candidates


def test_task_manager_persists_pipeline_trace(tmp_path) -> None:
    manager = MagNovaTaskManager(
        MagNovaToolAdapter(
            MockPipelineMagNovaBackend(),
            policy=policy(),
        ),
        SQLiteTaskStore(tmp_path / "tasks.sqlite3"),
    )
    submitted = manager.submit(
        request("生成带隙大于 1.5 eV 的材料"),
        "session",
    )

    generation = manager.refresh(submitted.task_id, "session")
    prediction = manager.refresh(submitted.task_id, "session")
    completed = manager.refresh(submitted.task_id, "session")
    reopened = SQLiteTaskStore(manager.store.path).get(
        submitted.task_id,
        "session",
    )

    assert generation.stage == WorkflowStage.GENERATE
    assert prediction.stage == WorkflowStage.PREDICT
    assert completed.stage == WorkflowStage.COMPLETED
    assert reopened == completed
    assert len(reopened.workflow_trace) == 3


def test_sqlite_additively_migrates_v2_database(tmp_path) -> None:
    database = tmp_path / "old.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE agent_tasks (
                task_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                request_json TEXT NOT NULL,
                status TEXT NOT NULL,
                candidates_json TEXT NOT NULL,
                error_message TEXT,
                transient_error_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

    SQLiteTaskStore(database)

    with sqlite3.connect(database) as connection:
        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(agent_tasks)"
            ).fetchall()
        }
    assert {"stage", "progress", "workflow_trace_json"} <= columns
