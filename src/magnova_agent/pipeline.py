from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol
from uuid import uuid4

from .models import (
    CandidateMaterial,
    Comparator,
    GeneratedMaterial,
    JobSnapshot,
    JobStatus,
    MaterialRequest,
    PredictedMaterial,
    PropertyConstraint,
    WorkflowEvent,
    WorkflowStage,
)
from .tooling import PermanentMagNovaError


class MaterialGenerator(Protocol):
    def generate(self, request: MaterialRequest) -> list[GeneratedMaterial]:
        """Generate candidate crystal structures from a validated request."""


class PropertyPredictor(Protocol):
    def predict(
        self,
        materials: list[GeneratedMaterial],
        constraints: list[PropertyConstraint],
    ) -> list[PredictedMaterial]:
        """Predict requested material properties and uncertainty."""


class CandidateScreener(Protocol):
    def screen(
        self,
        materials: list[PredictedMaterial],
        request: MaterialRequest,
    ) -> list[CandidateMaterial]:
        """Apply hard constraints, score and rank the surviving candidates."""


ELEMENT_PATTERN = re.compile(r"[A-Z][a-z]?")
RARE_EARTH_ELEMENTS = {
    "Sc", "Y", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
}


class MockMaterialGenerator:
    """Deterministic interface simulator; it is not a generative model."""

    BASE_FORMULAS = (
        "FeCoB2",
        "Fe2NiP",
        "MnAlC",
        "Co2FeSi",
        "NiMnSb",
        "Fe3Si",
    )

    @staticmethod
    def _elements(formula: str) -> set[str]:
        return set(ELEMENT_PATTERN.findall(formula))

    def generate(self, request: MaterialRequest) -> list[GeneratedMaterial]:
        required = set(request.composition.required_elements)
        excluded = set(request.composition.excluded_elements)
        if request.composition.exclude_rare_earth:
            excluded |= RARE_EARTH_ELEMENTS

        formulas = []
        for formula in self.BASE_FORMULAS:
            elements = self._elements(formula)
            if required and not required.issubset(elements):
                continue
            if excluded.intersection(elements):
                continue
            formulas.append(formula)

        if not formulas and required:
            fallback = "".join(sorted(required)) + "C"
            if not excluded.intersection(self._elements(fallback)):
                formulas.append(fallback)

        return [
            GeneratedMaterial(
                structure_id=f"generated-{index:03d}",
                formula=formula,
                generation_score=round(max(0.55, 0.96 - index * 0.06), 3),
                generator_metadata={
                    "backend": "mock_generator",
                    "material_family": request.material_family,
                },
            )
            for index, formula in enumerate(formulas, start=1)
        ]


class MockPropertyPredictor:
    """Produce deterministic values around each requested constraint."""

    DEFAULTS = {
        "magnetic_moment_density": 0.28,
        "curie_temperature": 620.0,
        "formation_energy": -0.62,
        "band_gap": 1.5,
    }
    OFFSETS = (1.5, 0.8, 0.25, -0.4, -1.0, 0.5)

    @classmethod
    def _predict_value(
        cls,
        constraint: PropertyConstraint,
        index: int,
    ) -> float:
        offset = cls.OFFSETS[index % len(cls.OFFSETS)]
        default = cls.DEFAULTS.get(constraint.name, 1.0)
        if constraint.operator == Comparator.RANGE:
            assert constraint.lower_bound is not None
            assert constraint.upper_bound is not None
            midpoint = (constraint.lower_bound + constraint.upper_bound) / 2
            half_width = max(
                (constraint.upper_bound - constraint.lower_bound) / 2,
                0.05,
            )
            return midpoint + offset * half_width * 0.75
        if constraint.operator in {Comparator.MAXIMIZE, Comparator.MINIMIZE}:
            direction = 1 if constraint.operator == Comparator.MAXIMIZE else -1
            return default + direction * offset * max(abs(default) * 0.1, 0.05)

        assert constraint.value is not None
        margin = max(abs(constraint.value) * 0.1, 0.05)
        if constraint.operator in {Comparator.GT, Comparator.GTE}:
            return constraint.value + offset * margin
        if constraint.operator in {Comparator.LT, Comparator.LTE}:
            return constraint.value - offset * margin
        if index % 3 == 0:
            return constraint.value
        return constraint.value + offset * margin * 0.25

    def predict(
        self,
        materials: list[GeneratedMaterial],
        constraints: list[PropertyConstraint],
    ) -> list[PredictedMaterial]:
        results = []
        for index, material in enumerate(materials):
            predicted = {
                constraint.name: round(
                    self._predict_value(constraint, index),
                    6,
                )
                for constraint in constraints
            }
            uncertainties = {
                name: round(max(abs(value) * 0.03, 0.01), 6)
                for name, value in predicted.items()
            }
            results.append(
                PredictedMaterial(
                    structure_id=material.structure_id,
                    formula=material.formula,
                    generation_score=material.generation_score,
                    predicted_properties=predicted,
                    uncertainties=uncertainties,
                )
            )
        return results


class RuleBasedCandidateScreener:
    """Deterministic hard filtering; the LLM never decides numeric matches."""

    @staticmethod
    def matches(value: float, constraint: PropertyConstraint) -> bool:
        if constraint.operator == Comparator.GT:
            return constraint.value is not None and value > constraint.value
        if constraint.operator == Comparator.GTE:
            return constraint.value is not None and value >= constraint.value
        if constraint.operator == Comparator.LT:
            return constraint.value is not None and value < constraint.value
        if constraint.operator == Comparator.LTE:
            return constraint.value is not None and value <= constraint.value
        if constraint.operator == Comparator.EQ:
            return constraint.value is not None and abs(value - constraint.value) <= 1e-9
        if constraint.operator == Comparator.RANGE:
            return (
                constraint.lower_bound is not None
                and constraint.upper_bound is not None
                and constraint.lower_bound <= value <= constraint.upper_bound
            )
        return True

    def screen(
        self,
        materials: list[PredictedMaterial],
        request: MaterialRequest,
    ) -> list[CandidateMaterial]:
        survivors: list[CandidateMaterial] = []
        for material in materials:
            checks = [
                self.matches(
                    material.predicted_properties[constraint.name],
                    constraint,
                )
                for constraint in request.target_properties
            ]
            if not all(checks):
                continue
            relative_uncertainties = [
                material.uncertainties[name]
                / max(abs(material.predicted_properties[name]), 1e-12)
                for name in material.uncertainties
            ]
            uncertainty_penalty = sum(relative_uncertainties) / max(
                len(relative_uncertainties), 1
            )
            score = min(
                1.0,
                max(
                    0.0,
                    0.65 * material.generation_score
                    + 0.35 * (1.0 / (1.0 + uncertainty_penalty)),
                ),
            )
            survivors.append(
                CandidateMaterial(
                    candidate_id=material.structure_id,
                    formula=material.formula,
                    score=round(score, 3),
                    predicted_properties=material.predicted_properties,
                    rationale=(
                        "通过确定性筛选器的全部数值约束；"
                        "分数综合生成分数与预测不确定性。"
                    ),
                )
            )
        return sorted(survivors, key=lambda item: item.score, reverse=True)


@dataclass
class _PipelineJob:
    request: MaterialRequest
    generated: list[GeneratedMaterial] = field(default_factory=list)
    predicted: list[PredictedMaterial] = field(default_factory=list)
    candidates: list[CandidateMaterial] = field(default_factory=list)
    trace: list[WorkflowEvent] = field(default_factory=list)
    poll_count: int = 0


class MockPipelineMagNovaBackend:
    """Asynchronous three-stage MagNova workflow simulator."""

    def __init__(
        self,
        generator: MaterialGenerator | None = None,
        predictor: PropertyPredictor | None = None,
        screener: CandidateScreener | None = None,
    ) -> None:
        self.generator = generator or MockMaterialGenerator()
        self.predictor = predictor or MockPropertyPredictor()
        self.screener = screener or RuleBasedCandidateScreener()
        self._jobs: dict[str, _PipelineJob] = {}

    def submit_job(self, request: MaterialRequest) -> JobSnapshot:
        task_id = f"mock-pipeline-{uuid4().hex[:12]}"
        self._jobs[task_id] = _PipelineJob(
            request=request.model_copy(deep=True)
        )
        return JobSnapshot(
            task_id=task_id,
            status=JobStatus.QUEUED,
            stage=WorkflowStage.QUEUED,
            progress=0.0,
        )

    def get_job(self, task_id: str) -> JobSnapshot:
        if task_id not in self._jobs:
            raise PermanentMagNovaError(f"unknown MagNova task: {task_id}")
        job = self._jobs[task_id]
        job.poll_count += 1
        try:
            if job.poll_count == 1:
                job.generated = self.generator.generate(job.request)
                job.trace.append(
                    WorkflowEvent(
                        stage=WorkflowStage.GENERATE,
                        message="材料生成阶段完成",
                        artifact_count=len(job.generated),
                    )
                )
                return JobSnapshot(
                    task_id=task_id,
                    status=JobStatus.RUNNING,
                    stage=WorkflowStage.GENERATE,
                    progress=0.33,
                    workflow_trace=job.trace,
                )

            if job.poll_count == 2:
                job.predicted = self.predictor.predict(
                    job.generated,
                    job.request.target_properties,
                )
                job.trace.append(
                    WorkflowEvent(
                        stage=WorkflowStage.PREDICT,
                        message="性质预测阶段完成",
                        artifact_count=len(job.predicted),
                    )
                )
                return JobSnapshot(
                    task_id=task_id,
                    status=JobStatus.RUNNING,
                    stage=WorkflowStage.PREDICT,
                    progress=0.67,
                    workflow_trace=job.trace,
                )

            job.candidates = self.screener.screen(
                job.predicted,
                job.request,
            )
            job.trace.append(
                WorkflowEvent(
                    stage=WorkflowStage.SCREEN,
                    message="条件筛选与排序阶段完成",
                    artifact_count=len(job.candidates),
                )
            )
            return JobSnapshot(
                task_id=task_id,
                status=JobStatus.SUCCEEDED,
                stage=WorkflowStage.COMPLETED,
                progress=1.0,
                workflow_trace=job.trace,
                candidates=job.candidates,
            )
        except Exception as exc:
            job.trace.append(
                WorkflowEvent(
                    stage=WorkflowStage.COMPLETED,
                    message=f"工作流失败：{exc}",
                )
            )
            return JobSnapshot(
                task_id=task_id,
                status=JobStatus.FAILED,
                stage=WorkflowStage.COMPLETED,
                progress=1.0,
                workflow_trace=job.trace,
                error_message=str(exc),
            )
