from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class TaskType(str, Enum):
    GENERATE = "generate_material"
    PREDICT = "predict_properties"
    SCREEN = "screen_candidates"


class Comparator(str, Enum):
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    EQ = "eq"
    RANGE = "range"
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class AgentStatus(str, Enum):
    NEEDS_CLARIFICATION = "needs_clarification"
    READY = "ready"
    SUBMITTED = "submitted"
    FAILED = "failed"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class WorkflowStage(str, Enum):
    QUEUED = "queued"
    GENERATE = "generation"
    PREDICT = "prediction"
    SCREEN = "screening"
    COMPLETED = "completed"


class PropertyConstraint(StrictModel):
    name: str = Field(min_length=1)
    operator: Comparator
    value: float | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    unit: str | None = None
    raw_name: str | None = None

    @field_validator("name", "unit", "raw_name")
    @classmethod
    def empty_string_to_none_or_clean(cls, value: str | None, info: Any) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned and info.field_name != "name":
            return None
        return cleaned

    def completeness_issues(self, index: int) -> list[str]:
        prefix = f"target_properties[{index}]"
        issues: list[str] = []
        if self.operator == Comparator.RANGE:
            if self.lower_bound is None:
                issues.append(f"{prefix}.lower_bound")
            if self.upper_bound is None:
                issues.append(f"{prefix}.upper_bound")
            if (
                self.lower_bound is not None
                and self.upper_bound is not None
                and self.lower_bound > self.upper_bound
            ):
                issues.append(f"{prefix}.range_order")
        elif self.operator not in {Comparator.MAXIMIZE, Comparator.MINIMIZE}:
            if self.value is None:
                issues.append(f"{prefix}.value")
        if self.operator not in {Comparator.MAXIMIZE, Comparator.MINIMIZE} and self.unit is None:
            issues.append(f"{prefix}.unit")
        return issues


class CompositionConstraint(StrictModel):
    required_elements: list[str] = Field(default_factory=list)
    excluded_elements: list[str] = Field(default_factory=list)
    exclude_rare_earth: bool = False

    @field_validator("required_elements", "excluded_elements")
    @classmethod
    def unique_elements(cls, values: list[str]) -> list[str]:
        result: list[str] = []
        for value in values:
            cleaned = value.strip()
            if cleaned and cleaned not in result:
                result.append(cleaned)
        return result


class MaterialRequest(StrictModel):
    task: TaskType = TaskType.GENERATE
    material_family: str | None = None
    target_properties: list[PropertyConstraint] = Field(default_factory=list)
    composition: CompositionConstraint = Field(default_factory=CompositionConstraint)
    free_text_constraints: list[str] = Field(default_factory=list)
    original_text: str = Field(min_length=1)
    parser_warnings: list[str] = Field(default_factory=list)

    def readiness_issues(self) -> list[str]:
        issues: list[str] = []
        if not self.target_properties:
            issues.append("target_properties")
        for index, prop in enumerate(self.target_properties):
            issues.extend(prop.completeness_issues(index))
        return sorted(set(issues))

    @property
    def ready_for_submission(self) -> bool:
        return not self.readiness_issues()

    def clarification_questions(self) -> list[str]:
        questions: list[str] = []
        for issue in self.readiness_issues():
            if issue == "target_properties":
                questions.append("你希望候选材料满足哪些目标性质？例如磁矩密度、居里温度、形成能或带隙。")
            elif issue.endswith(".value"):
                questions.append("请补充该目标性质的数值或改为“尽可能高/尽可能低”。")
            elif issue.endswith(".unit"):
                questions.append("请补充该数值的单位，避免把不同物理量或量纲混淆。")
            elif issue.endswith(".lower_bound") or issue.endswith(".upper_bound"):
                questions.append("请补充目标范围的上下界。")
            elif issue.endswith(".range_order"):
                questions.append("目标范围的下界大于上界，请确认数值顺序。")
        return list(dict.fromkeys(questions))


class ToolCall(StrictModel):
    name: str
    arguments: dict[str, Any]


class GeneratedMaterial(StrictModel):
    structure_id: str = Field(min_length=1)
    formula: str = Field(min_length=1)
    generation_score: float = Field(ge=0.0, le=1.0)
    generator_metadata: dict[str, Any] = Field(default_factory=dict)


class PredictedMaterial(StrictModel):
    structure_id: str = Field(min_length=1)
    formula: str = Field(min_length=1)
    generation_score: float = Field(ge=0.0, le=1.0)
    predicted_properties: dict[str, float]
    uncertainties: dict[str, float] = Field(default_factory=dict)


class CandidateMaterial(StrictModel):
    candidate_id: str
    formula: str
    score: float = Field(ge=0.0, le=1.0)
    predicted_properties: dict[str, float]
    rationale: str


class WorkflowEvent(StrictModel):
    stage: WorkflowStage
    message: str = Field(min_length=1)
    artifact_count: int = Field(default=0, ge=0)


class JobSnapshot(StrictModel):
    task_id: str = Field(min_length=1)
    status: JobStatus
    stage: WorkflowStage = WorkflowStage.QUEUED
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    workflow_trace: list[WorkflowEvent] = Field(default_factory=list)
    candidates: list[CandidateMaterial] = Field(default_factory=list)
    error_message: str | None = None


class ToolResult(StrictModel):
    task_id: str = Field(min_length=1)
    backend_status: JobStatus
    candidates: list[CandidateMaterial] = Field(default_factory=list)


class AgentOutcome(StrictModel):
    status: AgentStatus
    request: MaterialRequest
    clarification_questions: list[str] = Field(default_factory=list)
    tool_call: ToolCall | None = None
    task_id: str | None = None
    backend_status: JobStatus | None = None
    candidates: list[CandidateMaterial] = Field(default_factory=list)
    message: str
