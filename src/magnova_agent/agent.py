from __future__ import annotations

from .interpreter import RequirementInterpreter, RuleBasedInterpreter
from .models import AgentOutcome, AgentStatus, MaterialRequest, ToolCall
from .tooling import (
    MagNovaBlocked,
    MagNovaNeedsClarification,
    MagNovaTool,
    MockMagNovaTool,
)


class MagNovaAgent:
    def __init__(
        self,
        interpreter: RequirementInterpreter | None = None,
        tool: MagNovaTool | None = None,
    ) -> None:
        self.interpreter = interpreter or RuleBasedInterpreter()
        self.tool = tool or MockMagNovaTool()

    def run(self, user_text: str) -> AgentOutcome:
        request = self.interpreter.interpret(user_text)
        return self.execute(request)

    def prepare(self, user_text: str) -> AgentOutcome:
        """Interpret and validate without waiting for the MagNova backend."""
        request = self.interpreter.interpret(user_text)
        return self.prepare_request(request)

    def prepare_request(self, request: MaterialRequest) -> AgentOutcome:
        """Return clarification or a tool-ready request without executing it."""
        questions = request.clarification_questions()
        if questions:
            return AgentOutcome(
                status=AgentStatus.NEEDS_CLARIFICATION,
                request=request,
                clarification_questions=questions,
                message="在调用 MagNova 前还需要补充信息。",
            )

        tool_call = ToolCall(
            name="submit_magnova_task",
            arguments=request.model_dump(mode="json"),
        )
        return AgentOutcome(
            status=AgentStatus.READY,
            request=request,
            tool_call=tool_call,
            message="需求已通过校验，可以提交 MagNova。",
        )

    def execute(self, request: MaterialRequest) -> AgentOutcome:
        """Validate and execute an already interpreted request."""
        prepared = self.prepare_request(request)
        if prepared.status != AgentStatus.READY:
            return prepared

        try:
            result = self.tool.submit(request)
        except MagNovaNeedsClarification as exc:
            return AgentOutcome(
                status=AgentStatus.NEEDS_CLARIFICATION,
                request=request,
                tool_call=prepared.tool_call,
                clarification_questions=list(exc.questions)
                or [str(exc)],
                message=f"MagNova 需要补充信息：{exc}",
            )
        except MagNovaBlocked as exc:
            return AgentOutcome(
                status=AgentStatus.FAILED,
                request=request,
                tool_call=prepared.tool_call,
                message=(
                    f"MagNova 拒绝执行（{exc.block_reason or 'blocked'}）：{exc}。"
                    "未回退 Mock 候选。"
                ),
            )
        except Exception as exc:  # backend boundary: convert failures to domain outcome
            return AgentOutcome(
                status=AgentStatus.FAILED,
                request=request,
                tool_call=prepared.tool_call,
                message=f"MagNova 工具调用失败（未回退 Mock）：{exc}",
            )

        return AgentOutcome(
            status=AgentStatus.SUBMITTED,
            request=request,
            tool_call=prepared.tool_call,
            task_id=result.task_id,
            backend_status=result.backend_status,
            candidates=result.candidates,
            message=(
                f"MagNova 真实任务 {result.task_id} 已完成，"
                f"返回 {len(result.candidates)} 个 AI 筛选候选（非 DFT/实验验证）。"
            ),
        )
