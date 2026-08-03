from magnova_agent import MagNovaAgent, MiniMindInterpreter, RuleBasedInterpreter


agent = MagNovaAgent(
    interpreter=MiniMindInterpreter(
        repair_attempts=0,
        fallback=RuleBasedInterpreter(),
    )
)
outcome = agent.run(
    "请生成磁矩密度大于 0.2 μB/Å³ 且不含稀土元素的材料"
)
print(outcome.model_dump_json(indent=2))

assert outcome.status.value == "submitted"
assert outcome.tool_call is not None
assert any(
    warning.startswith("minimind_fallback:")
    for warning in outcome.request.parser_warnings
)
