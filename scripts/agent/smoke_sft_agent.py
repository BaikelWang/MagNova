from magnova_agent import MagNovaAgent, MiniMindInterpreter


agent = MagNovaAgent(
    interpreter=MiniMindInterpreter(
        repair_attempts=1,
        fallback=None,
    )
)
cases = [
    "请生成磁矩密度大于 0.2 μB/Å³ 且不含稀土元素的材料",
    "设计磁矩密度大于 0.2 的材料",
    "请从候选中筛选带隙至少 1.5 eV 的材料",
]

for text in cases:
    outcome = agent.run(text)
    print(outcome.model_dump_json(indent=2))
    assert not any(
        warning.startswith("minimind_fallback:")
        for warning in outcome.request.parser_warnings
    )
