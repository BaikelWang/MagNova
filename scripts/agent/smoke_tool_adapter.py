from __future__ import annotations

from magnova_agent import (
    InMemoryMagNovaBackend,
    MagNovaAgent,
    MagNovaToolAdapter,
    MiniMindInterpreter,
    MultiTurnMagNovaAgent,
    PollingPolicy,
)


tool = MagNovaToolAdapter(
    InMemoryMagNovaBackend(),
    PollingPolicy(
        timeout_seconds=5,
        initial_interval_seconds=0,
        max_interval_seconds=0,
        backoff_multiplier=1,
    ),
)
agent = MultiTurnMagNovaAgent(
    MagNovaAgent(
        interpreter=MiniMindInterpreter(repair_attempts=1),
        tool=tool,
    )
)

first = agent.run("请设计磁矩密度大于0.2的材料", session_id="smoke")
second = agent.run("单位是 μB/Å³", session_id="smoke")

print(f"first_status={first.status.value}")
print(second.model_dump_json(indent=2))
