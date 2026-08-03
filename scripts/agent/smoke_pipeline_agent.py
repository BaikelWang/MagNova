from __future__ import annotations

import tempfile
from pathlib import Path

from magnova_agent import (
    MagNovaAgent,
    MagNovaTaskManager,
    MagNovaToolAdapter,
    MiniMindInterpreter,
    MockPipelineMagNovaBackend,
    MultiTurnMagNovaAgent,
    PollingPolicy,
    SQLiteTaskStore,
)


with tempfile.TemporaryDirectory() as directory:
    adapter = MagNovaToolAdapter(
        MockPipelineMagNovaBackend(),
        PollingPolicy(
            timeout_seconds=10,
            initial_interval_seconds=0,
            max_interval_seconds=0,
            backoff_multiplier=1,
        ),
    )
    manager = MagNovaTaskManager(
        adapter,
        SQLiteTaskStore(Path(directory) / "tasks.sqlite3"),
    )
    conversation = MultiTurnMagNovaAgent(
        MagNovaAgent(
            interpreter=MiniMindInterpreter(repair_attempts=1),
            tool=adapter,
        )
    )

    prepared = conversation.prepare(
        "请生成带隙大于 1.2 eV 且居里温度高于 500 K 的材料",
        session_id="pipeline-smoke",
    )
    submitted = manager.submit(prepared.request, "pipeline-smoke")
    snapshots = [submitted]
    while not snapshots[-1].terminal:
        snapshots.append(
            manager.refresh(submitted.task_id, "pipeline-smoke")
        )

    final = snapshots[-1]
    print(f"prepare_status={prepared.status.value}")
    print(f"task_id={final.task_id}")
    print(
        "states="
        + " -> ".join(
            f"{item.status.value}/{item.stage.value}/{item.progress:.0%}"
            for item in snapshots
        )
    )
    print(
        "trace="
        + " -> ".join(
            f"{event.stage.value}:{event.artifact_count}"
            for event in final.workflow_trace
        )
    )
    print(f"candidate_count={len(final.candidates)}")
    for candidate in final.candidates:
        print(
            candidate.formula,
            candidate.predicted_properties,
            candidate.score,
        )
