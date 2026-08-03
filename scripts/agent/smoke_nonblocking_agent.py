from __future__ import annotations

import tempfile
import time
from pathlib import Path

from magnova_agent import (
    InMemoryMagNovaBackend,
    MagNovaAgent,
    MagNovaTaskManager,
    MagNovaToolAdapter,
    MiniMindInterpreter,
    MultiTurnMagNovaAgent,
    PollingPolicy,
    SQLiteTaskStore,
)


with tempfile.TemporaryDirectory() as directory:
    adapter = MagNovaToolAdapter(
        InMemoryMagNovaBackend(polls_before_success=2),
        PollingPolicy(
            timeout_seconds=5,
            initial_interval_seconds=0,
            max_interval_seconds=0,
            backoff_multiplier=1,
        ),
    )
    database = Path(directory) / "tasks.sqlite3"
    manager = MagNovaTaskManager(adapter, SQLiteTaskStore(database))
    conversation = MultiTurnMagNovaAgent(
        MagNovaAgent(
            interpreter=MiniMindInterpreter(repair_attempts=1),
            tool=adapter,
        )
    )

    first = conversation.prepare(
        "请设计磁矩密度大于0.2的材料",
        session_id="smoke",
    )
    second = conversation.prepare(
        "单位是 μB/Å³",
        session_id="smoke",
    )

    started = time.monotonic()
    submitted = manager.submit(second.request, session_id="smoke")
    submit_elapsed = time.monotonic() - started
    conversation.track_task(submitted.task_id, "smoke")

    reopened = SQLiteTaskStore(database).get(
        submitted.task_id,
        "smoke",
    )
    running = manager.refresh(submitted.task_id, "smoke")
    succeeded = manager.refresh(submitted.task_id, "smoke")
    conversation.complete_task(submitted.task_id, "smoke")

    print(f"first_status={first.status.value}")
    print(f"second_status={second.status.value}")
    print(f"submit_status={submitted.status.value}")
    print(f"submit_elapsed_ms={submit_elapsed * 1000:.2f}")
    print(f"reopened_status={reopened.status.value}")
    print(f"poll_statuses={running.status.value}->{succeeded.status.value}")
    print(f"task_id={succeeded.task_id}")
    print(f"candidate_count={len(succeeded.candidates)}")
