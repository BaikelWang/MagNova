# Non-blocking task architecture

```text
User
  → MultiTurnMagNovaAgent.prepare
      → MiniMindInterpreter
      → MaterialRequest
      ├─ incomplete → needs_clarification
      └─ complete   → ready
                         ↓
                 MagNovaTaskManager.submit
                         ↓
                 MagNovaBackend.submit_job
                         ↓
                  task_id + queued
                         ↓
                  SQLiteTaskStore
                         ↓
          Streamlit fragment refreshes every second
                         ↓
                 MagNovaTaskManager.refresh
                         ↓
                 MagNovaBackend.get_job
                  ├─ queued/running
                  ├─ failed
                  └─ succeeded → candidates
```

The synchronous `MagNovaAgent.run` and `MagNovaToolAdapter.submit` APIs remain
available for command-line tools and backward-compatible tests. The web path
uses `prepare`, `submit_job` and `get_job` separately.
