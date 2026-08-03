# Architecture

## Boundary

`RequirementInterpreter` is the only LLM-facing boundary. It converts user
text to `MaterialRequest`. Everything after it is deterministic and testable.

```text
User
  → RequirementInterpreter
  → MaterialRequest
  → readiness validation
      ├─ incomplete → clarification
      └─ complete   → MagNovaTool
                         → submit_job
                         → task_id
                         → poll_job
                         ├─ queued/running → retry with backoff
                         ├─ failed         → AgentOutcome.failed
                         └─ succeeded      → candidates
```

## Tool Adapter layers

The integration boundary is deliberately split into three layers:

1. `MagNovaBackend` describes the real asynchronous job protocol:
   `submit_job(request)` and `get_job(task_id)`.
2. `MagNovaToolAdapter` owns polling, backoff, retry limits, timeout and
   terminal-state handling.
3. `MagNovaAgent` only receives a normalized `ToolResult`; it does not know
   whether the backend is HTTP, in-memory, or a future Python SDK.

`HTTPMagNovaBackend` expects the following replaceable contract:

```text
POST /v1/tasks
GET  /v1/tasks/{task_id}
```

Both responses use:

```json
{
  "task_id": "job-123",
  "status": "queued | running | succeeded | failed",
  "candidates": [],
  "error_message": null
}
```

`InMemoryMagNovaBackend` implements the same contract with the deterministic
state transition `queued → running → succeeded`. It is an interface simulator,
not a scientific model.

## Replacement points

- `RuleBasedInterpreter` → `MiniMindInterpreter`
- `InMemoryMagNovaBackend` → `HTTPMagNovaBackend` or a MagNova Python SDK
- in-memory conversation store → Redis/PostgreSQL
- synchronous `MagNovaToolAdapter.submit` → background worker and UI polling
- CLI → FastAPI/Streamlit UI

The domain Schema and golden evaluation set remain stable across these
replacements.
