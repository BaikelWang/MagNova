from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


_REDACTION_RULES = (
    (re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"), "<EMAIL>"),
    (re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)"), "<PHONE>"),
    (re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)"), "<IP>"),
    (re.compile(r"(?i)\b(?:sk-[A-Za-z0-9_-]{12,}|bearer\s+[A-Za-z0-9._-]{12,})\b"), "<SECRET>"),
)


def redact_text(value: str | None) -> str | None:
    """Redact common identifiers without deleting scientific numbers/units."""
    if value is None:
        return None
    redacted = value
    for pattern, replacement in _REDACTION_RULES:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def prompt_fingerprint(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class InteractionRecorder:
    """Append privacy-aware MiniMind interaction traces to JSONL.

    The recorder is opt-in: callers must construct and inject it explicitly.
    Each record is written with one O_APPEND syscall so concurrent sessions do
    not overwrite previous samples.
    """

    path: Path
    model_version: str

    def record(
        self,
        *,
        prompt_hash: str,
        original_text: str,
        normalized_text: str,
        raw_output: str,
        outcome: str,
        latency_ms: float,
        validated_request: dict[str, Any] | None = None,
        warnings: list[str] | None = None,
        error: BaseException | None = None,
    ) -> str:
        trace_id = uuid4().hex
        payload = {
            "schema_version": "magnova.interaction.v1",
            "trace_id": trace_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_version": self.model_version,
            "prompt_sha256": prompt_hash,
            "outcome": outcome,
            "latency_ms": round(latency_ms, 3),
            "input": {
                "user_text": redact_text(original_text),
                "normalized_text": redact_text(normalized_text),
            },
            "model": {"raw_output": redact_text(raw_output)},
            "validation": {
                "request": validated_request,
                "warnings": list(warnings or []),
            },
            "error": None
            if error is None
            else {
                "type": type(error).__name__,
                "message": redact_text(str(error)),
            },
            "review": {"status": "pending", "training_approved": False},
        }
        encoded = (
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            self.path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o600,
        )
        try:
            os.write(descriptor, encoded)
        finally:
            os.close(descriptor)
        return trace_id
