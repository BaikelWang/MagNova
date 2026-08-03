from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from transformers import AutoTokenizer


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--limit", type=int, default=896)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    report: dict[str, object] = {
        "tokenizer": str(args.tokenizer),
        "limit": args.limit,
        "splits": {},
    }
    for split in (item.strip() for item in args.splits.split(",")):
        path = args.data_dir / f"{split}.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        lengths: list[int] = []
        longest: list[dict[str, int]] = []
        for index, row in enumerate(rows):
            token_ids = tokenizer.apply_chat_template(
                row["conversations"],
                tokenize=True,
                add_generation_prompt=False,
                open_thinking=False,
            )
            length = len(token_ids)
            lengths.append(length)
            longest.append({"index": index, "tokens": length})
        longest.sort(key=lambda item: item["tokens"], reverse=True)
        report["splits"][split] = {
            "samples": len(lengths),
            "min": min(lengths),
            "p50": percentile(lengths, 0.50),
            "p95": percentile(lengths, 0.95),
            "max": max(lengths),
            "over_limit": sum(length > args.limit for length in lengths),
            "longest": longest[:5],
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if any(
        split["over_limit"]
        for split in report["splits"].values()
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
