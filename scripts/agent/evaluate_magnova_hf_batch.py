from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from magnova_agent.interpreter import (
    apply_critical_guardrails,
    extract_json_object,
)
from magnova_agent.models import MaterialRequest
from magnova_agent.normalization import normalize_for_llm


FIELDS = (
    "task",
    "material_family",
    "target_properties",
    "composition",
    "free_text_constraints",
    "status",
)


def sort_key(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def comparable(request: MaterialRequest) -> dict[str, Any]:
    """Compare business semantics while excluding audit-only raw_name."""
    payload = request.model_dump(mode="json")
    properties = []
    for prop in payload["target_properties"]:
        prop = dict(prop)
        prop.pop("raw_name", None)
        properties.append(prop)
    composition = dict(payload["composition"])
    composition["required_elements"] = sorted(composition["required_elements"])
    composition["excluded_elements"] = sorted(composition["excluded_elements"])
    return {
        "task": payload["task"],
        "material_family": payload["material_family"],
        "target_properties": sorted(properties, key=sort_key),
        "composition": composition,
        "free_text_constraints": sorted(payload["free_text_constraints"]),
        "status": (
            "submitted"
            if request.ready_for_submission
            else "needs_clarification"
        ),
    }


def current_pipeline_finalize(
    request: MaterialRequest,
    original_text: str,
) -> MaterialRequest:
    """Replay the current production entity gate and critical guardrails."""
    result = request.model_copy(deep=True)
    normalization = normalize_for_llm(original_text)
    recognized_names = {
        mention.canonical_name for mention in normalization.property_mentions
    }
    unsupported = [
        prop.name
        for prop in result.target_properties
        if prop.name not in recognized_names
    ]
    if unsupported:
        result.target_properties = [
            prop
            for prop in result.target_properties
            if prop.name in recognized_names
        ]
        result.parser_warnings.append(
            "entity_gate_dropped_unlinked_properties:"
            + ",".join(sorted(set(unsupported)))
        )
    for prop in result.target_properties:
        if prop.raw_name is None:
            prop.raw_name = normalization.raw_name_for(prop.name)
    result.parser_warnings.extend(
        f"normalized:{item}" for item in normalization.trace
    )
    return apply_critical_guardrails(result, original_text)


def load_rows(data_path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in data_path.read_text(encoding="utf-8").splitlines()
    ]
    metadata_path = data_path.with_suffix(".meta.jsonl")
    if metadata_path.exists():
        metadata = [
            json.loads(line)
            for line in metadata_path.read_text(encoding="utf-8").splitlines()
        ]
        if len(metadata) != len(rows):
            raise ValueError(f"metadata length mismatch for {data_path}")
        rows = [
            dict(row, **meta)
            for row, meta in zip(rows, metadata, strict=True)
        ]
    return rows


def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
) -> tuple[list[str], float]:
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    encoded.pop("token_type_ids", None)
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - started
    prompt_width = encoded["input_ids"].shape[1]
    texts = tokenizer.batch_decode(
        generated[:, prompt_width:],
        skip_special_tokens=True,
    )
    return [text.strip() for text in texts], elapsed


def update_semantic_counts(
    counts: Counter[str],
    prefix: str,
    actual: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    for field in FIELDS:
        if actual[field] == expected[field]:
            counts[f"{prefix}:field:{field}"] += 1
    if actual == expected:
        counts[f"{prefix}:contract_exact"] += 1
    expected_names = {item["name"] for item in expected["target_properties"]}
    actual_names = {item["name"] for item in actual["target_properties"]}
    if actual_names == expected_names:
        counts[f"{prefix}:property_name_set_exact"] += 1
    for name in expected_names:
        counts[f"{prefix}:property_expected:{name}"] += 1
        if name in actual_names:
            counts[f"{prefix}:property_recalled:{name}"] += 1


def summarize(
    counts: Counter[str],
    samples: int,
    batch_seconds: list[float],
    wall_seconds: float,
) -> dict[str, Any]:
    def layer(prefix: str) -> dict[str, Any]:
        property_recall = {}
        for name in (
            "band_gap",
            "curie_temperature",
            "formation_energy",
            "magnetic_moment_density",
        ):
            expected = counts[f"{prefix}:property_expected:{name}"]
            recalled = counts[f"{prefix}:property_recalled:{name}"]
            property_recall[name] = {
                "expected": expected,
                "recalled": recalled,
                "recall": recalled / expected if expected else None,
            }
        return {
            "contract_exact_rate": counts[f"{prefix}:contract_exact"] / samples,
            "field_accuracy": {
                field: counts[f"{prefix}:field:{field}"] / samples
                for field in FIELDS
            },
            "property_name_set_exact_rate": (
                counts[f"{prefix}:property_name_set_exact"] / samples
            ),
            "property_recall": property_recall,
        }

    return {
        "samples": samples,
        "json_valid_rate": counts["json_valid"] / samples,
        "schema_valid_rate": counts["schema_valid"] / samples,
        "raw_model": layer("raw"),
        "current_pipeline": layer("final"),
        "generation_wall_seconds": round(wall_seconds, 3),
        "mean_batch_seconds": round(statistics.mean(batch_seconds), 3),
    }


def evaluate_split(
    model,
    tokenizer,
    data_path: Path,
    batch_size: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    rows = load_rows(data_path)
    counts: Counter[str] = Counter()
    results: list[dict[str, Any]] = []
    batch_seconds: list[float] = []
    wall_started = time.perf_counter()

    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                row["conversations"][:2],
                tokenize=False,
                add_generation_prompt=True,
                open_thinking=False,
            )
            for row in batch
        ]
        outputs, elapsed = generate_batch(
            model,
            tokenizer,
            prompts,
            max_new_tokens,
        )
        batch_seconds.append(elapsed)

        for offset, (row, raw_output) in enumerate(
            zip(batch, outputs, strict=True)
        ):
            index = start + offset
            model_text = row["conversations"][1]["content"]
            original_text = row.get("original_user", model_text)
            expected_payload = json.loads(
                row["conversations"][2]["content"]
            )
            expected_payload["original_text"] = original_text
            expected_request = MaterialRequest.model_validate(expected_payload)
            expected = comparable(expected_request)
            raw_actual = None
            final_actual = None
            error = None
            warnings: list[str] = []
            try:
                payload = extract_json_object(raw_output)
                counts["json_valid"] += 1
                payload["original_text"] = original_text
                payload["parser_warnings"] = []
                raw_request = MaterialRequest.model_validate(payload)
                counts["schema_valid"] += 1
                raw_actual = comparable(raw_request)
                update_semantic_counts(
                    counts,
                    "raw",
                    raw_actual,
                    expected,
                )
                final_request = current_pipeline_finalize(
                    raw_request,
                    original_text,
                )
                final_actual = comparable(final_request)
                warnings = list(final_request.parser_warnings)
                update_semantic_counts(
                    counts,
                    "final",
                    final_actual,
                    expected,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

            results.append(
                {
                    "index": index,
                    "case_family": row.get("case_family"),
                    "input_mode": row.get("input_mode"),
                    "original_input": original_text,
                    "model_input": model_text,
                    "expected": expected,
                    "raw_actual": raw_actual,
                    "current_pipeline_actual": final_actual,
                    "pipeline_warnings": warnings,
                    "error": error,
                    "raw_output": raw_output,
                }
            )
        print(
            f"{data_path.stem}: evaluated={min(start + batch_size, len(rows))}"
            f"/{len(rows)} schema_valid={counts['schema_valid']}"
            f" raw_exact={counts['raw:contract_exact']}",
            flush=True,
        )

    summary = summarize(
        counts,
        len(rows),
        batch_seconds,
        time.perf_counter() - wall_started,
    )
    return {"summary": summary, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--splits",
        default="id_hidden,ood_hidden,adversarial_hidden",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluation")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
    ).eval().to("cuda")

    split_names = [
        item.strip() for item in args.splits.split(",") if item.strip()
    ]
    report: dict[str, Any] = {
        "model": str(args.model),
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "splits": {},
    }
    for split in split_names:
        report["splits"][split] = evaluate_split(
            model,
            tokenizer,
            args.data_dir / f"{split}.jsonl",
            args.batch_size,
            args.max_new_tokens,
        )
        print(
            json.dumps(
                {
                    split: report["splits"][split]["summary"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"report={args.output}", flush=True)


if __name__ == "__main__":
    main()
