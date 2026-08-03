from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any


CASES: tuple[dict[str, Any], ...] = (
    {"name": "strict_gt", "text": "当前没有候选，请设计带隙严格大于 1.5 eV 的材料", "task": "generate_material", "properties": {"band_gap": {"operator": "gt", "value": 1.5, "unit": "eV"}}},
    {"name": "inclusive_lte_negative", "text": "已有候选，只保留形成能小于等于 -0.5 eV/atom 的条目", "task": "screen_candidates", "properties": {"formation_energy": {"operator": "lte", "value": -0.5, "unit": "eV/atom"}}},
    {"name": "exact_eq", "text": "目前没有方案，请寻找居里温度恰好为 600 K 的材料", "task": "generate_material", "properties": {"curie_temperature": {"operator": "eq", "value": 600.0, "unit": "K"}}},
    {"name": "two_property_binding", "text": "没有现成结果，请研发磁矩密度高于 0.2 μB/Å³且居里温度至少 500 K 的材料", "task": "generate_material", "properties": {"magnetic_moment_density": {"operator": "gt", "value": 0.2, "unit": "μB/Å³"}, "curie_temperature": {"operator": "gte", "value": 500.0, "unit": "K"}}},
    {"name": "two_ranges", "text": "候选已生成，只筛选形成能在 -1.2 到 -0.6 eV/atom、带隙在 1.0 到 2.0 eV 的材料，两个区间都含端点", "task": "screen_candidates", "properties": {"formation_energy": {"operator": "range", "lower_bound": -1.2, "upper_bound": -0.6, "unit": "eV/atom"}, "band_gap": {"operator": "range", "lower_bound": 1.0, "upper_bound": 2.0, "unit": "eV"}}},
    {"name": "optimization_pair", "text": "已有候选，不设置数值阈值，优先让居里温度最大、形成能最小", "task": "screen_candidates", "properties": {"curie_temperature": {"operator": "maximize"}, "formation_energy": {"operator": "minimize"}}},
    {"name": "ood_aliases", "text": "从零设计价带上沿至导带下沿的禁带宽度超过 1.1 eV，并且晶体磁矩的体积归一值不低于 0.15 玻尔磁子每立方埃的材料", "task": "generate_material", "properties": {"band_gap": {"operator": "gt", "value": 1.1, "unit": "eV"}, "magnetic_moment_density": {"operator": "gte", "value": 0.15, "unit": "μB/Å³"}}},
    {"name": "task_conflict_generate", "text": "虽然这些是筛选线，但当前候选池为空，请生成带隙至少 1.3 eV 的新材料", "task": "generate_material", "properties": {"band_gap": {"operator": "gte", "value": 1.3, "unit": "eV"}}},
    {"name": "task_conflict_screen", "text": "设计指标已经给出且候选计算完成，不要创造新结构，只保留居里温度高于 550 K 的条目", "task": "screen_candidates", "properties": {"curie_temperature": {"operator": "gt", "value": 550.0, "unit": "K"}}},
    {"name": "missing_unit", "text": "没有候选，请设计居里温度至少 600 的材料", "task": "generate_material", "properties": {"curie_temperature": {"operator": "gte", "value": 600.0, "unit": None}}},
    {"name": "composition", "text": "请从零设计必须含 Fe 和 Co、排除 Ni 且不含稀土的材料", "task": "generate_material", "required": ["Fe", "Co"], "excluded": ["Ni"], "exclude_rare_earth": True},
    {"name": "negative_gt", "text": "已有候选，只保留形成能严格大于 -0.5 eV/atom 的材料", "task": "screen_candidates", "properties": {"formation_energy": {"operator": "gt", "value": -0.5, "unit": "eV/atom"}}},
)


def extract_json(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object")
    return json.loads(text[start : end + 1])


def request_model(endpoint: str, system_prompt: str, text: str) -> tuple[dict[str, Any], str]:
    body = json.dumps({"model": "minimind", "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}], "temperature": 0.01, "max_tokens": 512, "stream": True}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(endpoint, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=90) as response:
        response_text = response.read().decode("utf-8")
    if response_text.lstrip().startswith("data:"):
        parts: list[str] = []
        for line in response_text.splitlines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            event = json.loads(payload)
            if "error" in event:
                raise RuntimeError(f"API stream error: {event['error']}")
            choice = event["choices"][0]
            content_piece = choice.get("delta", {}).get("content")
            if content_piece is None:
                content_piece = choice.get("message", {}).get("content", "")
            parts.append(content_piece or "")
        content = "".join(parts)
    else:
        envelope = json.loads(response_text)
        content = envelope["choices"][0]["message"]["content"]
    return extract_json(content), content


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8505/v1/chat/completions")
    parser.add_argument("--manifest", type=Path, default=Path("data/sft_v52b/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/v52b_model_smoke.json"))
    args = parser.parse_args()
    system_prompt = json.loads(args.manifest.read_text(encoding="utf-8"))["system_prompt"]
    results = []
    passed = 0
    for case in CASES:
        errors: list[str] = []
        try:
            actual, raw = request_model(args.endpoint, system_prompt, case["text"])
            if actual.get("task") != case["task"]:
                errors.append(f"task expected={case['task']} actual={actual.get('task')}")
            props = {item["name"]: item for item in actual.get("target_properties", [])}
            for name, expected in case.get("properties", {}).items():
                if name not in props:
                    errors.append(f"missing_property:{name}")
                    continue
                for field, value in expected.items():
                    if props[name].get(field) != value:
                        errors.append(f"{name}.{field} expected={value!r} actual={props[name].get(field)!r}")
            composition = actual.get("composition", {})
            if "required" in case and set(composition.get("required_elements", [])) != set(case["required"]):
                errors.append("required_elements_mismatch")
            if "excluded" in case and set(composition.get("excluded_elements", [])) != set(case["excluded"]):
                errors.append("excluded_elements_mismatch")
            if "exclude_rare_earth" in case and composition.get("exclude_rare_earth") != case["exclude_rare_earth"]:
                errors.append("exclude_rare_earth_mismatch")
        except Exception as exc:
            actual = None
            raw = None
            errors.append(f"{type(exc).__name__}: {exc}")
        if not errors:
            passed += 1
        results.append({"name": case["name"], "input": case["text"], "passed": not errors, "errors": errors, "actual": actual, "raw": raw})
        print(case["name"], "PASS" if not errors else f"FAIL {errors}", flush=True)
    report = {"endpoint": args.endpoint, "passed": passed, "total": len(CASES), "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": passed, "total": len(CASES)}))
    raise SystemExit(0 if passed == len(CASES) else 1)


if __name__ == "__main__":
    main()
