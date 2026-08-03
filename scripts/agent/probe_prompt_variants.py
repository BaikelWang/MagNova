from __future__ import annotations

import json
import time

import requests

from magnova_agent.interpreter import SYSTEM_PROMPT, extract_json_object


ENDPOINT = "http://127.0.0.1:8503/v1/chat/completions"
USER_TEXT = "设计带隙至少 1.5 eV 的材料"

COMPACT_PROMPT = """你是信息抽取器。只输出JSON，禁止解释。
字段：task,material_family,target_properties,composition,free_text_constraints,original_text,parser_warnings。
target_properties元素字段：name,operator,value,lower_bound,upper_bound,unit,raw_name。
composition字段：required_elements,excluded_elements,exclude_rare_earth。
缺失值用null或[]；task默认generate_material；至少=g​​te。"""

ONE_SHOT_PROMPT = """只输出一行合法JSON，不解释。
示例输入：生成居里温度高于500 K且不含稀土的材料
示例输出：{"task":"generate_material","material_family":null,"target_properties":[{"name":"curie_temperature","operator":"gt","value":500,"lower_bound":null,"upper_bound":null,"unit":"K","raw_name":"居里温度"}],"composition":{"required_elements":[],"excluded_elements":[],"exclude_rare_earth":true},"free_text_constraints":[],"original_text":"生成居里温度高于500 K且不含稀土的材料","parser_warnings":[]}
现在按相同字段结构转换用户输入。"""

VARIANTS = [
    ("full_temp_07", SYSTEM_PROMPT, 0.7),
    ("compact_temp_07", COMPACT_PROMPT, 0.7),
    ("one_shot_temp_03", ONE_SHOT_PROMPT, 0.3),
]


for name, system_prompt, temperature in VARIANTS:
    started = time.perf_counter()
    response = requests.post(
        ENDPOINT,
        json={
            "model": "minimind",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": USER_TEXT},
            ],
            "temperature": temperature,
            "top_p": 0.8,
            "max_tokens": 384,
            "stream": False,
            "open_thinking": False,
        },
        timeout=90,
    )
    response.raise_for_status()
    output = response.json()["choices"][0]["message"]["content"]
    try:
        payload = extract_json_object(output)
        json_valid = True
        keys = sorted(payload)
    except Exception:
        json_valid = False
        keys = []
    print(
        json.dumps(
            {
                "variant": name,
                "latency_seconds": round(time.perf_counter() - started, 3),
                "json_valid": json_valid,
                "keys": keys,
                "output": output,
            },
            ensure_ascii=False,
        )
    )
