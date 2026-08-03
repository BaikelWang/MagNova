from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from magnova_agent.models import (
    Comparator,
    CompositionConstraint,
    MaterialRequest,
    PropertyConstraint,
    TaskType,
)
from magnova_agent.normalization import normalize_for_llm


SYSTEM_PROMPT = """你是 MagNova 材料需求解析器。用户输入既可能是原始自然语言，也可能含有 <canonical_name> 规范标记；没有标记时也必须依据完整语义识别需求。把需求转换成一个合法 JSON 对象，不解释、不使用 Markdown。
只输出 task、material_family、target_properties、composition、free_text_constraints。
task 只能是 generate_material、screen_candidates 或 predict_properties。判断 task 时看当前候选状态和用户最终动作，不要只看某个关键词：当前没有候选、需要提出新组成或从零设计时用 generate_material；已经有候选、列表、计算结果或材料池，并且只要求保留、淘汰、过滤或核验时用 screen_candidates。
target_properties 每项只能包含 name、operator、value、lower_bound、upper_bound、unit，并按 name 字母顺序排列；不要输出 raw_name。operator 只能是 gt、gte、lt、lte、eq、range、maximize、minimize。
规范性质名只有 magnetic_moment_density、curie_temperature、formation_energy、band_gap。不能猜测用户未给出的数值或单位；缺失值用 null，缺失列表用 []。
composition 只能包含 required_elements、excluded_elements、exclude_rare_earth。
free_text_constraints 只保留规范短语：室温稳定、低成本、无毒、可合成。
不要输出 original_text、parser_warnings、raw_name 或任何其他字段。"""


PROPERTY_CONFIG: dict[str, dict[str, Any]] = {
    "magnetic_moment_density": {
        "unit": "μB/Å³",
        "values": (0.12, 0.18, 0.2, 0.25, 0.3),
        "ranges": ((0.15, 0.25), (0.2, 0.35)),
        "operators": ("gt", "gte", "eq", "range", "maximize"),
    },
    "curie_temperature": {
        "unit": "K",
        "values": (300.0, 450.0, 500.0, 600.0, 750.0),
        "ranges": ((300.0, 500.0), (450.0, 700.0)),
        "operators": ("gt", "gte", "eq", "range", "maximize"),
    },
    "formation_energy": {
        "unit": "eV/atom",
        "values": (-1.2, -0.8, -0.5, -0.3),
        "ranges": ((-1.2, -0.6), (-0.8, -0.3)),
        "operators": ("lt", "lte", "eq", "range", "minimize"),
    },
    "band_gap": {
        "unit": "eV",
        "values": (0.8, 1.0, 1.5, 2.0, 2.5),
        "ranges": ((0.8, 1.5), (1.5, 2.5)),
        "operators": ("gt", "gte", "lte", "eq", "range"),
    },
}


# known: 当前规范化器能够识别；train_ood: V4 已消费的 OOD 词和新增训练同义词；
# dev_ood / hidden_ood: 分别只在开发集 / 全新隐藏集中出现。
PROPERTY_ALIASES = {
    "magnetic_moment_density": {
        "known": (
            "磁矩密度",
            "磁化密度",
            "单位体积磁矩",
            "体积磁矩",
            "单位体积的磁矩",
            "magnetic moment density",
            "体积磁化强度指标",
            "每单位体积磁矩",
        ),
        "train_ood": (
            "磁性体积强度",
            "单位体积磁响应",
            "体积归一化磁矩",
            "单位体积磁化矩",
        ),
        "dev_ood": (
            "体积磁偶极矩密度",
            "每立方体积的磁矩",
        ),
        "hidden_ood": (
            "单位体积所承载的磁偶极矩",
            "体积归一化总磁矩",
        ),
    },
    "curie_temperature": {
        "known": (
            "居里温度",
            "磁性相变温度",
            "Tc",
            "铁磁转变温度",
            "Curie temperature",
            "居里点",
            "铁磁有序温度",
        ),
        "train_ood": (
            "磁有序临界温度",
            "铁磁序消失温度",
            "磁序转变临界温度",
            "铁磁性消失温度",
        ),
        "dev_ood": (
            "铁磁到顺磁的临界温度",
            "磁化强度消失临界点",
        ),
        "hidden_ood": (
            "铁磁—顺磁转变温度",
            "自发磁化消失温度",
        ),
    },
    "formation_energy": {
        "known": (
            "形成能",
            "生成能",
            "formation energy",
            "formation-energy",
            "每原子形成能",
            "热力学生成能",
            "单原子形成能",
        ),
        "train_ood": (
            "化合稳定能",
            "组分稳定能",
            "每原子生成能",
            "结构形成能",
        ),
        "dev_ood": (
            "相对单质的生成能量",
            "化合物形成稳定能",
        ),
        "hidden_ood": (
            "相对于元素基态的每原子能量差",
            "晶体生成稳定度",
        ),
    },
    "band_gap": {
        "known": (
            "带隙",
            "禁带宽度",
            "band gap",
            "band-gap",
            "能带间隙",
            "电子能隙",
            "禁带间隔",
        ),
        "train_ood": (
            "价导带分离能",
            "导带价带间隔量",
            "价带导带能量差",
            "电子禁带宽度",
        ),
        "dev_ood": (
            "价带顶到导带底的能差",
            "电子跃迁能量间隔",
        ),
        "hidden_ood": (
            "最高价带与最低导带能差",
            "电子激发能隙",
        ),
    },
}


UNIT_TEXT = {
    "magnetic_moment_density": {
        "known": ("μB/Å³", "μB / Å³", "μB·Å⁻³", "uB/A^3", "玻尔磁子/立方埃"),
        "train_ood": ("Bohr-magneton per cubic angstrom", "玻尔磁子每立方埃"),
        "dev_ood": ("Bohr magnetons per cubic angstrom",),
        "hidden_ood": ("每立方埃的玻尔磁子数", "μ_B·angstrom^{-3}"),
    },
    "curie_temperature": {
        "known": ("K", "开尔文", "kelvin", "Kelvin"),
        "train_ood": ("absolute-degree", "开氏绝对温标"),
        "dev_ood": ("absolute kelvin",),
        "hidden_ood": ("开尔文绝对温度单位",),
    },
    "formation_energy": {
        "known": ("eV/atom", "eV per atom", "eV 每原子", "每原子 eV", "eV/原子"),
        "train_ood": ("electronvolt per constituent atom", "电子伏特每原子"),
        "dev_ood": ("electronvolts for each atom",),
        "hidden_ood": ("按原子计的电子伏特",),
    },
    "band_gap": {
        "known": ("eV", "电子伏特", "electron volt", "electron-volt"),
        "train_ood": ("electronvolt", "电子伏特能量单位"),
        "dev_ood": ("electronvolts",),
        "hidden_ood": ("以电子伏特计",),
    },
}


COMPARATOR_TEXT = {
    "train": {
        "gt": ("大于", "高于", "超过", "必须超出"),
        "gte": ("至少", "不低于", "大于等于", "下限为"),
        "lt": ("小于", "低于", "严格少于"),
        "lte": ("不超过", "至多", "小于等于", "上限为"),
        "eq": ("约为", "等于", "接近", "目标值为"),
        "maximize": ("尽可能高", "越高越好", "需要最大化"),
        "minimize": ("尽可能低", "越低越好", "需要最小化"),
    },
    "dev": {
        "gt": ("必须高过且不能取等号",),
        "gte": ("达到这个下限即可",),
        "lt": ("必须低过且不能取等号",),
        "lte": ("不高于给定上界",),
        "eq": ("希望精确达到",),
        "maximize": ("优先选择更高数值",),
        "minimize": ("优先选择更低数值",),
    },
    "id_hidden": {
        "gt": ("严格高于", "一定要超过"),
        "gte": ("不得小于", "起码达到"),
        "lt": ("严格低于",),
        "lte": ("不得超过", "最高只能是"),
        "eq": ("以此数值为目标",),
        "maximize": ("数值越大越理想",),
        "minimize": ("数值越小越理想",),
    },
    "ood_hidden": {
        "gt": ("必须超过但不能等于",),
        "gte": ("允许等于下限且不能更低",),
        "lt": ("必须低于且不取等号",),
        "lte": ("可等于上限但不能更高",),
        "eq": ("精确目标设为",),
        "maximize": ("在可行范围内数值越大越优",),
        "minimize": ("在可行范围内数值越小越优",),
    },
    "adversarial_hidden": {
        "gt": ("只接受严格超过",),
        "gte": ("等于门槛也算合格，不能更低",),
        "lt": ("只接受严格低于",),
        "lte": ("等于上限也算合格，不能更高",),
        "eq": ("数值需要正好是",),
        "maximize": ("其他条件满足时优先最大",),
        "minimize": ("其他条件满足时优先最小",),
    },
}


RANGE_TEXT = {
    "train": (
        "{alias}介于 {lower} 和 {upper}",
        "{alias}位于 {lower} 到 {upper} 之间",
    ),
    "dev": ("{alias}处在从 {lower} 至 {upper} 的范围",),
    "id_hidden": ("{alias}落在 {lower}—{upper} 区间",),
    "ood_hidden": ("{alias}需要处于包含端点的 {lower} 至 {upper} 区间",),
    "adversarial_hidden": ("{alias}闭区间限定为 [{lower}, {upper}]",),
}


WRAPPERS = {
    "train": {
        "generate_material": (
            "请设计满足{body}的新材料",
            "帮我生成一种{body}的材料",
            "从零开发符合{body}的材料",
            "当前没有候选，请按{body}研发新方案",
            "围绕{body}进行新材料设计",
        ),
        "screen_candidates": (
            "按照{body}给候选材料做初筛",
            "已有一批候选，只留下符合{body}的",
            "从候选池中剔除不满足{body}的材料",
            "对当前材料列表执行过滤，要求{body}",
            "候选已经生成，按{body}二次筛选",
        ),
    },
    "train_contrast": {
        "generate_material": (
            "当前候选列表为空，虽然已经给出筛选指标，但请从零设计满足{body}的新材料",
            "现在没有任何候选可供筛选，请生成满足{body}的材料方案",
        ),
        "screen_candidates": (
            "生成步骤已经完成，不要设计新结构，只从现有候选中保留满足{body}的项",
            "虽然目标里提到材料设计，但目前已有候选，仅按{body}筛选",
        ),
    },
    "dev": {
        "generate_material": (
            "尚无备选结果，请依据{body}构思新的组成",
            "请提出一个此前没有的材料方案，使它满足{body}",
        ),
        "screen_candidates": (
            "现成备选材料已经准备好，请按{body}做条件审查",
            "不增加新方案，只对已有材料按{body}进行复筛",
        ),
    },
    "dev_contrast": {
        "generate_material": (
            "筛选池目前是空的，因此不是过滤任务；请新建满足{body}的材料",
        ),
        "screen_candidates": (
            "设计阶段已经结束，即使下文出现“生成”，也只核验现有清单是否满足{body}",
        ),
    },
    "id_hidden": {
        "generate_material": (
            "我们手里还没有候选，请探索一种满足{body}的新材料",
            "请从头提出可满足{body}的材料组成",
        ),
        "screen_candidates": (
            "候选结果已经在表中，请按{body}选出合格项",
            "只检查现有材料池，把不符合{body}的条目排除",
        ),
    },
    "ood_hidden": {
        "generate_material": (
            "目前不存在可用备选，请围绕{body}提出新的化学组成",
            "需要从无到有寻找满足{body}的材料方案",
        ),
        "screen_candidates": (
            "备选集合已经计算完毕，请依据{body}做准入判定",
            "只在当前结果集合中按{body}淘汰不合格项",
        ),
    },
    "adversarial_hidden": {
        "generate_material": (
            "候选池目前为空，无法执行筛选；请从零生成满足{body}的新材料",
            "现在没有现成结果，所谓筛选条件就是设计目标，请研发满足{body}的方案",
        ),
        "screen_candidates": (
            "生成步骤已经结束，下面不是新材料设计任务，只在现成清单中按{body}淘汰",
            "已有候选可用，不要重新生成；即使提到“设计指标”，也只保留满足{body}的项",
        ),
    },
}


RARE_EARTH = {
    "known": (
        "不含稀土元素",
        "排除所有稀土",
        "避免使用任何稀土",
        "要求完全无稀土",
    ),
    "train_ood": (
        "镧系及相关稀土成分为零",
        "材料不借助镧系组分",
        "不得采用任何稀土组分",
    ),
    "dev_ood": (
        "组成中稀土元素占比必须为零",
        "配方完全避开镧系和其他稀土",
    ),
    "hidden_ood": (
        "元素组成不得包含稀土族成员",
        "配方中不能引入任何稀土原子",
    ),
}


FREE_TEXT = {
    "known": {
        "室温稳定": ("室温稳定", "常温下不能失稳"),
        "低成本": ("低成本", "预算友好"),
        "无毒": ("无毒", "不能有毒"),
        "可合成": ("可合成", "应当具有合成可行性"),
    },
    "train_ood": {
        "室温稳定": ("环境温度下相结构保持", "日常温度中结构可靠"),
        "低成本": ("经济性要好", "原料和制备费用不能高"),
        "无毒": ("生物安全风险低", "不得带来毒性危害"),
        "可合成": ("实验路线现实可行", "现有工艺应能制备"),
    },
    "dev_ood": {
        "室温稳定": ("常规环境温度下维持结构完整",),
        "低成本": ("总制造开销需要受控",),
        "无毒": ("避免对人体产生毒性",),
        "可合成": ("实验室具备实现制备的可能",),
    },
    "hidden_ood": {
        "室温稳定": ("二十五摄氏度附近不发生结构失稳",),
        "低成本": ("原料采购与加工费用均需低廉",),
        "无毒": ("不能包含会造成毒害的成分",),
        "可合成": ("应能通过现实实验路线制得",),
    },
}


FAMILIES = ("Fe-Co", "Fe-Ni", "Mn-Al", "Co-B", "Fe-P")
ELEMENTS = ("Fe", "Co", "Ni", "Mn", "Al", "B", "C", "P", "Pb")


def number(value: float) -> str:
    return f"{value:g}"


def profile_for(tier: str, index: int) -> dict[str, str]:
    if tier == "train":
        position = index % 20
        if position < 7:
            return {
                "input_mode": "normalized",
                "alias_group": "known",
                "wrapper_group": "train",
                "case_family": "marked_known",
                "language_group": "train",
            }
        if position < 12:
            return {
                "input_mode": "raw",
                "alias_group": "known",
                "wrapper_group": "train",
                "case_family": "raw_known",
                "language_group": "train",
            }
        if position < 17:
            return {
                "input_mode": "raw",
                "alias_group": "train_ood",
                "wrapper_group": "train",
                "case_family": "raw_paraphrase",
                "language_group": "train",
            }
        return {
            "input_mode": "raw",
            "alias_group": "train_ood",
            "wrapper_group": "train_contrast",
            "case_family": "task_contrast",
            "language_group": "train",
        }
    if tier == "dev":
        profiles = (
            ("normalized", "known", "dev", "marked_known"),
            ("raw", "known", "dev", "raw_known"),
            ("raw", "dev_ood", "dev", "raw_paraphrase"),
            ("raw", "dev_ood", "dev_contrast", "task_contrast"),
        )
        input_mode, alias_group, wrapper_group, case_family = profiles[index % 4]
        return {
            "input_mode": input_mode,
            "alias_group": alias_group,
            "wrapper_group": wrapper_group,
            "case_family": case_family,
            "language_group": "dev",
        }
    if tier == "id_hidden":
        return {
            "input_mode": "normalized" if index % 2 == 0 else "raw",
            "alias_group": "known",
            "wrapper_group": "id_hidden",
            "case_family": "marked_known" if index % 2 == 0 else "raw_known",
            "language_group": "id_hidden",
        }
    if tier == "ood_hidden":
        return {
            "input_mode": "raw",
            "alias_group": "hidden_ood",
            "wrapper_group": "ood_hidden",
            "case_family": "raw_hidden_paraphrase",
            "language_group": "ood_hidden",
        }
    if tier == "adversarial_hidden":
        return {
            "input_mode": "raw",
            "alias_group": "hidden_ood",
            "wrapper_group": "adversarial_hidden",
            "case_family": "task_contrast_hidden",
            "language_group": "adversarial_hidden",
        }
    raise ValueError(f"unknown tier: {tier}")


def property_phrase(
    rng: random.Random,
    name: str,
    alias_group: str,
    language_group: str,
    omit_unit: bool,
) -> tuple[str, PropertyConstraint, str]:
    config = PROPERTY_CONFIG[name]
    alias = rng.choice(PROPERTY_ALIASES[name][alias_group])
    operator = rng.choice(config["operators"])
    unit_text = None if omit_unit else rng.choice(UNIT_TEXT[name][alias_group])
    unit = None if omit_unit else config["unit"]

    if operator == "range":
        lower, upper = rng.choice(config["ranges"])
        phrase = rng.choice(RANGE_TEXT[language_group]).format(
            alias=alias,
            lower=number(lower),
            upper=number(upper),
        )
        if unit_text:
            phrase += f" {unit_text}"
        prop = PropertyConstraint(
            name=name,
            raw_name=alias,
            operator=Comparator.RANGE,
            lower_bound=lower,
            upper_bound=upper,
            unit=unit,
        )
    elif operator in ("maximize", "minimize"):
        phrase = f"{alias}{rng.choice(COMPARATOR_TEXT[language_group][operator])}"
        prop = PropertyConstraint(
            name=name,
            raw_name=alias,
            operator=Comparator(operator),
        )
    else:
        value = rng.choice(config["values"])
        phrase = (
            f"{alias}{rng.choice(COMPARATOR_TEXT[language_group][operator])} "
            f"{number(value)}"
        )
        if unit_text:
            phrase += f" {unit_text}"
        prop = PropertyConstraint(
            name=name,
            raw_name=alias,
            operator=Comparator(operator),
            value=value,
            unit=unit,
        )
    return phrase, prop, alias


def build_case(
    rng: random.Random,
    tier: str,
    index: int,
) -> tuple[str, MaterialRequest, dict[str, Any]]:
    profile = profile_for(tier, index)
    task = TaskType.GENERATE if index % 2 == 0 else TaskType.SCREEN
    allow_empty = tier not in {"ood_hidden", "adversarial_hidden"}
    missing_property = allow_empty and index % 37 == 0
    missing_unit = not missing_property and index % 11 == 0
    property_count = 0 if missing_property else (2 if rng.random() < 0.62 else 1)

    phrases: list[str] = []
    properties: list[PropertyConstraint] = []
    aliases: dict[str, str] = {}
    if property_count:
        names = rng.sample(tuple(PROPERTY_CONFIG), k=property_count)
        for property_index, name in enumerate(names):
            phrase, prop, alias = property_phrase(
                rng,
                name,
                profile["alias_group"],
                profile["language_group"],
                omit_unit=missing_unit and property_index == 0,
            )
            phrases.append(phrase)
            properties.append(prop)
            aliases[name] = alias
    properties.sort(key=lambda item: item.name)

    family = rng.choice(FAMILIES) if rng.random() < 0.2 else None
    required: list[str] = []
    excluded: list[str] = []
    exclude_rare_earth = rng.random() < 0.42
    free_constraints: list[str] = []

    if family:
        phrases.append(f"材料体系限定为 {family}")
    if rng.random() < 0.2:
        required = rng.sample(ELEMENTS[:-1], k=2)
        phrases.append(f"组成必须同时含有 {' 和 '.join(required)}")
    if rng.random() < 0.2:
        available = [item for item in ELEMENTS if item not in required]
        excluded = [rng.choice(available)]
        phrases.append(f"配方中排除元素 {excluded[0]}")
    if exclude_rare_earth:
        phrases.append(rng.choice(RARE_EARTH[profile["alias_group"]]))
    if rng.random() < 0.4:
        canonical = rng.choice(tuple(FREE_TEXT[profile["alias_group"]]))
        free_constraints.append(canonical)
        phrases.append(rng.choice(FREE_TEXT[profile["alias_group"]][canonical]))

    if not phrases:
        phrases.append("适用于后续材料研究")
    rng.shuffle(phrases)
    separator = rng.choice(("，并且", "；同时", "、", "，还要"))
    body = separator.join(phrases)
    original = rng.choice(WRAPPERS[profile["wrapper_group"]][task.value]).format(
        body=body
    )
    request = MaterialRequest(
        task=task,
        material_family=family,
        target_properties=properties,
        composition=CompositionConstraint(
            required_elements=required,
            excluded_elements=excluded,
            exclude_rare_earth=exclude_rare_earth,
        ),
        free_text_constraints=free_constraints,
        original_text=original,
        parser_warnings=[],
    )
    metadata = {
        **profile,
        "task": task.value,
        "property_aliases": aliases,
        "operators": [prop.operator.value for prop in properties],
    }
    return original, request, metadata


def semantic_payload(request: MaterialRequest) -> dict[str, Any]:
    payload = request.model_dump(
        mode="json",
        exclude={"original_text", "parser_warnings"},
    )
    for prop in payload["target_properties"]:
        prop.pop("raw_name", None)
    return payload


def write_split(
    output_dir: Path,
    tier: str,
    count: int,
    seed: int,
    used_originals: set[str],
    used_model_inputs: set[str],
) -> dict[str, Any]:
    rng = random.Random(seed)
    rows: list[str] = []
    metadata_rows: list[str] = []
    local_originals: set[str] = set()
    local_model_inputs: set[str] = set()
    stats: Counter[str] = Counter()
    attempts = 0

    while len(rows) < count:
        attempts += 1
        if attempts > count * 500:
            raise RuntimeError(f"unable to generate {count} unique rows for {tier}")
        index = len(rows)
        original, request, metadata = build_case(rng, tier, index)
        normalized = normalize_for_llm(original)
        model_input = (
            normalized.model_text
            if metadata["input_mode"] == "normalized"
            else original
        )
        if (
            original in local_originals
            or original in used_originals
            or model_input in local_model_inputs
            or model_input in used_model_inputs
        ):
            continue

        payload = semantic_payload(request)
        validation_payload = dict(payload)
        validation_payload["original_text"] = original
        validated = MaterialRequest.model_validate(validation_payload)
        assistant = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        conversation = {
            "conversations": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": model_input},
                {"role": "assistant", "content": assistant},
            ]
        }
        metadata.update(
            {
                "evaluation_tier": tier,
                "original_user": original,
                "normalization_trace": list(normalized.trace),
                "model_input_sha256": hashlib.sha256(
                    model_input.encode()
                ).hexdigest(),
                "contains_property_marker": any(
                    f"<{name}>" in model_input for name in PROPERTY_CONFIG
                ),
            }
        )
        rows.append(
            json.dumps(conversation, ensure_ascii=False, separators=(",", ":"))
        )
        metadata_rows.append(
            json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
        )
        local_originals.add(original)
        local_model_inputs.add(model_input)

        stats[f"task:{validated.task.value}"] += 1
        stats[f"input_mode:{metadata['input_mode']}"] += 1
        stats[f"case_family:{metadata['case_family']}"] += 1
        stats[f"alias_group:{metadata['alias_group']}"] += 1
        stats[f"properties:{len(validated.target_properties)}"] += 1
        stats[
            f"status:{'ready' if validated.ready_for_submission else 'clarify'}"
        ] += 1
        stats[
            f"property_marker:{metadata['contains_property_marker']}"
        ] += 1
        for prop in validated.target_properties:
            stats[f"property:{prop.name}"] += 1
            stats[f"operator:{prop.operator.value}"] += 1

    used_originals.update(local_originals)
    used_model_inputs.update(local_model_inputs)
    content = "\n".join(rows) + "\n"
    metadata_content = "\n".join(metadata_rows) + "\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / f"{tier}.jsonl"
    metadata_path = output_dir / f"{tier}.meta.jsonl"
    data_path.write_text(content, encoding="utf-8")
    metadata_path.write_text(metadata_content, encoding="utf-8")
    return {
        "count": len(rows),
        "sha256": hashlib.sha256(content.encode()).hexdigest(),
        "metadata_sha256": hashlib.sha256(
            metadata_content.encode()
        ).hexdigest(),
        "stats": dict(sorted(stats.items())),
    }


def validate_alias_partitions() -> None:
    for property_name, groups in PROPERTY_ALIASES.items():
        seen: set[str] = set()
        for group, aliases in groups.items():
            overlap = seen.intersection(alias.casefold() for alias in aliases)
            if overlap:
                raise ValueError(
                    f"property alias leakage for {property_name}/{group}: {overlap}"
                )
            seen.update(alias.casefold() for alias in aliases)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/sft_v5"))
    parser.add_argument("--train-size", type=int, default=8000)
    parser.add_argument("--dev-size", type=int, default=800)
    parser.add_argument("--id-hidden-size", type=int, default=600)
    parser.add_argument("--ood-hidden-size", type=int, default=400)
    parser.add_argument("--adversarial-hidden-size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()

    validate_alias_partitions()
    used_originals: set[str] = set()
    used_model_inputs: set[str] = set()
    # 隐藏集必须先生成、先冻结；随后 dev/train 只能避开它们。
    specs = (
        ("id_hidden", args.id_hidden_size),
        ("ood_hidden", args.ood_hidden_size),
        ("adversarial_hidden", args.adversarial_hidden_size),
        ("dev", args.dev_size),
        ("train", args.train_size),
    )
    manifest: dict[str, Any] = {
        "schema_version": "material_request_semantic_fields_v5",
        "seed": args.seed,
        "system_prompt": SYSTEM_PROMPT,
        "system_prompt_sha256": hashlib.sha256(
            SYSTEM_PROMPT.encode()
        ).hexdigest(),
        "model_generates_raw_name": False,
        "program_restores_raw_name": True,
        "frozen_before_training": [
            "id_hidden",
            "ood_hidden",
            "adversarial_hidden",
        ],
        "generation_order": [name for name, _ in specs],
        "previous_v4_tests_used_for_v5_training_decisions": True,
        "previous_v4_tests_are_not_v5_hidden_tests": True,
        "task_scope": ["generate_material", "screen_candidates"],
        "predict_properties_note": (
            "Not trained in V5 because the current request schema cannot identify "
            "a concrete existing material without overloading other fields."
        ),
        "split_policy": (
            "Hidden tiers are generated first. Exact original text and exact model "
            "input are globally disjoint. Train/dev/hidden OOD property aliases are "
            "partitioned. Training mixes normalized marked inputs, raw known inputs, "
            "raw paraphrases, and task-contrast examples."
        ),
        "splits": {},
    }
    for offset, (tier, count) in enumerate(specs):
        manifest["splits"][tier] = write_split(
            args.output_dir,
            tier,
            count,
            args.seed + offset,
            used_originals,
            used_model_inputs,
        )
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
