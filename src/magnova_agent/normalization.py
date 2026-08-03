from __future__ import annotations

import re
from dataclasses import dataclass


PROPERTY_ALIASES = {
    "magnetic_moment_density": (
        "磁矩密度",
        "磁化密度",
        "单位体积磁矩",
        "体积磁矩",
        "单位体积的磁矩",
        "magnetic moment density",
        "体积磁化强度指标",
        "每单位体积磁矩",
    ),
    "curie_temperature": (
        "居里温度",
        "磁性相变温度",
        "铁磁转变温度",
        "curie temperature",
        "居里点",
        "铁磁有序温度",
        "tc",
    ),
    "formation_energy": (
        "形成能",
        "生成能",
        "formation energy",
        "formation-energy",
        "每原子形成能",
        "热力学生成能",
        "单原子形成能",
    ),
    "band_gap": (
        "带隙",
        "禁带宽度",
        "band gap",
        "band-gap",
        "能带间隙",
        "电子能隙",
        "禁带间隔",
    ),
}

PROPERTY_DISPLAY = {
    "magnetic_moment_density": "磁矩密度",
    "curie_temperature": "居里温度",
    "formation_energy": "形成能",
    "band_gap": "带隙",
}

UNIT_ALIASES = {
    "μB/Å³": (
        "μB / Å³",
        "μB·Å⁻³",
        "μB·Å^-3",
        "μB/Å3",
        "uB/A^3",
        "玻尔磁子/立方埃",
    ),
    "eV/atom": (
        "eV per atom",
        "eV 每原子",
        "eV atom^-1",
        "eV·atom⁻¹",
        "每原子 eV",
        "eV/原子",
    ),
    "eV": ("电子伏特", "electron volt", "electron-volt"),
    "K": ("开尔文", "kelvin"),
}

FREE_CONSTRAINT_ALIASES = {
    "室温稳定": (
        "常温下不能失稳",
        "在日常温度下保持稳定",
        "室温条件下结构不发生破坏",
    ),
    "低成本": (
        "预算友好",
        "制造费用要低",
        "原料与制备成本受控",
    ),
    "无毒": ("不能有毒", "避免毒性风险", "不得具有毒性"),
    "可合成": (
        "应当具有合成可行性",
        "实验制备可实现",
        "现有实验条件能够制备",
    ),
}

RARE_EARTH_ALIASES = (
    "不依赖稀土成分",
    "组成里不能出现稀土",
    "排除所有稀土",
    "避免使用任何稀土",
    "稀土元素不得参与组成",
    "要求完全无稀土",
    "稀土含量必须为零",
    "不允许含任何稀土元素",
    "采用零稀土设计",
)

RARE_EARTH_PATTERN = re.compile(
    "|".join(
        re.escape(alias)
        for alias in sorted(
            RARE_EARTH_ALIASES + ("不含稀土元素", "无稀土"),
            key=len,
            reverse=True,
        )
    )
    + r"|(?:不含|不得含|不能含|不得包含|不能包含|不能引入|排除|避免(?:使用)?|无|不依赖)"
    + r"\s*(?:任何)?\s*稀土"
    + r"|稀土(?:族|元素|原子|组分|成分)?(?:成员)?(?:含量|占比)?"
    + r"\s*(?:必须)?\s*为零"
)


@dataclass(frozen=True)
class PropertyMention:
    canonical_name: str
    raw_name: str
    start: int
    end: int


@dataclass(frozen=True)
class NormalizationResult:
    original_text: str
    model_text: str
    property_mentions: tuple[PropertyMention, ...]
    trace: tuple[str, ...]

    def raw_name_for(self, canonical_name: str) -> str | None:
        for mention in self.property_mentions:
            if mention.canonical_name == canonical_name:
                return mention.raw_name
        return None


def _replace_aliases(
    text: str,
    aliases: dict[str, tuple[str, ...]],
    replacement,
    trace_prefix: str,
) -> tuple[str, list[str]]:
    alias_to_canonical = {
        alias.casefold(): canonical
        for canonical, values in aliases.items()
        for alias in values
        if alias.casefold() != canonical.casefold()
    }
    if not alias_to_canonical:
        return text, []
    pattern = re.compile(
        "|".join(
            re.escape(alias)
            for alias in sorted(alias_to_canonical, key=len, reverse=True)
        ),
        re.IGNORECASE,
    )
    trace: list[str] = []

    def substitute(match: re.Match[str]) -> str:
        raw = match.group(0)
        canonical = alias_to_canonical[raw.casefold()]
        value = replacement(canonical)
        trace.append(f"{trace_prefix}:{raw}->{canonical}")
        return value

    return pattern.sub(substitute, text), trace


def normalize_for_llm(text: str) -> NormalizationResult:
    mentions: list[PropertyMention] = []
    candidates: list[tuple[int, int, str, str]] = []
    for canonical, aliases in PROPERTY_ALIASES.items():
        for alias in sorted(aliases, key=len, reverse=True):
            for match in re.finditer(re.escape(alias), text, re.IGNORECASE):
                candidates.append((match.start(), match.end(), canonical, match.group(0)))
    candidates.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    occupied_until = -1
    for start, end, canonical, raw in candidates:
        if start < occupied_until:
            continue
        mentions.append(PropertyMention(canonical, raw, start, end))
        occupied_until = end

    property_aliases = {
        canonical: tuple(
            alias
            for alias in aliases
            if alias != f"{PROPERTY_DISPLAY[canonical]}<{canonical}>"
        )
        for canonical, aliases in PROPERTY_ALIASES.items()
    }
    model_text, trace = _replace_aliases(
        text,
        property_aliases,
        lambda canonical: f"{PROPERTY_DISPLAY[canonical]}<{canonical}>",
        "property",
    )
    model_text, unit_trace = _replace_aliases(
        model_text, UNIT_ALIASES, lambda canonical: canonical, "unit"
    )
    trace.extend(unit_trace)
    model_text, free_trace = _replace_aliases(
        model_text,
        FREE_CONSTRAINT_ALIASES,
        lambda canonical: canonical,
        "constraint",
    )
    trace.extend(free_trace)

    rare_pattern = re.compile(
        "|".join(
            re.escape(alias)
            for alias in sorted(RARE_EARTH_ALIASES, key=len, reverse=True)
        )
    )
    if rare_pattern.search(model_text):
        model_text = rare_pattern.sub("不含稀土元素", model_text)
        trace.append("composition:rare_earth_exclusion->不含稀土元素")

    return NormalizationResult(
        original_text=text,
        model_text=model_text,
        property_mentions=tuple(mentions),
        trace=tuple(dict.fromkeys(trace)),
    )


def has_rare_earth_exclusion(text: str) -> bool:
    return bool(RARE_EARTH_PATTERN.search(text))


def explicit_task_hint(text: str) -> str | None:
    lowered = text.casefold()
    no_candidate_patterns = (
        r"(?:没有|不存在)[^，；。]{0,12}(?:候选|备选|现成结果|可筛(?:选)?条目)",
        r"(?:候选|备选|筛选池|候选列表)[^，；。]{0,10}(?:为空|为零)",
    )
    existing_candidate_phrases = (
        "生成步骤已经结束",
        "生成步骤已经完成",
        "生成工作已经完成",
        "设计阶段已经结束",
        "不要重新生成",
        "不再生成",
        "候选已备好",
        "候选已经生成",
        "现有候选",
        "已有候选",
        "已有一批候选",
        "备选集合",
        "备选材料",
        "当前清单",
        "候选清单",
        "材料列表",
        "材料清单",
        "已有计算结果",
        "已有结果",
        "现有结果",
        "当前材料结果",
        "现有条目",
        "已有集合",
    )
    creation_phrases = (
        "从零",
        "全新材料",
        "全新方案",
        "创造一种材料",
        "提出新组成",
        "寻找新的组成",
        "开展设计",
        "进行材料设计",
        "研发",
        "开发",
    )
    if any(re.search(pattern, lowered) for pattern in no_candidate_patterns):
        return "generate_material"
    if any(phrase in lowered for phrase in existing_candidate_phrases):
        return "screen_candidates"
    if any(phrase in lowered for phrase in creation_phrases):
        return "generate_material"
    return None
