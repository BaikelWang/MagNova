from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Protocol, Sequence


class EmbeddingBackend(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


PROPERTY_PROTOTYPES: dict[str, tuple[str, ...]] = {
    "magnetic_moment_density": (
        "磁矩密度是晶体总净磁矩除以晶体体积，描述单位体积承载的磁矩或体积归一化自旋磁矩。",
        "magnetic moment density, net magnetic moment per unit crystal volume",
    ),
    "curie_temperature": (
        "居里温度是铁磁长程有序或自发磁化消失、铁磁态转为顺磁态的临界温度。",
        "Curie temperature, thermal critical point where ferromagnetic order disappears",
    ),
    "formation_energy": (
        "每原子形成能是目标晶体相相对于组成元素稳定参考态的平均原子能量差。",
        "formation energy per atom relative to elemental reference states and phase stability",
    ),
    "band_gap": (
        "带隙是价带顶与导带底、最高占据态与最低未占据态之间没有允许电子态的能量间隔。",
        "electronic band gap between valence-band maximum and conduction-band minimum",
    ),
}


@dataclass(frozen=True)
class SemanticPropertyMatch:
    canonical_name: str | None
    score: float
    runner_up_score: float
    margin: float
    accepted: bool
    reason: str
    text: str


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions differ")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def optimization_clauses(text: str) -> list[str]:
    """Return clauses that explicitly request maximizing or minimizing a property."""
    markers = (
        "越大越好",
        "越高越好",
        "最大化",
        "推到更大",
        "优先增大",
        "越小越好",
        "越低越好",
        "最小化",
        "压到更小",
        "优先减小",
    )
    clauses = [item.strip() for item in re.split(r"[；。\n]", text) if item.strip()]
    return [clause for clause in clauses if any(marker in clause for marker in markers)]


class SemanticPropertyLinker:
    """Map an unseen scientific paraphrase to a canonical property with abstention.

    The linker is deliberately narrow: it only classifies property meaning. Numeric
    values, units, operators, elements and task routing remain deterministic slots.
    """

    def __init__(
        self,
        embeddings: EmbeddingBackend,
        *,
        prototypes: dict[str, tuple[str, ...]] | None = None,
        min_score: float = 0.45,
        min_margin: float = 0.03,
    ) -> None:
        self.embeddings = embeddings
        self.prototypes = prototypes or PROPERTY_PROTOTYPES
        self.min_score = min_score
        self.min_margin = min_margin
        self._prototype_names: list[str] = []
        prototype_texts: list[str] = []
        for name, descriptions in self.prototypes.items():
            for description in descriptions:
                self._prototype_names.append(name)
                prototype_texts.append(description)
        vectors = embeddings.embed_documents(prototype_texts)
        if len(vectors) != len(prototype_texts):
            raise RuntimeError("embedding backend returned an unexpected vector count")
        self._prototype_vectors = vectors

    def link(self, text: str) -> SemanticPropertyMatch:
        query = self.embeddings.embed_query(text)
        property_scores = {name: -1.0 for name in self.prototypes}
        for name, vector in zip(self._prototype_names, self._prototype_vectors):
            property_scores[name] = max(
                property_scores[name], cosine_similarity(query, vector)
            )
        ranked = sorted(property_scores.items(), key=lambda item: item[1], reverse=True)
        best_name, best_score = ranked[0]
        runner_up_score = ranked[1][1]
        margin = best_score - runner_up_score
        accepted = best_score >= self.min_score and margin >= self.min_margin
        if best_score < self.min_score:
            reason = "below_score_threshold"
        elif margin < self.min_margin:
            reason = "ambiguous_margin"
        else:
            reason = "accepted"
        return SemanticPropertyMatch(
            canonical_name=best_name if accepted else None,
            score=best_score,
            runner_up_score=runner_up_score,
            margin=margin,
            accepted=accepted,
            reason=reason,
            text=text,
        )
