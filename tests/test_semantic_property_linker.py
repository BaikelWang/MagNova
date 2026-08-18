from __future__ import annotations

from magnova_agent.semantic_property_linker import (
    SemanticPropertyLinker,
    cosine_similarity,
    optimization_clauses,
)


class FakeEmbeddings:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.vectors[text] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.vectors[text]


def test_cosine_similarity_handles_normalized_and_zero_vectors() -> None:
    assert cosine_similarity([1.0, 0.0], [2.0, 0.0]) == 1.0
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_optimization_clauses_only_keeps_directional_requests() -> None:
    text = "候选已就绪；价带与导带间隔越大越好；还要求低成本。"
    assert optimization_clauses(text) == ["价带与导带间隔越大越好"]


def test_linker_accepts_clear_match() -> None:
    prototypes = {"band_gap": ("gap",), "curie_temperature": ("curie",)}
    embeddings = FakeEmbeddings(
        {"gap": [1.0, 0.0], "curie": [0.0, 1.0], "query": [0.9, 0.1]}
    )
    result = SemanticPropertyLinker(
        embeddings, prototypes=prototypes, min_score=0.8, min_margin=0.2
    ).link("query")
    assert result.accepted is True
    assert result.canonical_name == "band_gap"


def test_linker_abstains_when_top_two_are_ambiguous() -> None:
    prototypes = {"band_gap": ("gap",), "curie_temperature": ("curie",)}
    embeddings = FakeEmbeddings(
        {"gap": [1.0, 0.0], "curie": [0.99, 0.01], "query": [1.0, 0.0]}
    )
    result = SemanticPropertyLinker(
        embeddings, prototypes=prototypes, min_score=0.5, min_margin=0.05
    ).link("query")
    assert result.accepted is False
    assert result.canonical_name is None
    assert result.reason == "ambiguous_margin"
