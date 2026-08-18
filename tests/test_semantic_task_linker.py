from __future__ import annotations

from magnova_agent.semantic_task_linker import TASK_PROTOTYPES, SemanticTaskLinker


class FakeEmbeddings:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.vectors[text] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.vectors[text]


def test_task_linker_accepts_clear_generation_request() -> None:
    texts = [text for values in TASK_PROTOTYPES.values() for text in values]
    vectors = {
        texts[0]: [1.0, 0.0],
        texts[1]: [1.0, 0.0],
        texts[2]: [0.0, 1.0],
        texts[3]: [0.0, 1.0],
        "new": [0.9, 0.1],
    }
    result = SemanticTaskLinker(
        FakeEmbeddings(vectors), min_score=0.8, min_margin=0.2
    ).link("new")
    assert result.accepted is True
    assert result.canonical_name == "generate_material"
