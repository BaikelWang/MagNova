from __future__ import annotations

from .semantic_property_linker import EmbeddingBackend, SemanticPropertyLinker


TASK_PROTOTYPES: dict[str, tuple[str, ...]] = {
    "generate_material": (
        "当前没有候选结构，需要从零开始设计、生成或提出一种新的材料。",
        "empty candidate pool; create and generate a new material composition from scratch",
    ),
    "screen_candidates": (
        "已经存在一批计算完成的候选材料，需要筛选、保留合格项或剔除不合格项。",
        "existing candidate structures are ready; filter, screen, retain or reject them",
    ),
}


class SemanticTaskLinker(SemanticPropertyLinker):
    def __init__(
        self,
        embeddings: EmbeddingBackend,
        *,
        min_score: float = 0.45,
        min_margin: float = 0.03,
    ) -> None:
        super().__init__(
            embeddings,
            prototypes=TASK_PROTOTYPES,
            min_score=min_score,
            min_margin=min_margin,
        )
