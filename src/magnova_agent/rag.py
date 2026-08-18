from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import requests
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


class DeterministicHashEmbeddings(Embeddings):
    """Small, dependency-free baseline embedding for teaching and smoke tests.

    It hashes character 1/2/3-grams into a fixed-size vector. This captures lexical
    overlap, not deep semantics, so it is intentionally replaceable by a learned
    Chinese embedding model after the RAG pipeline is verified.
    """

    def __init__(self, dimensions: int = 2048) -> None:
        if dimensions < 128:
            raise ValueError("dimensions must be at least 128")
        self.dimensions = dimensions

    @staticmethod
    def _normalise(text: str) -> str:
        cleaned = text.lower()
        for phrase in ("是什么", "为什么", "怎么样", "如何", "怎样", "请问"):
            cleaned = cleaned.replace(phrase, "")
        return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", cleaned)

    def _embed(self, text: str) -> list[float]:
        cleaned = self._normalise(text)
        vector = [0.0] * self.dimensions
        if not cleaned:
            return vector

        features: list[tuple[str, float]] = []
        for width, weight in ((2, 1.0), (3, 1.25), (4, 1.5)):
            features.extend(
                (cleaned[index : index + width], weight)
                for index in range(max(0, len(cleaned) - width + 1))
            )
        if not features:
            features.append((cleaned, 1.0))
        for feature, weight in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            bucket = value % self.dimensions
            sign = 1.0 if value & 1 else -1.0
            vector[bucket] += sign * weight

        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class HTTPBGEEmbeddings(Embeddings):
    """Lightweight LangChain adapter for a separately deployed BGE service."""

    def __init__(
        self,
        endpoint: str,
        *,
        model: str = "BAAI/bge-small-zh-v1.5",
        timeout_seconds: float = 60.0,
        query_instruction: str = "为这个句子生成表示以用于检索相关文章：",
        batch_size: int = 32,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.endpoint = endpoint
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.query_instruction = query_instruction
        self.batch_size = batch_size

    def _embed(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for offset in range(0, len(texts), self.batch_size):
            batch = texts[offset : offset + self.batch_size]
            response = requests.post(
                self.endpoint,
                json={"model": self.model, "input": batch},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = sorted(response.json()["data"], key=lambda item: item["index"])
            if len(data) != len(batch):
                raise RuntimeError(
                    "embedding service returned "
                    f"{len(data)} vectors for a batch of {len(batch)} texts"
                )
            embeddings.extend(item["embedding"] for item in data)
        return embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([self.query_instruction + text])[0]


class MaterialScienceQueryGuard:
    """Transparent scope gate for the material-science prototype corpus."""

    _domain_terms = re.compile(
        r"(?:mattergen|cgcnn|mpnn|magnova|minimind|sacgt|crystal|material|atom|"
        r"lattice|band.?gap|magnetic|curie|perovskite|diffusion|adapter|"
        r"superconduct|defect|hamiltonian|phase.?transition|resistiv|pressure|"
        r"mattersim|hamgnn|cri3|lah10|lk-?99|cu2s|"
        r"classifier-free|wrapped\s+normal|晶体|材料|原子|晶格|元素|磁性|"
        r"磁矩|稀磁|铁磁|磁化|自旋|半导体|氧化物|异质结|单层|纳米管|"
        r"斯格明子|带隙|形成能|居里温度|钙钛矿|扩散|周期边界|图神经网络|"
        r"消息传递|化学环境|超导|缺陷|哈密顿量|相变|电阻|压力|温度)",
        re.IGNORECASE,
    )

    def is_in_scope(self, query: str) -> bool:
        return bool(self._domain_terms.search(query))


class GuardedEvidenceRetriever:
    """Skip vector retrieval when a query is clearly outside project scope."""

    def __init__(self, retriever, guard: MaterialScienceQueryGuard | None = None):
        self.retriever = retriever
        self.guard = guard or MaterialScienceQueryGuard()
        self.last_scope_decision = True

    def retrieve(self, query: str, k: int = 4) -> list[dict]:
        self.last_scope_decision = self.guard.is_in_scope(query)
        if not self.last_scope_decision:
            return []
        return self.retriever.retrieve(query, k=k)


class SentenceEvidencePacker:
    """Rerank sentences inside retrieved chunks and pack a compact context."""

    def __init__(
        self,
        retriever,
        embeddings: Embeddings,
        *,
        candidate_k: int = 32,
        max_sentences: int = 8,
        max_total_chars: int = 1800,
        max_sentence_chars: int = 320,
        max_sentences_per_source: int = 8,
    ) -> None:
        if min(candidate_k, max_sentences, max_total_chars, max_sentence_chars) < 1:
            raise ValueError("sentence packing limits must be positive")
        self.retriever = retriever
        self.embeddings = embeddings
        self.candidate_k = candidate_k
        self.max_sentences = max_sentences
        self.max_total_chars = max_total_chars
        self.max_sentence_chars = max_sentence_chars
        self.max_sentences_per_source = max_sentences_per_source
        self.last_candidates: list[dict] = []

    def _split_sentences(self, text: str) -> list[str]:
        text = re.sub(r"\A---.*?---", " ", text, flags=re.DOTALL)
        text = re.sub(r"!\[\[.*?\]\]", " ", text)
        pieces = re.split(r"(?<=[。！？!?；;])\s*|[\r\n]+", text)
        sentences: list[str] = []
        for piece in pieces:
            cleaned = re.sub(r"\s+", " ", piece).strip(" -+#*\t")
            if len(cleaned) < 10:
                continue
            if len(cleaned) <= self.max_sentence_chars:
                sentences.append(cleaned)
                continue
            start = 0
            while start < len(cleaned):
                end = min(start + self.max_sentence_chars, len(cleaned))
                sentences.append(cleaned[start:end])
                if end == len(cleaned):
                    break
                start = max(start + 1, end - 40)
        return sentences

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        numerator = sum(a * b for a, b in zip(left, right, strict=True))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0

    def retrieve(self, query: str, k: int = 4) -> list[dict]:
        chunks = self.retriever.retrieve(query, k=max(k, self.candidate_k))
        ranking_query = getattr(self.retriever, "last_expanded_query", "") or query
        candidates: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for chunk in chunks:
            sentences = self._split_sentences(chunk["content"])
            for sentence_index, sentence in enumerate(sentences):
                key = (str(chunk.get("source", "unknown")), sentence.casefold())
                if key in seen:
                    continue
                seen.add(key)
                # Keep the focused sentence first, then its following and previous
                # neighbours. PDF extraction often breaks one semantic sentence at
                # a visual line boundary; a small window restores the missing half.
                neighbours = [sentence]
                if sentence_index + 1 < len(sentences):
                    neighbours.append(sentences[sentence_index + 1])
                if sentence_index + 2 < len(sentences):
                    neighbours.append(sentences[sentence_index + 2])
                if sentence_index > 0:
                    neighbours.append(sentences[sentence_index - 1])
                candidates.append(
                    {
                        **{name: value for name, value in chunk.items() if name != "content"},
                        "content": " ".join(neighbours),
                        "focus_sentence": sentence,
                        "sentence_index": sentence_index,
                        "parent_score": float(chunk.get("score", 0.0)),
                    }
                )
        if not candidates:
            self.last_candidates = []
            return []

        query_vector = self.embeddings.embed_query(ranking_query)
        sentence_vectors = self.embeddings.embed_documents(
            [item["content"] for item in candidates]
        )
        for item, vector in zip(candidates, sentence_vectors, strict=True):
            semantic_score = self._cosine(query_vector, vector)
            item["sentence_score"] = semantic_score
            query_anchors = EvidenceAnswerabilityGate._anchors(ranking_query)
            normalised_content = EvidenceAnswerabilityGate._normalise(item["content"])
            anchor_matches = sum(
                EvidenceAnswerabilityGate._normalise(anchor) in normalised_content
                for anchor in query_anchors
            )
            numeric_matches = sum(
                EvidenceAnswerabilityGate._normalise(constraint) in normalised_content
                for constraint in EvidenceAnswerabilityGate._numeric_constraint.findall(
                    ranking_query
                )
            )
            item["packing_score"] = (
                semantic_score
                + 0.08 * item["parent_score"]
                + (0.04 if item.get("source_hint_match") else 0.0)
                + 0.08 * anchor_matches
                + 0.16 * numeric_matches
            )
        candidates.sort(key=lambda item: item["packing_score"], reverse=True)
        self.last_candidates = candidates

        packed: list[dict] = []
        source_counts: dict[str, int] = {}
        total_chars = 0
        for item in candidates:
            source = str(item.get("source", "unknown"))
            if source_counts.get(source, 0) >= self.max_sentences_per_source:
                continue
            if packed and total_chars + len(item["content"]) > self.max_total_chars:
                continue
            packed.append(item)
            source_counts[source] = source_counts.get(source, 0) + 1
            total_chars += len(item["content"])
            if len(packed) == self.max_sentences:
                break
        return packed


class EvidenceFaithfulnessGate:
    """Conservative deterministic proxy for whether claims are supported."""

    _citation_pattern = re.compile(
        r"\[([^\]\s/\\]+\.(?:md|txt|pdf))(?:\s+p\.\d+)?\]",
        re.IGNORECASE,
    )
    _refusal_markers = (
        "无法根据现有证据",
        "证据不足",
        "没有足够证据",
        "不能确定",
        "无法确定",
    )
    _uncited_refusal_templates = {
        "根据给定检索证据无法确定这个问题的答案",
        "根据给定检索证据无法确定答案",
    }
    _number_pattern = re.compile(r"\d+(?:[.,]\d+)*(?:%|Å|eV|K)?", re.IGNORECASE)

    def __init__(self, *, min_bigram_coverage: float = 0.28) -> None:
        if not 0 <= min_bigram_coverage <= 1:
            raise ValueError("min_bigram_coverage must be in [0, 1]")
        self.min_bigram_coverage = min_bigram_coverage

    @staticmethod
    def _normalise(text: str) -> str:
        return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text.casefold())

    @classmethod
    def _bigrams(cls, text: str) -> set[str]:
        cleaned = cls._normalise(text)
        return {cleaned[index : index + 2] for index in range(max(0, len(cleaned) - 1))}

    def check(self, answer: str, evidence: list[dict]) -> list[str]:
        reasons: list[str] = []
        citations = list(dict.fromkeys(self._citation_pattern.findall(answer)))
        evidence_sources = {str(item.get("source", "unknown")) for item in evidence}
        answer_without_citations = self._citation_pattern.sub("", answer)
        if not citations:
            normalized_answer = re.sub(r"[\s，。！？、；：,.!?;:]", "", answer_without_citations)
            if normalized_answer in self._uncited_refusal_templates:
                return reasons
            return ["missing_evidence_citation"]
        if any(source not in evidence_sources for source in citations):
            reasons.append("unknown_evidence_citation")

        cited_text = "\n".join(
            str(item.get("content", ""))
            for item in evidence
            if item.get("source") in citations
        )
        if any(marker in answer_without_citations for marker in self._refusal_markers):
            return reasons

        unsupported_numbers = [
            number
            for number in self._number_pattern.findall(answer_without_citations)
            if self._normalise(number) not in self._normalise(cited_text)
        ]
        if unsupported_numbers:
            reasons.append("unsupported_number")

        claim_bigrams = self._bigrams(answer_without_citations)
        evidence_bigrams = self._bigrams(cited_text)
        coverage = (
            len(claim_bigrams.intersection(evidence_bigrams)) / len(claim_bigrams)
            if claim_bigrams
            else 0.0
        )
        if coverage < self.min_bigram_coverage:
            reasons.append("low_lexical_grounding")
        return reasons


class EvidenceAnswerabilityGate:
    """Independent pre-generation gate based on retrieval and evidence support.

    Dense similarity answers "what looks related". This gate separately checks
    whether distinctive entities and numeric constraints in the question are
    actually present in the retrieved text before MiniMind is allowed to answer.
    """

    _ascii_token = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*")
    _numeric_constraint = re.compile(
        r"\b\d+(?:\.\d+)?\s*(?:K|GPa|MPa|T|eV|meV|%|Å|nm)\b",
        re.IGNORECASE,
    )
    _common_tokens = {
        "what", "which", "where", "when", "with", "from", "into", "about",
        "model", "material", "materials", "exact", "reported", "paper",
    }

    def __init__(self, *, min_top_score: float = 0.48) -> None:
        if not 0 <= min_top_score <= 1:
            raise ValueError("min_top_score must be in [0, 1]")
        self.min_top_score = min_top_score

    @classmethod
    def _anchors(cls, query: str) -> list[str]:
        anchors = []
        for token in cls._ascii_token.findall(query):
            lowered = token.casefold()
            if len(token) >= 3 and lowered not in cls._common_tokens:
                anchors.append(token)
        return list(dict.fromkeys(anchors))

    @staticmethod
    def _normalise(text: str) -> str:
        return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text.casefold())

    def assess(self, query: str, evidence: list[dict]) -> dict:
        reasons: list[str] = []
        if not evidence:
            return {"can_answer": False, "reasons": ["no_evidence"]}

        scores = [
            float(item["score"])
            for item in evidence
            if item.get("score") is not None
        ]
        top_score = max(scores) if scores else None
        if top_score is not None and top_score < self.min_top_score:
            reasons.append("low_top_score")

        evidence_text = self._normalise(
            "\n".join(str(item.get("content", "")) for item in evidence)
        )
        for anchor in self._anchors(query):
            if self._normalise(anchor) not in evidence_text:
                reasons.append(f"missing_anchor:{anchor}")

        for constraint in self._numeric_constraint.findall(query):
            if self._normalise(constraint) not in evidence_text:
                reasons.append(f"missing_numeric_constraint:{constraint}")

        return {"can_answer": not reasons, "reasons": reasons, "top_score": top_score}


class LangChainPaperRetriever:
    """Load local papers/notes, split them, and expose source-aware retrieval."""

    SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}
    DEFAULT_QUERY_EXPANSIONS = {
        "mattergen": "无机晶体逆向设计 联合扩散 原子类型 原子坐标 晶格 前向加噪 反向去噪",
        "cgcnn": "晶体图卷积神经网络 原子邻居 图表示 材料性质预测",
        "mpnn": "消息传递神经网络 message passing 节点消息 聚合 更新",
        "minimind": "MagNova Agent 科研讨论 检索证据 需求理解 Schema 澄清 用户确认 职责边界",
        "局部化学环境": "CGCNN 晶体图卷积 原子连接 位点贡献 可解释性",
    }
    DEFAULT_SOURCE_HINTS = {
        "mattergen": "mattergen.md",
        "cgcnn": "cgcnn.md",
        "mpnn": "mpnn.md",
        "magnova": "magnova-plan.md",
        "minimind": "magnova-plan.md",
    }

    def __init__(
        self,
        documents: Sequence[Document],
        *,
        embeddings: Embeddings | None = None,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        score_threshold: float = 0.10,
        query_expansions: Mapping[str, str] | None = None,
        source_hints: Mapping[str, str] | None = None,
        min_candidate_pool: int = 0,
    ) -> None:
        if not documents:
            raise ValueError("at least one document is required")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must be in [0, chunk_size)")
        self.embeddings = embeddings or DeterministicHashEmbeddings()
        self.score_threshold = score_threshold
        self.query_expansions = dict(
            query_expansions or self.DEFAULT_QUERY_EXPANSIONS
        )
        self.source_hints = dict(source_hints or self.DEFAULT_SOURCE_HINTS)
        self.min_candidate_pool = max(0, int(min_candidate_pool))
        self.last_expanded_query = ""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n## ", "\n### ", "\n\n", "\n", "。", "；", "，", " ", ""],
        )
        self.chunks = splitter.split_documents(list(documents))
        for index, chunk in enumerate(self.chunks):
            chunk.metadata["chunk_id"] = f"chunk-{index:05d}"
        self.vector_store = InMemoryVectorStore(embedding=self.embeddings)
        self.vector_store.add_documents(self.chunks)

    @staticmethod
    def _load_path(path: Path) -> list[Document]:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            documents = [
                Document(
                    page_content=page.extract_text() or "",
                    metadata={"page": page_number},
                )
                for page_number, page in enumerate(PdfReader(str(path)).pages)
            ]
        elif suffix in {".md", ".txt"}:
            documents = [
                Document(page_content=path.read_text(encoding="utf-8"), metadata={})
            ]
        else:
            raise ValueError(f"unsupported document type: {path.suffix}")

        for document in documents:
            document.metadata["source"] = str(path)
            document.metadata["source_name"] = path.name
        return documents

    @classmethod
    def from_paths(cls, paths: Iterable[str | Path], **kwargs) -> "LangChainPaperRetriever":
        documents: list[Document] = []
        for raw_path in paths:
            path = Path(raw_path)
            documents.extend(cls._load_path(path))
        return cls(documents, **kwargs)

    @classmethod
    def from_directory(cls, directory: str | Path, **kwargs) -> "LangChainPaperRetriever":
        root = Path(directory)
        alias_path = root / "retrieval-aliases.json"
        if alias_path.exists():
            import json

            alias_config = json.loads(alias_path.read_text(encoding="utf-8"))
            merged_expansions = dict(cls.DEFAULT_QUERY_EXPANSIONS)
            merged_expansions.update(alias_config.get("query_expansions", {}))
            merged_expansions.update(kwargs.pop("query_expansions", {}) or {})
            kwargs["query_expansions"] = merged_expansions
            merged_hints = dict(cls.DEFAULT_SOURCE_HINTS)
            merged_hints.update(alias_config.get("source_hints", {}))
            merged_hints.update(kwargs.pop("source_hints", {}) or {})
            kwargs["source_hints"] = merged_hints
            kwargs.setdefault(
                "min_candidate_pool", int(alias_config.get("min_candidate_pool", 0))
            )
        paths = sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in cls.SUPPORTED_SUFFIXES
        )
        if not paths:
            raise ValueError(f"no supported documents found in {root}")
        return cls.from_paths(paths, **kwargs)

    def retrieve(self, query: str, k: int = 4) -> list[dict]:
        if not query.strip():
            return []
        expanded_query = query
        lowered = query.lower()
        for keyword, expansion in self.query_expansions.items():
            if keyword.lower() in lowered:
                expanded_query += " " + expansion
        self.last_expanded_query = expanded_query
        # Retrieve a wider dense candidate pool before metadata re-ranking. A
        # relevant paper can otherwise be hidden by several chunks from a long
        # but only loosely related project document.
        candidate_k = max(k, k * 3, self.min_candidate_pool)
        results = self.vector_store.similarity_search_with_score(
            expanded_query, k=candidate_k
        )
        evidence: list[dict] = []
        for dense_rank, (document, score) in enumerate(results, start=1):
            if score < self.score_threshold:
                continue
            item = {
                "content": document.page_content.strip(),
                "source": document.metadata.get("source_name")
                or Path(str(document.metadata.get("source", "unknown"))).name,
                "chunk_id": document.metadata.get("chunk_id", ""),
                "score": float(score),
                "dense_rank": dense_rank,
            }
            if "page" in document.metadata:
                item["page"] = int(document.metadata["page"]) + 1
            evidence.append(item)
        expanded_lowered = expanded_query.lower()
        hint_match_counts: dict[str, int] = {}
        for keyword, source in self.source_hints.items():
            if keyword.lower() in expanded_lowered:
                hint_match_counts[source] = hint_match_counts.get(source, 0) + 1
        if hint_match_counts:
            for item in evidence:
                item["source_hint_matches"] = hint_match_counts.get(
                    item["source"], 0
                )
                item["source_hint_match"] = bool(item["source_hint_matches"])
            evidence.sort(
                key=lambda item: (
                    -item["source_hint_matches"],
                    item["dense_rank"],
                )
            )
        return evidence[:k]


HYBRID_CONTROL_DOCUMENTS = {"paper-catalog.md"}
MATERIAL_BILINGUAL_QUERY_EXPANSIONS = {
    "稀磁半导体": "dilute diluted magnetic semiconductor DMS",
    "异质结": "heterojunction",
    "铁磁性": "ferromagnetism",
    "次近邻": "next-nearest-neighbor",
    "跃迁": "hopping",
    "单层": "monolayer",
    "掺杂": "doped dopant",
    "居里温度": "Curie temperature",
    "自旋电子": "spintronics",
}


def expand_material_query(
    query: str, expansions: Mapping[str, str] | None = None
) -> str:
    glossary = expansions or MATERIAL_BILINGUAL_QUERY_EXPANSIONS
    additions = [expansion for term, expansion in glossary.items() if term in query]
    return " ".join([query, *additions])


def hybrid_corpus_paths(root: str | Path) -> list[Path]:
    directory = Path(root)
    return sorted(
        (
            path
            for path in directory.rglob("*")
            if path.is_file()
            and path.suffix.lower() in LangChainPaperRetriever.SUPPORTED_SUFFIXES
            and path.name not in HYBRID_CONTROL_DOCUMENTS
        ),
        key=lambda item: str(item).casefold(),
    )


def load_corpus_metadata_documents(root: str | Path) -> list[Document]:
    directory = Path(root)
    records: list[dict] = []
    v6_catalog = directory / "v6-document-catalog.json"
    if v6_catalog.exists():
        records.extend(json.loads(v6_catalog.read_text(encoding="utf-8")))

    legacy_catalog = directory / "paper-catalog.md"
    if legacy_catalog.exists():
        title: str | None = None
        for line in legacy_catalog.read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                title = line[3:].strip()
            elif title and line.startswith("- 检索来源：`") and line.endswith("`"):
                records.append(
                    {
                        "title": title,
                        "source": line[len("- 检索来源：`") : -1],
                        "subject": None,
                    }
                )
                title = None

    documents: list[Document] = []
    seen_sources: set[str] = set()
    for record in records:
        source = str(record.get("source") or "").strip()
        title = str(record.get("title") or "").strip()
        if not source or not title or source in seen_sources:
            continue
        seen_sources.add(source)
        subject = str(record.get("subject") or "未标注").strip()
        documents.append(
            Document(
                page_content=(
                    f"论文标题 / Paper title: {title}\n"
                    f"学科主题 / Subject: {subject}"
                ),
                metadata={
                    "source": str(directory / source),
                    "source_name": source,
                    "document_type": "verified_metadata",
                },
            )
        )
    return documents


class HybridSourceDiversifiedRetriever:
    """Hybrid reranking and source diversification over a dense retriever."""

    _latin_anchor = re.compile(r"[A-Za-z][A-Za-z0-9+_.-]{2,}")

    def __init__(
        self,
        base: LangChainPaperRetriever,
        *,
        max_chunks_per_source: int = 1,
        min_candidate_chunks: int = 128,
        candidate_multiplier: int = 16,
        bilingual_expansions: Mapping[str, str] | None = None,
    ) -> None:
        if max_chunks_per_source < 1:
            raise ValueError("max_chunks_per_source must be positive")
        if min_candidate_chunks < 1 or candidate_multiplier < 1:
            raise ValueError("candidate pool settings must be positive")
        self.base = base
        self.max_chunks_per_source = max_chunks_per_source
        self.min_candidate_chunks = min_candidate_chunks
        self.candidate_multiplier = candidate_multiplier
        self.bilingual_expansions = dict(
            bilingual_expansions or MATERIAL_BILINGUAL_QUERY_EXPANSIONS
        )
        self.chunks = base.chunks
        self.last_expanded_query = ""

    def retrieve(self, query: str, k: int = 4) -> list[dict]:
        if not query.strip():
            return []
        expanded_query = expand_material_query(query, self.bilingual_expansions)
        self.last_expanded_query = expanded_query
        candidate_k = max(
            self.min_candidate_chunks, k * self.candidate_multiplier
        )
        raw_candidates = self.base.retrieve(expanded_query, k=candidate_k)
        anchors = {
            token.casefold() for token in self._latin_anchor.findall(expanded_query)
        }
        candidates: list[dict] = []
        for position, raw_item in enumerate(raw_candidates, start=1):
            item = dict(raw_item)
            content = str(item.get("content", "")).casefold()
            item["lexical_anchor_matches"] = sum(
                anchor in content for anchor in anchors
            )
            item["pre_hybrid_rank"] = position
            candidates.append(item)
        if anchors:
            candidates.sort(
                key=lambda item: (
                    -item["lexical_anchor_matches"],
                    item["pre_hybrid_rank"],
                )
            )

        selected: list[dict] = []
        source_counts: dict[str, int] = {}
        for item in candidates:
            source = str(item.get("source", "unknown"))
            if source_counts.get(source, 0) >= self.max_chunks_per_source:
                continue
            selected.append(item)
            source_counts[source] = source_counts.get(source, 0) + 1
            if len(selected) >= k:
                break
        return selected


def build_hybrid_paper_retriever(
    directory: str | Path,
    *,
    embeddings: Embeddings,
    score_threshold: float = 0.0,
    max_chunks_per_source: int = 1,
    min_candidate_chunks: int = 128,
    candidate_multiplier: int = 16,
) -> HybridSourceDiversifiedRetriever:
    root = Path(directory)
    paths = hybrid_corpus_paths(root)
    if not paths:
        raise ValueError(f"no retrieval documents found in {root}")
    documents: list[Document] = []
    for path in paths:
        documents.extend(LangChainPaperRetriever._load_path(path))
    documents.extend(load_corpus_metadata_documents(root))

    alias_options: dict = {}
    alias_path = root / "retrieval-aliases.json"
    if alias_path.exists():
        payload = json.loads(alias_path.read_text(encoding="utf-8"))
        alias_options = {
            "query_expansions": payload.get("query_expansions", {}),
            "source_hints": payload.get("source_hints", {}),
            "min_candidate_pool": int(payload.get("min_candidate_pool", 0)),
        }
    base = LangChainPaperRetriever(
        documents,
        embeddings=embeddings,
        score_threshold=score_threshold,
        **alias_options,
    )
    return HybridSourceDiversifiedRetriever(
        base,
        max_chunks_per_source=max_chunks_per_source,
        min_candidate_chunks=min_candidate_chunks,
        candidate_multiplier=candidate_multiplier,
    )


class MiniMindRAGResponder:
    """Answer with an OpenAI-compatible MiniMind endpoint and explicit evidence."""

    SYSTEM_PROMPT = (
        "你是 MagNova 的材料科研助手。只能依据给定检索证据回答；"
        "证据不足时必须明确说不知道。回答应先给结论，再解释原理，"
        "并在相关句子后用[来源文件 p.页码]标注来源。不要执行材料生成流程。"
    )

    def __init__(
        self,
        endpoint: str,
        *,
        model: str = "minimind",
        timeout_seconds: float = 60.0,
        max_history_messages: int = 6,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        max_evidence_items: int = 8,
        max_evidence_chars: int = 320,
        faithfulness_gate: EvidenceFaithfulnessGate | None = None,
        answerability_gate: EvidenceAnswerabilityGate | None = None,
    ) -> None:
        if repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        if no_repeat_ngram_size < 0:
            raise ValueError("no_repeat_ngram_size must be non-negative")
        if max_evidence_items < 1 or max_evidence_chars < 100:
            raise ValueError("evidence budget must be positive and useful")
        self.endpoint = endpoint
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_history_messages = max_history_messages
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.max_evidence_items = max_evidence_items
        self.max_evidence_chars = max_evidence_chars
        self.faithfulness_gate = faithfulness_gate or EvidenceFaithfulnessGate()
        self.answerability_gate = answerability_gate or EvidenceAnswerabilityGate()
        self.last_raw_answer: str | None = None
        self.last_quality_reasons: list[str] = []
        self.last_used_fallback = False

    @staticmethod
    def _source_label(item: dict) -> str:
        page = f" p.{item['page']}" if item.get("page") else ""
        return f"{item.get('source', 'unknown')}{page}"

    @classmethod
    def format_user_prompt(
        cls,
        question: str,
        *,
        summary: str,
        evidence: list[dict],
    ) -> str:
        evidence_text = "\n\n".join(
            f"[{index}] 来源：{cls._source_label(item)}\n{item['content']}"
            for index, item in enumerate(evidence, start=1)
        )
        context_parts = []
        if summary.strip():
            context_parts.append(f"此前讨论摘要：{summary.strip()}")
        context_parts.append(f"检索证据：\n{evidence_text}")
        context_parts.append(f"当前问题：{question}")
        return "\n\n".join(context_parts)

    @staticmethod
    def _looks_repetitive(text: str) -> bool:
        if not text.strip():
            return True
        sentences = [
            re.sub(r"\s+", "", sentence)
            for sentence in re.split(r"[。！？!?；;\n]+", text)
            if sentence.strip()
        ]
        if len(sentences) >= 4 and len(set(sentences)) / len(sentences) < 0.65:
            return True
        if len(text) >= 240:
            windows = [text[index : index + 24] for index in range(len(text) - 23)]
            if windows and len(set(windows)) / len(windows) < 0.55:
                return True
        return False

    def prepare_evidence(self, evidence: list[dict]) -> list[dict]:
        """Apply the same deterministic evidence budget in training and serving."""

        prepared = []
        for item in evidence[: self.max_evidence_items]:
            clipped = dict(item)
            content = re.sub(r"\s+", " ", str(item.get("content", ""))).strip()
            clipped["content"] = content[: self.max_evidence_chars]
            prepared.append(clipped)
        return prepared

    def _extractive_fallback(self, evidence: list[dict]) -> str:
        parts = []
        source_counts: dict[str, int] = {}
        best_score = evidence[0].get("score") if evidence else None
        for item in evidence:
            source = item.get("source", "unknown")
            item_score = item.get("score")
            if (
                best_score is not None
                and item_score is not None
                and item_score < best_score - 0.10
            ):
                continue
            if source_counts.get(source, 0) >= 2:
                continue
            source_counts[source] = source_counts.get(source, 0) + 1
            content = re.sub(r"\s+", " ", item["content"]).strip()
            excerpt = content[:260] + ("…" if len(content) > 260 else "")
            parts.append(f"- [{self._source_label(item)}] {excerpt}")
            if len(parts) == 3:
                break
        return (
            "MiniMind 本次生成未通过证据引用或重复性质量检查。以下返回检索到的原始证据摘要，"
            "避免把不可靠的生成内容当作科研结论：\n" + "\n".join(parts)
        )

    def answer(self, question, *, messages, summary, evidence) -> str:
        self.last_raw_answer = None
        self.last_quality_reasons = []
        self.last_used_fallback = False
        if not evidence:
            self.last_quality_reasons = ["no_evidence"]
            return (
                "当前论文知识库中没有检索到足够相关的证据。"
                "我暂不根据记忆猜测；请补充论文、关键词或限定研究范围。"
            )

        answerability = self.answerability_gate.assess(question, evidence)
        if not answerability["can_answer"]:
            self.last_quality_reasons = list(answerability["reasons"])
            return (
                "当前检索结果与问题相关，但没有覆盖问题中的关键实体或数值条件。"
                "为避免把相似论文误当作直接证据，我暂不作答；请补充对应论文或放宽问题条件。"
            )

        evidence = self.prepare_evidence(evidence)

        api_messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        api_messages.extend(messages[-self.max_history_messages : -1])
        api_messages.append(
            {
                "role": "user",
                "content": self.format_user_prompt(
                    question,
                    summary=summary,
                    evidence=evidence,
                ),
            }
        )
        response = requests.post(
            self.endpoint,
            json={
                "model": self.model,
                "messages": api_messages,
                "temperature": 0.2,
                "do_sample": False,
                "max_tokens": 256,
                "repetition_penalty": self.repetition_penalty,
                "no_repeat_ngram_size": self.no_repeat_ngram_size,
                "stream": False,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"].strip()
        self.last_raw_answer = answer

        if self._looks_repetitive(answer):
            self.last_quality_reasons.append("repetitive_or_empty")
        self.last_quality_reasons.extend(
            reason
            for reason in self.faithfulness_gate.check(answer, evidence)
            if reason not in self.last_quality_reasons
        )
        if self.last_quality_reasons:
            self.last_used_fallback = True
            return self._extractive_fallback(evidence)
        return answer
