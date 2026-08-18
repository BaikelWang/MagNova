from __future__ import annotations

from langchain_core.documents import Document

from magnova_agent.rag import (
    DeterministicHashEmbeddings,
    EvidenceAnswerabilityGate,
    EvidenceFaithfulnessGate,
    GuardedEvidenceRetriever,
    HTTPBGEEmbeddings,
    HybridSourceDiversifiedRetriever,
    LangChainPaperRetriever,
    MaterialScienceQueryGuard,
    MiniMindRAGResponder,
    SentenceEvidencePacker,
    expand_material_query,
    hybrid_corpus_paths,
    load_corpus_metadata_documents,
)


def test_retrieval_ranks_lexically_relevant_source_first() -> None:
    retriever = LangChainPaperRetriever(
        [
            Document(
                page_content="CGCNN uses crystal graph convolution to predict material properties.",
                metadata={"source_name": "cgcnn.md"},
            ),
            Document(
                page_content="MatterGen is a diffusion model for inverse material design.",
                metadata={"source_name": "mattergen.md"},
            ),
        ],
        chunk_size=200,
        chunk_overlap=20,
        score_threshold=0.0,
    )
    evidence = retriever.retrieve("How does MatterGen diffusion design materials?", k=2)
    assert evidence[0]["source"] == "mattergen.md"
    assert evidence[0]["score"] >= evidence[1]["score"]
    assert evidence[0]["chunk_id"]


def test_directory_loader_preserves_file_source(tmp_path) -> None:
    note = tmp_path / "paper-note.md"
    note.write_text("形成能用于衡量材料的热力学稳定性。", encoding="utf-8")
    retriever = LangChainPaperRetriever.from_directory(
        tmp_path,
        score_threshold=0.0,
    )
    evidence = retriever.retrieve("形成能和热力学稳定性", k=1)
    assert evidence[0]["source"] == "paper-note.md"
    assert "形成能" in evidence[0]["content"]


def test_directory_loader_applies_generated_alias_configuration(tmp_path) -> None:
    import json

    (tmp_path / "paper.md").write_text("CrI3 ferromagnetism", encoding="utf-8")
    (tmp_path / "retrieval-aliases.json").write_text(
        json.dumps(
            {
                "min_candidate_pool": 64,
                "source_hints": {"CrI3": "paper.md"},
                "query_expansions": {"CrI3": "chromium triiodide"},
            }
        ),
        encoding="utf-8",
    )
    retriever = LangChainPaperRetriever.from_directory(tmp_path, score_threshold=0.0)
    retriever.retrieve("CrI3 的磁性", k=1)
    assert retriever.min_candidate_pool == 64
    assert retriever.source_hints["CrI3"] == "paper.md"
    assert "chromium triiodide" in retriever.last_expanded_query


def test_high_threshold_returns_no_evidence() -> None:
    retriever = LangChainPaperRetriever(
        [Document(page_content="晶体图神经网络", metadata={"source_name": "a.md"})],
        score_threshold=1.01,
    )
    assert retriever.retrieve("完全无关的问题") == []


def test_domain_query_expansion_is_explicit_and_auditable() -> None:
    retriever = LangChainPaperRetriever(
        [Document(page_content="MatterGen 扩散去噪", metadata={"source_name": "a.md"})],
        score_threshold=0.0,
    )
    retriever.retrieve("MatterGen 如何工作", k=1)
    assert "原子坐标" in retriever.last_expanded_query
    assert retriever.last_expanded_query.startswith("MatterGen 如何工作")


def test_domain_glossary_expands_local_chemistry_language() -> None:
    retriever = LangChainPaperRetriever(
        [Document(page_content="局部化学环境贡献", metadata={"source_name": "a.md"})],
        score_threshold=0.0,
    )
    retriever.retrieve("如何提取局部化学环境的贡献？", k=1)
    assert "CGCNN" in retriever.last_expanded_query


def test_material_scope_guard_rejects_clear_out_of_domain_queries() -> None:
    guard = MaterialScienceQueryGuard()
    assert guard.is_in_scope("MatterGen 如何处理晶体周期边界？")
    assert guard.is_in_scope("MPNN 的消息传递过程是什么？")
    assert guard.is_in_scope("如何研究稀磁半导体异质结中的铁磁性？")
    assert guard.is_in_scope("激光 MBE 生长自旋电子氧化物有什么特点？")
    assert not guard.is_in_scope("AlphaFold 如何预测蛋白质三维结构？")
    assert not guard.is_in_scope("PostgreSQL 的 B-tree 索引如何工作？")


def test_guarded_retriever_skips_embedding_for_out_of_scope_query() -> None:
    class StubRetriever:
        def __init__(self):
            self.called = False

        def retrieve(self, query, k=4):
            self.called = True
            return [{"source": "noise.md"}]

    base = StubRetriever()
    guarded = GuardedEvidenceRetriever(base)
    assert guarded.retrieve("李白是谁？") == []
    assert base.called is False


def test_explicit_model_name_reranks_matching_source(monkeypatch) -> None:
    retriever = LangChainPaperRetriever(
        [Document(page_content="placeholder", metadata={"source_name": "x.md"})],
        score_threshold=0.0,
    )
    wrong = Document(
        page_content="generic graph content",
        metadata={"source_name": "cgcnn.md", "chunk_id": "wrong"},
    )
    expected = Document(
        page_content="message passing content",
        metadata={"source_name": "mpnn.md", "chunk_id": "expected"},
    )
    monkeypatch.setattr(
        retriever.vector_store,
        "similarity_search_with_score",
        lambda query, k: [(wrong, 0.80), (expected, 0.70)],
    )
    evidence = retriever.retrieve("MPNN 主要描述哪些过程？", k=2)
    assert evidence[0]["source"] == "mpnn.md"
    assert evidence[0]["dense_rank"] == 2
    assert evidence[0]["source_hint_match"] is True


def test_source_hint_uses_wider_candidate_pool_then_returns_requested_k(monkeypatch) -> None:
    retriever = LangChainPaperRetriever(
        [Document(page_content="seed", metadata={"source_name": "seed.md"})],
        score_threshold=0.0,
    )
    wrong = Document(
        page_content="general project text",
        metadata={"source_name": "magnova-plan.md", "chunk_id": "wrong"},
    )
    expected = Document(
        page_content="message passing content",
        metadata={"source_name": "mpnn.md", "chunk_id": "expected"},
    )
    requested_sizes = []

    def search(query, k):
        del query
        requested_sizes.append(k)
        return [
            (wrong, 0.80),
            (wrong, 0.79),
            (wrong, 0.78),
            (wrong, 0.77),
            (expected, 0.70),
        ]

    monkeypatch.setattr(retriever.vector_store, "similarity_search_with_score", search)
    evidence = retriever.retrieve("MPNN 的核心计算过程是什么？", k=2)
    assert requested_sizes == [6]
    assert len(evidence) == 2
    assert evidence[0]["source"] == "mpnn.md"
    assert evidence[0]["dense_rank"] == 5


def test_multiple_intent_hints_beat_a_single_entity_hint(monkeypatch) -> None:
    retriever = LangChainPaperRetriever(
        [Document(page_content="seed", metadata={"source_name": "seed.md"})],
        score_threshold=0.0,
        source_hints={
            "LK-99": "claim.pdf",
            "Cu2S": "counter.pdf",
            "替代解释": "counter.pdf",
        },
    )
    claim = Document(
        page_content="claimed superconductivity",
        metadata={"source_name": "claim.pdf", "chunk_id": "claim"},
    )
    counter = Document(
        page_content="structural transition counterevidence",
        metadata={"source_name": "counter.pdf", "chunk_id": "counter"},
    )
    monkeypatch.setattr(
        retriever.vector_store,
        "similarity_search_with_score",
        lambda query, k: [(claim, 0.90), (counter, 0.70)],
    )
    evidence = retriever.retrieve("含 Cu2S 的 LK-99 有什么替代解释？", k=2)
    assert evidence[0]["source"] == "counter.pdf"
    assert evidence[0]["source_hint_matches"] == 2


def test_minimind_is_in_scope_and_hints_magnova_plan(monkeypatch) -> None:
    assert MaterialScienceQueryGuard().is_in_scope("MiniMind 的职责是什么？")
    retriever = LangChainPaperRetriever(
        [Document(page_content="seed", metadata={"source_name": "seed.md"})],
        score_threshold=0.0,
    )
    wrong = Document(
        page_content="generic model text",
        metadata={"source_name": "mattergen.md", "chunk_id": "wrong"},
    )
    expected = Document(
        page_content="agent responsibilities",
        metadata={"source_name": "magnova-plan.md", "chunk_id": "expected"},
    )
    monkeypatch.setattr(
        retriever.vector_store,
        "similarity_search_with_score",
        lambda query, k: [(wrong, 0.80), (expected, 0.70)],
    )
    evidence = retriever.retrieve("MiniMind 的职责是什么？", k=2)
    assert evidence[0]["source"] == "magnova-plan.md"
    assert evidence[0]["source_hint_match"] is True


def test_http_bge_adapter_adds_query_instruction(monkeypatch) -> None:
    payloads = []

    class StubResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": [
                    {"index": index, "embedding": [float(index), 1.0]}
                    for index, _ in enumerate(payloads[-1]["input"])
                ]
            }

    def fake_post(url, *, json, timeout):
        payloads.append(json)
        return StubResponse()

    monkeypatch.setattr("magnova_agent.rag.requests.post", fake_post)
    embeddings = HTTPBGEEmbeddings("http://localhost:8508/v1/embeddings")
    assert len(embeddings.embed_documents(["文档一", "文档二"])) == 2
    assert len(embeddings.embed_query("查询")) == 2
    assert payloads[0]["input"] == ["文档一", "文档二"]
    assert payloads[1]["input"][0].endswith("查询")
    assert payloads[1]["input"][0] != "查询"


def test_http_bge_adapter_batches_large_document_sets(monkeypatch) -> None:
    payloads = []

    class StubResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": [
                    {"index": index, "embedding": [float(index)]}
                    for index, _ in enumerate(payloads[-1]["input"])
                ]
            }

    def fake_post(url, *, json, timeout):
        payloads.append(json)
        return StubResponse()

    monkeypatch.setattr("magnova_agent.rag.requests.post", fake_post)
    embeddings = HTTPBGEEmbeddings(
        "http://localhost:8508/v1/embeddings", batch_size=2
    )
    vectors = embeddings.embed_documents(["a", "b", "c", "d", "e"])
    assert [payload["input"] for payload in payloads] == [
        ["a", "b"],
        ["c", "d"],
        ["e"],
    ]
    assert vectors == [[0.0], [1.0], [0.0], [1.0], [0.0]]


def test_responder_refuses_when_evidence_is_empty() -> None:
    responder = MiniMindRAGResponder("http://unused/v1/chat/completions")
    answer = responder.answer("问题", messages=[], summary="", evidence=[])
    assert "没有检索到足够相关的证据" in answer


def test_responder_sends_evidence_and_rejects_uncited_answer(monkeypatch) -> None:
    captured = {}

    class StubResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "基于证据的回答"}}]}

    def fake_post(url, *, json, timeout):
        captured.update({"url": url, "json": json, "timeout": timeout})
        return StubResponse()

    monkeypatch.setattr("magnova_agent.rag.requests.post", fake_post)
    responder = MiniMindRAGResponder("http://localhost:8507/v1/chat/completions")
    answer = responder.answer(
        "MatterGen 是什么？",
        messages=[{"role": "user", "content": "MatterGen 是什么？"}],
        summary="讨论生成模型",
        evidence=[
            {
                "source": "mattergen.md",
                "page": 2,
                "content": "MatterGen 是材料生成扩散模型。",
            }
        ],
    )
    assert captured["url"].endswith("/v1/chat/completions")
    assert "MatterGen 是材料生成扩散模型" in captured["json"]["messages"][-1]["content"]
    assert "质量检查" in answer
    assert "mattergen.md p.2" in answer
    assert captured["json"]["max_tokens"] == 256
    assert captured["json"]["repetition_penalty"] == 1.0
    assert captured["json"]["no_repeat_ngram_size"] == 0


def test_responder_accepts_answer_with_real_inline_source(monkeypatch) -> None:
    class StubResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {"message": {"content": "坐标扩散使用 wrapped Normal。[paper.md p.3]"}}
                ]
            }

    monkeypatch.setattr(
        "magnova_agent.rag.requests.post",
        lambda *args, **kwargs: StubResponse(),
    )
    responder = MiniMindRAGResponder("http://localhost:8507/v1/chat/completions")
    answer = responder.answer(
        "问题",
        messages=[],
        summary="",
        evidence=[{"source": "paper.md", "page": 3, "content": "坐标扩散使用 wrapped Normal 处理周期边界。"}],
    )
    assert answer == "坐标扩散使用 wrapped Normal。[paper.md p.3]"


def test_responder_replaces_repetitive_generation_with_evidence(monkeypatch) -> None:
    repeated = "。".join(["重复内容没有新增信息"] * 8)

    class StubResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": repeated}}]}

    monkeypatch.setattr(
        "magnova_agent.rag.requests.post",
        lambda *args, **kwargs: StubResponse(),
    )
    responder = MiniMindRAGResponder("http://localhost:8507/v1/chat/completions")
    answer = responder.answer(
        "问题",
        messages=[],
        summary="",
        evidence=[{"source": "paper.md", "content": "可信的论文证据"}],
    )
    assert "质量检查" in answer
    assert "可信的论文证据" in answer
    assert responder.last_raw_answer == repeated
    assert "repetitive_or_empty" in responder.last_quality_reasons
    assert "missing_evidence_citation" in responder.last_quality_reasons
    assert responder.last_used_fallback is True


def test_extractive_fallback_drops_evidence_far_below_top_score(monkeypatch) -> None:
    class StubResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "无来源的回答"}}]}

    monkeypatch.setattr(
        "magnova_agent.rag.requests.post",
        lambda *args, **kwargs: StubResponse(),
    )
    responder = MiniMindRAGResponder("http://localhost:8507/v1/chat/completions")
    answer = responder.answer(
        "问题",
        messages=[],
        summary="",
        evidence=[
            {"source": "top.md", "content": "最高分证据", "score": 0.80},
            {"source": "near.md", "content": "接近最高分的证据", "score": 0.74},
            {"source": "weak.md", "content": "弱相关噪声", "score": 0.60},
        ],
    )
    assert "最高分证据" in answer
    assert "接近最高分的证据" in answer
    assert "弱相关噪声" not in answer


def test_rag_prompt_formatter_matches_runtime_contract() -> None:
    prompt = MiniMindRAGResponder.format_user_prompt(
        "MatterGen 如何处理周期边界？",
        summary="此前在讨论 MatterGen。",
        evidence=[
            {
                "source": "mattergen.md",
                "page": 2,
                "content": "坐标扩散使用 wrapped Normal。",
            }
        ],
    )
    assert "此前讨论摘要：此前在讨论 MatterGen。" in prompt
    assert "[1] 来源：mattergen.md p.2" in prompt
    assert "当前问题：MatterGen 如何处理周期边界？" in prompt


def test_responder_applies_deterministic_evidence_budget() -> None:
    responder = MiniMindRAGResponder(
        "http://localhost:8507/v1/chat/completions",
        max_evidence_items=2,
        max_evidence_chars=100,
    )
    evidence = [
        {"source": f"paper-{index}.md", "content": "  证据\n" + "甲" * 150}
        for index in range(3)
    ]
    prepared = responder.prepare_evidence(evidence)
    assert len(prepared) == 2
    assert len(prepared[0]["content"]) == 100
    assert "\n" not in prepared[0]["content"]
    assert evidence[0]["content"].startswith("  ")


def test_sentence_packer_reranks_relevant_sentence_and_preserves_source() -> None:
    class StubRetriever:
        def retrieve(self, query, k=4):
            del query, k
            return [
                {
                    "source": "mattergen.md",
                    "score": 0.8,
                    "chunk_id": "c1",
                    "content": "项目背景与其他内容。坐标扩散使用 wrapped Normal 处理周期边界。部署采用离线流程。",
                }
            ]

    packer = SentenceEvidencePacker(
        StubRetriever(),
        DeterministicHashEmbeddings(),
        max_sentences=2,
    )
    packed = packer.retrieve("wrapped Normal 周期边界", k=2)
    assert packed[0]["source"] == "mattergen.md"
    assert "wrapped Normal" in packed[0]["content"]
    assert "sentence_score" in packed[0]


def test_sentence_packer_restores_pdf_line_break_context() -> None:
    class StubRetriever:
        def retrieve(self, query, k=4):
            del query, k
            return [
                {
                    "source": "cri3.pdf",
                    "score": 0.5,
                    "content": (
                        "CrI3 is an Ising ferromagnet.\n"
                        "Its Curie temperature is 45 K, only slightly\n"
                        "lower than the 61 K bulk value."
                    ),
                }
            ]

    packer = SentenceEvidencePacker(
        StubRetriever(), DeterministicHashEmbeddings(), max_sentences=1
    )
    packed = packer.retrieve("CrI3 居里温度是多少？", k=1)
    assert "45 K" in packed[0]["content"]
    assert "61 K" in packed[0]["content"]


def test_sentence_packer_boosts_exact_numeric_constraints_from_expansion() -> None:
    class StubRetriever:
        last_expanded_query = "LK-99 Tc 400 K Meissner effect"

        def retrieve(self, query, k=4):
            del query, k
            return [
                {
                    "source": "lk99.pdf",
                    "score": 0.7,
                    "content": (
                        "LK-99 Meissner effect evidence.\n"
                        "The preprint reports Tc above 400 K at ambient pressure."
                    ),
                }
            ]

    packer = SentenceEvidencePacker(
        StubRetriever(), DeterministicHashEmbeddings(), max_sentences=1
    )
    packed = packer.retrieve("LK-99 的 Tc 是多少？", k=1)
    assert "400 K" in packed[0]["content"]


def test_faithfulness_gate_rejects_unsupported_number_and_fake_citation() -> None:
    gate = EvidenceFaithfulnessGate(min_bigram_coverage=0.1)
    evidence = [{"source": "paper.md", "content": "模型使用 20 个原子的稳定结构。"}]
    reasons = gate.check("模型使用 999 个原子。[paper.md]", evidence)
    assert "unsupported_number" in reasons
    assert "unknown_evidence_citation" in gate.check(
        "模型使用 20 个原子。[fake.md]", evidence
    )


def test_faithfulness_gate_allows_grounded_answer_and_safe_refusal() -> None:
    gate = EvidenceFaithfulnessGate()
    evidence = [
        {
            "source": "mattergen.md",
            "content": "坐标扩散使用 wrapped Normal 处理周期边界。",
        }
    ]
    assert gate.check(
        "坐标扩散使用 wrapped Normal 处理周期边界。[mattergen.md]", evidence
    ) == []
    assert gate.check(
        "证据没有提供具体耗时，因此无法根据现有证据确定。[mattergen.md]",
        evidence,
    ) == []
    assert gate.check(
        "根据给定检索证据，无法确定这个问题的答案。", evidence
    ) == []
    assert "missing_evidence_citation" in gate.check(
        "无法确定，但居里温度为 999 K。", evidence
    )


def test_answerability_gate_requires_question_entities_and_numeric_constraints() -> None:
    gate = EvidenceAnswerabilityGate(min_top_score=0.48)
    unrelated = [{"score": 0.70, "content": "LaH10 reaches 250 K under pressure."}]
    result = gate.assess("MgB2 在 20 K 和 5 T 下的临界电流密度是多少？", unrelated)
    assert result["can_answer"] is False
    assert "missing_anchor:MgB2" in result["reasons"]
    assert any(reason.startswith("missing_numeric_constraint:20") for reason in result["reasons"])

    grounded = [{"score": 0.55, "content": "LaH10 shows superconductivity near 250 K."}]
    assert gate.assess("LaH10 的 250 K 超导结果是什么？", grounded)["can_answer"] is True


def test_responder_refuses_before_generation_when_answerability_gate_fails(monkeypatch) -> None:
    called = False

    def fake_post(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("generation must not be called")

    monkeypatch.setattr("magnova_agent.rag.requests.post", fake_post)
    responder = MiniMindRAGResponder("http://localhost:8507/v1/chat/completions")
    answer = responder.answer(
        "NIST 报告的硅在 300 K 时的精确晶格常数是多少？",
        messages=[],
        summary="",
        evidence=[{"score": 0.70, "source": "other.pdf", "content": "general crystal data"}],
    )
    assert "暂不作答" in answer
    assert called is False
    assert "missing_anchor:NIST" in responder.last_quality_reasons


def test_formal_hybrid_retriever_expands_reranks_and_diversifies() -> None:
    class Stub:
        chunks = []

        def retrieve(self, query, k=4):
            assert "monolayer" in query
            assert k == 128
            return [
                {"source": "generic.pdf", "content": "magnetic monolayer"},
                {"source": "target.pdf", "content": "vanadium-doped WSe2 monolayer"},
                {"source": "target.pdf", "content": "another WSe2 result"},
            ]

    result = HybridSourceDiversifiedRetriever(Stub()).retrieve(
        "WSe2 单层材料", k=2
    )
    assert [item["source"] for item in result] == ["target.pdf", "generic.pdf"]
    assert result[0]["lexical_anchor_matches"] >= 1
    assert "monolayer" in expand_material_query("单层材料")


def test_formal_hybrid_corpus_metadata_contract(tmp_path) -> None:
    (tmp_path / "paper.pdf").write_bytes(b"pdf")
    (tmp_path / "note.md").write_text("note", encoding="utf-8")
    (tmp_path / "paper-catalog.md").write_text(
        "# Catalog\n\n## Magnetic Paper\n\n- 检索来源：`paper.pdf`\n",
        encoding="utf-8",
    )
    (tmp_path / "v6-document-catalog.json").write_text(
        '[{"source":"paper.pdf","title":"Magnetic Paper","subject":"materials"}]',
        encoding="utf-8",
    )
    assert [path.name for path in hybrid_corpus_paths(tmp_path)] == [
        "note.md",
        "paper.pdf",
    ]
    metadata = load_corpus_metadata_documents(tmp_path)
    assert len(metadata) == 1
    assert metadata[0].metadata["source_name"] == "paper.pdf"
    assert metadata[0].metadata["document_type"] == "verified_metadata"
