"""MagNova material requirement agent."""

from .agent import MagNovaAgent
from .conversation import (
    ConversationState,
    InMemoryConversationStore,
    MultiTurnMagNovaAgent,
    merge_follow_up,
)
from .interpreter import MiniMindInterpreter, RuleBasedInterpreter
from .normalization import NormalizationResult, normalize_for_llm
from .pipeline import (
    MockMaterialGenerator,
    MockPipelineMagNovaBackend,
    MockPropertyPredictor,
    RuleBasedCandidateScreener,
)
from .task_manager import MagNovaTaskManager, SQLiteTaskStore, StoredTask
from .tooling import (
    CliMagNovaBackend,
    HTTPMagNovaBackend,
    InMemoryMagNovaBackend,
    MagNovaBlocked,
    MagNovaNeedsClarification,
    MagNovaToolAdapter,
    MockMagNovaTool,
    PollingPolicy,
)
from .telemetry import InteractionRecorder, prompt_fingerprint, redact_text
from .research_graph import (
    EvidenceRetriever,
    ExplicitIntentRouter,
    MagNovaResearchGraph,
    QueryCondenser,
    ResearchGraphState,
    ResearchResponder,
    RuleBasedQueryCondenser,
)
from .rag import (
    DeterministicHashEmbeddings,
    EvidenceAnswerabilityGate,
    EvidenceFaithfulnessGate,
    GuardedEvidenceRetriever,
    HTTPBGEEmbeddings,
    LangChainPaperRetriever,
    MaterialScienceQueryGuard,
    MiniMindRAGResponder,
    SentenceEvidencePacker,
)
from .rag_evaluation import RAGEvaluationCase, RAGEvaluator, load_rag_evaluation_cases
from .rag_answer_evaluation import (
    RAGAnswerEvaluationCase,
    RAGAnswerEvaluator,
    extract_cited_sources,
    load_rag_answer_evaluation_cases,
)
from .claim_evidence import (
    Answerability,
    CLAIM_EVIDENCE_SYSTEM_PROMPT,
    ClaimEvidenceAnswer,
    ClaimEvidenceValidator,
    ClaimValidationCode,
    ClaimValidationIssue,
    ClaimValidationReport,
    EvidenceClaim,
    index_evidence,
    format_claim_evidence_user_prompt,
    parse_claim_evidence_response,
    render_claim_evidence_answer,
)
from .rag_failure_analysis import (
    analyse_failure_reports,
    render_failure_analysis_markdown,
)
from .rag_v4_data import (
    ClaimEvidenceCandidate,
    EvidenceRecord,
    ReviewStatus,
    audit_candidate_pool,
)
from .claim_evidence_responder import MiniMindClaimEvidenceResponder

__all__ = [
    "MagNovaAgent",
    "MiniMindInterpreter",
    "RuleBasedInterpreter",
    "NormalizationResult",
    "normalize_for_llm",
    "ConversationState",
    "InMemoryConversationStore",
    "MultiTurnMagNovaAgent",
    "merge_follow_up",
    "MockMaterialGenerator",
    "MockPipelineMagNovaBackend",
    "MockPropertyPredictor",
    "RuleBasedCandidateScreener",
    "MagNovaTaskManager",
    "SQLiteTaskStore",
    "InteractionRecorder",
    "prompt_fingerprint",
    "redact_text",
    "ExplicitIntentRouter",
    "EvidenceRetriever",
    "MagNovaResearchGraph",
    "QueryCondenser",
    "ResearchGraphState",
    "ResearchResponder",
    "RuleBasedQueryCondenser",
    "DeterministicHashEmbeddings",
    "EvidenceAnswerabilityGate",
    "EvidenceFaithfulnessGate",
    "GuardedEvidenceRetriever",
    "HTTPBGEEmbeddings",
    "LangChainPaperRetriever",
    "MaterialScienceQueryGuard",
    "MiniMindRAGResponder",
    "SentenceEvidencePacker",
    "RAGEvaluationCase",
    "RAGEvaluator",
    "load_rag_evaluation_cases",
    "RAGAnswerEvaluationCase",
    "RAGAnswerEvaluator",
    "extract_cited_sources",
    "load_rag_answer_evaluation_cases",
    "Answerability",
    "CLAIM_EVIDENCE_SYSTEM_PROMPT",
    "ClaimEvidenceAnswer",
    "ClaimEvidenceValidator",
    "ClaimValidationCode",
    "ClaimValidationIssue",
    "ClaimValidationReport",
    "EvidenceClaim",
    "index_evidence",
    "format_claim_evidence_user_prompt",
    "parse_claim_evidence_response",
    "render_claim_evidence_answer",
    "analyse_failure_reports",
    "render_failure_analysis_markdown",
    "ClaimEvidenceCandidate",
    "EvidenceRecord",
    "ReviewStatus",
    "audit_candidate_pool",
    "MiniMindClaimEvidenceResponder",
    "StoredTask",
    "CliMagNovaBackend",
    "HTTPMagNovaBackend",
    "InMemoryMagNovaBackend",
    "MagNovaBlocked",
    "MagNovaNeedsClarification",
    "MagNovaToolAdapter",
    "MockMagNovaTool",
    "PollingPolicy",
]
__version__ = "0.4.0"
