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
    HTTPMagNovaBackend,
    InMemoryMagNovaBackend,
    MagNovaToolAdapter,
    MockMagNovaTool,
    PollingPolicy,
)

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
    "StoredTask",
    "HTTPMagNovaBackend",
    "InMemoryMagNovaBackend",
    "MagNovaToolAdapter",
    "MockMagNovaTool",
    "PollingPolicy",
]
__version__ = "0.4.0"
