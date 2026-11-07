"""Lexicon: Stateful parallel literary translation and evaluation engine."""

from lexicon.dataset import build_oz_dataset
from lexicon.engines import GoogleADKEngine, LangGraphEngine, PureAsyncioEngine
from lexicon.evals import TranslationEvaluator, compute_chrf, shannon_entropy_bits
from lexicon.glossary import GlossaryStore
from lexicon.models import (
    CharacterEntry,
    ChapterDocument,
    ChapterResult,
    ConflictResolution,
    EvalMetrics,
    GlossarySnapshot,
    TermEntry,
    TermProposal,
)
from lexicon.pipeline import BatchRunSummary, TranslationCoordinator
from lexicon.providers import LLMProvider, OpenAICompatibleProvider, TranslationResponse

__all__ = [
    "BatchRunSummary",
    "CharacterEntry",
    "ChapterDocument",
    "ChapterResult",
    "ConflictResolution",
    "EvalMetrics",
    "GlossarySnapshot",
    "GlossaryStore",
    "GoogleADKEngine",
    "LLMProvider",
    "LangGraphEngine",
    "OpenAICompatibleProvider",
    "PureAsyncioEngine",
    "TermEntry",
    "TermProposal",
    "TranslationCoordinator",
    "TranslationEvaluator",
    "TranslationResponse",
    "build_oz_dataset",
    "compute_chrf",
    "shannon_entropy_bits",
]
