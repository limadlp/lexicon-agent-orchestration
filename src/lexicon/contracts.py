"""Inputs and results shared by orchestration and translation adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from lexicon.models import ChapterDocument, ChapterResult, ConflictResolution, TermProposal


@dataclass(frozen=True)
class TranslationResponse:
    """Structured output returned by an LLM translation or reconciliation call."""

    translated_text: str
    proposals: list[TermProposal]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0


class LLMProvider(Protocol):
    """Protocol implemented by both API/Local LLM clients and deterministic test providers."""

    async def translate_chapter(
        self,
        chapter: ChapterDocument,
        context_directive: str,
    ) -> TranslationResponse:
        """Translate a single chapter using only its isolated lexical snapshot context."""
        ...

    async def reconcile_chapter(
        self,
        chapter: ChapterDocument,
        provisional_text: str,
        resolutions: list[ConflictResolution],
        context_directive: str,
    ) -> TranslationResponse:
        """Reconcile a provisional translation when its proposed terms lost in Phase 3."""
        ...


@dataclass
class BatchRunSummary:
    """Execution trace of a multi-chapter translation batch."""

    results: list[ChapterResult]
    resolutions: list[ConflictResolution]
    initial_glossary_version: int
    final_glossary_version: int
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    reconciled_chapters_count: int = 0
