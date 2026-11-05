"""Pure asyncio orchestration for chapter translation and lexical consensus."""

from __future__ import annotations

import asyncio

from lexicon.contracts import BatchRunSummary, LLMProvider
from lexicon.engines.batch import (
    complete_batch,
    conflicts_by_chapter,
    consolidate_batch,
    reconcile_chapter,
    translate_chapter,
    validate_concurrency,
)
from lexicon.glossary import GlossaryStore
from lexicon.models import ChapterDocument, ChapterResult, ConflictResolution


class PureAsyncioEngine:
    def __init__(
        self,
        store: GlossaryStore,
        provider: LLMProvider,
        max_concurrency: int = 4,
        enable_context_injection: bool = True,
        enable_reconciliation: bool = True,
    ) -> None:
        validate_concurrency(max_concurrency)
        self.store = store
        self.provider = provider
        self.max_concurrency = max_concurrency
        self.enable_context_injection = enable_context_injection
        self.enable_reconciliation = enable_reconciliation

    async def run_batch(self, chapters: list[ChapterDocument]) -> BatchRunSummary:
        initial = self.store.snapshot()
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def translate(chapter: ChapterDocument) -> ChapterResult:
            async with semaphore:
                return await translate_chapter(
                    self.provider, chapter, initial, self.enable_context_injection,
                )

        provisional = await asyncio.gather(
            *(translate(chapter) for chapter in sorted(chapters, key=lambda chapter: chapter.number))
        )
        updated, resolutions = consolidate_batch(initial, provisional, self.enable_reconciliation)
        affected = conflicts_by_chapter(resolutions)

        async def reconcile(
            result: ChapterResult, conflicts: list[ConflictResolution],
        ) -> ChapterResult:
            async with semaphore:
                return await reconcile_chapter(self.provider, result, updated, conflicts)

        reconciled = await asyncio.gather(
            *(reconcile(result, affected[result.chapter.number])
              for result in provisional if result.chapter.number in affected)
        )
        return complete_batch(self.store, initial, updated, resolutions, provisional, reconciled)
