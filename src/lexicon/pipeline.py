"""Unified translation pipeline coordinator supporting interchangeable orchestration engines."""

from __future__ import annotations

from typing import Literal

from lexicon.contracts import BatchRunSummary, LLMProvider
from lexicon.glossary import GlossaryStore
from lexicon.models import ChapterDocument


EngineType = Literal["asyncio", "langgraph", "adk"]


class TranslationCoordinator:
    """Multi-engine facade delegating to PureAsyncio, LangGraph, or Google ADK runtimes."""

    def __init__(
        self,
        store: GlossaryStore,
        provider: LLMProvider,
        engine: EngineType = "asyncio",
        max_concurrency: int = 4,
        enable_context_injection: bool = True,
        enable_reconciliation: bool = True,
    ) -> None:
        self.store = store
        self.provider = provider
        self.engine_type = engine
        self.runtime = engine
        self.max_concurrency = max_concurrency
        self.enable_context_injection = enable_context_injection
        self.enable_reconciliation = enable_reconciliation

        if engine == "langgraph":
            from lexicon.engines.langgraph_engine import USING_NATIVE_LANGGRAPH, LangGraphEngine

            if not USING_NATIVE_LANGGRAPH:
                self.runtime = "langgraph (fallback)"
            engine_class = LangGraphEngine
        elif engine == "adk":
            from lexicon.engines.adk_engine import USING_NATIVE_ADK, GoogleADKEngine

            if not USING_NATIVE_ADK:
                self.runtime = "adk (fallback)"
            engine_class = GoogleADKEngine
        else:
            from lexicon.engines.pure_asyncio import PureAsyncioEngine

            engine_class = PureAsyncioEngine

        self._engine = engine_class(
            store=store,
            provider=provider,
            max_concurrency=max_concurrency,
            enable_context_injection=enable_context_injection,
            enable_reconciliation=enable_reconciliation,
        )

    async def translate_batch(
        self,
        chapters: list[ChapterDocument],
    ) -> BatchRunSummary:
        if len({chapter.number for chapter in chapters}) != len(chapters):
            raise ValueError("Os números dos capítulos devem ser únicos")
        return await self._engine.run_batch(chapters)
