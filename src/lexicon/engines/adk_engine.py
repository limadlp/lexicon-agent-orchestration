"""Google ADK orchestration for chapter translation and lexical consensus."""

from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator, Awaitable, Callable
from uuid import uuid4

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
from lexicon.models import ChapterDocument, ChapterResult

try:
    from google.adk.agents import BaseAgent, LlmAgent, ParallelAgent, SequentialAgent
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.events import Event, EventActions
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.adk.tools import FunctionTool
    from google.genai.types import Content, Part

    USING_NATIVE_ADK = True
except ImportError:
    from lexicon.engines._adk_fallback import (
        BaseAgent, Event, EventActions, FunctionTool, InMemorySessionService,
        InvocationContext, LlmAgent, ParallelAgent, Runner, SequentialAgent,
    )

    USING_NATIVE_ADK = False


class StateUpdateAgent(BaseAgent):
    """Run a pipeline step and publish its outputs through the session."""

    step: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        update = await self.step(ctx.session.state)
        yield Event(author=self.name, actions=EventActions(state_delta=update))


class ReconciliationAgent(BaseAgent):
    worker_pool: Callable[[dict[str, Any]], list[BaseAgent]]

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        workers = self.worker_pool(ctx.session.state)
        if workers:
            pool = ParallelAgent(name="ParallelReconciliation", sub_agents=workers)
            async for event in pool.run_async(ctx):
                yield event


class GoogleADKEngine:
    def __init__(
        self,
        store: GlossaryStore,
        provider: LLMProvider,
        session_service: InMemorySessionService | None = None,
        enable_context_injection: bool = True,
        enable_reconciliation: bool = True,
        model_name: str = "gemini-2.5-flash",
        max_concurrency: int = 4,
    ) -> None:
        validate_concurrency(max_concurrency)
        self.store = store
        self.provider = provider
        self.session_service = session_service or InMemorySessionService()
        self.enable_context_injection = enable_context_injection
        self.enable_reconciliation = enable_reconciliation
        self.model_name = model_name
        self.max_concurrency = max_concurrency

    def build_root_pipeline(self, chapters: list[ChapterDocument]) -> SequentialAgent:
        semaphore = asyncio.Semaphore(self.max_concurrency)
        ordered = sorted(chapters, key=lambda chapter: chapter.number)

        async def prepare(state: dict[str, Any]) -> dict[str, Any]:
            return {"initial_snapshot": self.store.snapshot()}

        def translation_worker(chapter: ChapterDocument) -> BaseAgent:
            async def step(state: dict[str, Any]) -> dict[str, Any]:
                async with semaphore:
                    result = await translate_chapter(
                        self.provider, chapter, state["initial_snapshot"],
                        self.enable_context_injection,
                    )
                return {f"translation_{chapter.number}": result}

            return StateUpdateAgent(name=f"TranslateChapter{chapter.number}", step=step)

        async def consolidate(state: dict[str, Any]) -> dict[str, Any]:
            results = [state[f"translation_{chapter.number}"] for chapter in ordered]
            updated, resolutions = consolidate_batch(
                state["initial_snapshot"], results, self.enable_reconciliation,
            )
            return {"updated_snapshot": updated, "resolutions": resolutions}

        def reconciliation_workers(state: dict[str, Any]) -> list[BaseAgent]:
            affected = conflicts_by_chapter(state["resolutions"])

            def worker(result: ChapterResult) -> BaseAgent:
                number = result.chapter.number

                async def step(state: dict[str, Any]) -> dict[str, Any]:
                    async with semaphore:
                        revised = await reconcile_chapter(
                            self.provider, result, state["updated_snapshot"], affected[number],
                        )
                    return {f"reconciliation_{number}": revised}

                return StateUpdateAgent(name=f"ReconcileChapter{number}", step=step)

            return [
                worker(state[f"translation_{chapter.number}"])
                for chapter in ordered if chapter.number in affected
            ]

        return SequentialAgent(
            name="LexiconADKRootPipeline",
            sub_agents=[
                StateUpdateAgent(name="PrepareSnapshot", step=prepare),
                ParallelAgent(
                    name="ParallelChapterTranslation",
                    sub_agents=[translation_worker(chapter) for chapter in ordered],
                ),
                StateUpdateAgent(name="ConsolidateGlossary", step=consolidate),
                ReconciliationAgent(name="ReconcileGlossary", worker_pool=reconciliation_workers),
            ],
        )

    async def run_batch(
        self, chapters: list[ChapterDocument], session_id: str | None = None,
    ) -> BatchRunSummary:
        if not chapters:
            snapshot = self.store.snapshot()
            return complete_batch(self.store, snapshot, snapshot, [], [], [])

        session_id = session_id if session_id is not None else uuid4().hex
        session_options = {
            "app_name": "lexicon", "user_id": "lexicon_user", "session_id": session_id,
        }
        await self.session_service.create_session(**session_options)
        runner = Runner(
            agent=self.build_root_pipeline(chapters),
            app_name=session_options["app_name"],
            session_service=self.session_service,
        )
        message = (
            {"new_message": Content(role="user", parts=[Part(text="Translate chapters")])}
            if USING_NATIVE_ADK else {}
        )
        async for _ in runner.run_async(
            user_id=session_options["user_id"], session_id=session_id, **message,
        ):
            pass

        session = await self.session_service.get_session(**session_options)
        if session is None:
            raise RuntimeError(f"Sessão ADK não encontrada: {session_id}")
        state = session.state
        return complete_batch(
            self.store,
            state["initial_snapshot"],
            state["updated_snapshot"],
            state["resolutions"],
            [state[f"translation_{chapter.number}"] for chapter in chapters],
            [
                state[f"reconciliation_{chapter.number}"]
                for chapter in chapters if f"reconciliation_{chapter.number}" in state
            ],
        )


__all__ = [
    "BaseAgent", "FunctionTool", "GoogleADKEngine", "InMemorySessionService",
    "LlmAgent", "ParallelAgent", "Runner", "SequentialAgent", "USING_NATIVE_ADK",
]
