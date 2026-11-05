"""LangGraph StateGraph engine using dynamic Send() fan-out and Annotated reducers."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict
from uuid import uuid4

from lexicon.glossary import GlossaryStore
from lexicon.models import (
    ChapterDocument,
    ChapterResult,
    ConflictResolution,
    EvalMetrics,
    GlossarySnapshot,
)
from lexicon.contracts import BatchRunSummary, LLMProvider
from lexicon.engines.batch import (
    complete_batch,
    conflicts_by_chapter,
    consolidate_batch,
    reconcile_chapter,
    translate_chapter,
    validate_concurrency,
)

try:
    from langgraph.checkpoint.memory import MemorySaver  # type: ignore[import-not-found]
    from langgraph.graph import END, START, StateGraph  # type: ignore[import-not-found]
    from langgraph.types import Send  # type: ignore[import-not-found]

    USING_NATIVE_LANGGRAPH = True
except ImportError:
    USING_NATIVE_LANGGRAPH = False
    from lexicon.engines._langgraph_fallback import END, START, MemorySaver, Send, StateGraph


def _collect_results(
    current: list[ChapterResult], update: list[ChapterResult] | None,
) -> list[ChapterResult]:
    # A new batch clears the previous batch's fan-in channels on reused threads.
    return [] if update is None else current + update


class TranslationGraphState(TypedDict, total=False):
    chapters: list[ChapterDocument]
    enable_context_injection: bool
    enable_reconciliation: bool
    initial_snapshot: GlossarySnapshot
    updated_snapshot: GlossarySnapshot
    provisional_results: Annotated[list[ChapterResult], _collect_results]
    resolutions: list[ConflictResolution]
    reconciled_results: Annotated[list[ChapterResult], _collect_results]
    final_translations: dict[int, str]
    eval_metrics: EvalMetrics


class MapWorkerPayload(TypedDict):
    chapter: ChapterDocument
    snapshot: GlossarySnapshot
    enable_context_injection: bool


class ReconcileWorkerPayload(TypedDict):
    result: ChapterResult
    snapshot: GlossarySnapshot
    resolutions: list[ConflictResolution]


class LangGraphEngine:
    """Executes the 4-phase consensus translation lifecycle using LangGraph StateGraph & Send API."""

    def __init__(
        self,
        store: GlossaryStore,
        provider: LLMProvider,
        checkpointer: MemorySaver | None = None,
        enable_context_injection: bool = True,
        enable_reconciliation: bool = True,
        max_concurrency: int = 4,
    ) -> None:
        validate_concurrency(max_concurrency)
        self.max_concurrency = max_concurrency
        self.store = store
        self.provider = provider
        self.checkpointer = checkpointer
        self.enable_context_injection = enable_context_injection
        self.enable_reconciliation = enable_reconciliation
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        store = self.store
        provider = self.provider

        def prepare_snapshot(state: TranslationGraphState) -> dict[str, Any]:
            snap = store.snapshot()
            return {
                "initial_snapshot": snap,
                "updated_snapshot": snap,
                "provisional_results": None,
                "resolutions": [],
                "reconciled_results": None,
            }

        def dispatch_map_workers(state: TranslationGraphState) -> list[Send] | str:
            packets = [
                Send("translate_chapter", {
                    "chapter": chapter,
                    "snapshot": state["initial_snapshot"],
                    "enable_context_injection": state.get("enable_context_injection", True),
                })
                for chapter in sorted(state["chapters"], key=lambda chapter: chapter.number)
            ]
            return packets or "finalize_batch"

        async def translate_worker(payload: MapWorkerPayload) -> dict[str, Any]:
            result = await translate_chapter(
                provider, payload["chapter"], payload["snapshot"],
                payload["enable_context_injection"],
            )
            return {"provisional_results": [result]}

        def consolidate_lexicon(state: TranslationGraphState) -> dict[str, Any]:
            updated, resolutions = consolidate_batch(
                state["initial_snapshot"],
                state["provisional_results"],
                state.get("enable_reconciliation", True),
            )
            return {"resolutions": resolutions, "updated_snapshot": updated}

        def route_after_consolidation(state: TranslationGraphState) -> list[Send] | str:
            affected = conflicts_by_chapter(state["resolutions"])
            packets = [
                Send("reconcile_chapter", {
                    "result": result,
                    "snapshot": state["updated_snapshot"],
                    "resolutions": affected[result.chapter.number],
                })
                for result in state["provisional_results"]
                if result.chapter.number in affected
            ]
            return packets or "finalize_batch"

        async def reconcile_worker(payload: ReconcileWorkerPayload) -> dict[str, Any]:
            result = await reconcile_chapter(
                provider, payload["result"], payload["snapshot"], payload["resolutions"],
            )
            return {"reconciled_results": [result]}

        def finalize_batch(state: TranslationGraphState) -> dict[str, Any]:
            final_map: dict[int, str] = {
                r.chapter.number: r.final_translation
                for r in state.get("provisional_results") or []
            }
            for rec in state.get("reconciled_results") or []:
                final_map[rec.chapter.number] = rec.final_translation
            return {"final_translations": final_map}

        builder = StateGraph(TranslationGraphState)
        builder.add_node("prepare_snapshot", prepare_snapshot)
        builder.add_node("translate_chapter", translate_worker)
        builder.add_node("consolidate_lexicon", consolidate_lexicon)
        builder.add_node("reconcile_chapter", reconcile_worker)
        builder.add_node("finalize_batch", finalize_batch)

        builder.add_edge(START, "prepare_snapshot")
        builder.add_conditional_edges(
            "prepare_snapshot", dispatch_map_workers, ["translate_chapter", "finalize_batch"]
        )
        builder.add_edge("translate_chapter", "consolidate_lexicon")
        builder.add_conditional_edges(
            "consolidate_lexicon",
            route_after_consolidation,
            ["reconcile_chapter", "finalize_batch"],
        )
        builder.add_edge("reconcile_chapter", "finalize_batch")
        builder.add_edge("finalize_batch", END)

        return builder.compile(checkpointer=self.checkpointer)

    async def run_batch(
        self,
        chapters: list[ChapterDocument],
        thread_id: str | None = None,
    ) -> BatchRunSummary:
        state = await self.graph.ainvoke(
            {
                "chapters": chapters,
                "enable_context_injection": self.enable_context_injection,
                "enable_reconciliation": self.enable_reconciliation,
            },
            config={
                "configurable": {"thread_id": thread_id if thread_id is not None else uuid4().hex},
                "max_concurrency": self.max_concurrency,
            },
        )

        return complete_batch(
            self.store,
            state["initial_snapshot"],
            state["updated_snapshot"],
            state["resolutions"],
            state.get("provisional_results") or [],
            state.get("reconciled_results") or [],
        )


__all__ = [
    "END",
    "LangGraphEngine",
    "MemorySaver",
    "START",
    "Send",
    "StateGraph",
    "TranslationGraphState",
    "USING_NATIVE_LANGGRAPH",
]
