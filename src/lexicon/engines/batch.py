"""Shared translation rules for engines with different orchestration runtimes."""

from dataclasses import replace

from lexicon.contracts import BatchRunSummary, LLMProvider
from lexicon.glossary import GlossaryStore
from lexicon.models import ChapterDocument, ChapterResult, ConflictResolution, GlossarySnapshot


def validate_concurrency(max_concurrency: int) -> None:
    if max_concurrency < 1:
        raise ValueError("O limite de concorrência deve ser maior que zero")


async def translate_chapter(
    provider: LLMProvider,
    chapter: ChapterDocument,
    snapshot: GlossarySnapshot,
    enable_context_injection: bool,
) -> ChapterResult:
    terms, characters, directive = [], [], ""
    if enable_context_injection:
        terms, characters = GlossaryStore.retrieve_chapter_context(snapshot, chapter.content)
        directive = GlossaryStore.format_context_directive(terms, characters)
    response = await provider.translate_chapter(chapter, directive)
    return ChapterResult(
        chapter=chapter,
        provisional_translation=response.translated_text,
        final_translation=response.translated_text,
        proposals=list(response.proposals),
        active_terms=terms,
        active_characters=characters,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        latency_ms=response.latency_ms,
    )


def consolidate_batch(
    snapshot: GlossarySnapshot,
    results: list[ChapterResult],
    enabled: bool,
) -> tuple[GlossarySnapshot, list[ConflictResolution]]:
    if not enabled:
        return snapshot, []
    proposals = [
        proposal
        for result in sorted(results, key=lambda result: result.chapter.number)
        for proposal in result.proposals
    ]
    if not proposals:
        return snapshot, []
    staged = GlossaryStore.from_snapshot(snapshot)
    resolutions = staged.consolidate_proposals(proposals)
    return staged.snapshot(), resolutions


def conflicts_by_chapter(
    resolutions: list[ConflictResolution],
) -> dict[int, list[ConflictResolution]]:
    affected: dict[int, list[ConflictResolution]] = {}
    for resolution in resolutions:
        for number in resolution.affected_chapters:
            affected.setdefault(number, []).append(resolution)
    return affected


async def reconcile_chapter(
    provider: LLMProvider,
    result: ChapterResult,
    snapshot: GlossarySnapshot,
    resolutions: list[ConflictResolution],
) -> ChapterResult:
    terms, characters = GlossaryStore.retrieve_chapter_context(snapshot, result.chapter.content)
    response = await provider.reconcile_chapter(
        chapter=result.chapter,
        provisional_text=result.provisional_translation,
        resolutions=resolutions,
        context_directive=GlossaryStore.format_context_directive(terms, characters),
    )
    return replace(
        result,
        final_translation=response.translated_text,
        active_terms=terms,
        active_characters=characters,
        reconciled=True,
        prompt_tokens=result.prompt_tokens + response.prompt_tokens,
        completion_tokens=result.completion_tokens + response.completion_tokens,
        latency_ms=result.latency_ms + response.latency_ms,
    )


def complete_batch(
    store: GlossaryStore,
    initial: GlossarySnapshot,
    updated: GlossarySnapshot,
    resolutions: list[ConflictResolution],
    provisional: list[ChapterResult],
    reconciled: list[ChapterResult],
) -> BatchRunSummary:
    results_by_number = {result.chapter.number: result for result in provisional}
    results_by_number.update((result.chapter.number, result) for result in reconciled)
    results = [results_by_number[number] for number in sorted(results_by_number)]
    if updated.version != initial.version:
        store.commit(GlossaryStore.from_snapshot(updated), expected_version=initial.version)
    return BatchRunSummary(
        results=results,
        resolutions=resolutions,
        initial_glossary_version=initial.version,
        final_glossary_version=updated.version,
        total_prompt_tokens=sum(result.prompt_tokens for result in results),
        total_completion_tokens=sum(result.completion_tokens for result in results),
        reconciled_chapters_count=len(reconciled),
    )
