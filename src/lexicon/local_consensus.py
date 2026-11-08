"""Collect lexical proposals and revise affected paragraphs in LM Studio."""

from __future__ import annotations

import json
from dataclasses import replace

from lexicon.glossary import GlossaryStore, english_inflections, normalize_key, occurs_in_text, portuguese_inflections
from lexicon.models import ChapterDocument, ConflictResolution, TermProposal
from lexicon.contracts import TranslationResponse
from lexicon.providers.local import BLOCK_SEPARATOR, LMStudioTranslator


class LMStudioConsensusTranslator(LMStudioTranslator):
    def __init__(self, store: GlossaryStore, **kwargs):
        super().__init__(**kwargs)
        self.store = store

    async def collect_proposals(
        self, chapter: ChapterDocument, translated: str,
    ) -> TranslationResponse:
        snapshot = self.store.snapshot()
        pairs = zip(BLOCK_SEPARATOR.split(chapter.content)[::2], BLOCK_SEPARATOR.split(translated)[::2], strict=True)
        proposals = []
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "latency_ms": 0.0}
        for original, target in pairs:
            if original.lstrip().startswith("#"):
                continue
            for term in self.store.retrieve_chapter_context(snapshot, original)[0]:
                if occurs_in_text(target, portuguese_inflections(term.target)):
                    observed = term.target
                else:
                    source_forms = [form for form in (term.source, *term.aliases)
                                    if occurs_in_text(original, english_inflections(form))]
                    response = await self._translate_plain(
                        json.dumps({
                            "termo_em_inglês": source_forms,
                            "parágrafo_em_inglês": original,
                            "parágrafo_traduzido": target,
                        }, ensure_ascii=False),
                        system_prompt=(
                            "Localize no parágrafo traduzido a expressão que corresponde ao "
                            "termo em inglês, inclusive se a tradução usar um sinônimo. "
                            "Responda somente com o trecho exato do parágrafo traduzido, "
                            "sem aspas, artigo inicial ou explicação. Se o termo tiver sido "
                            "omitido, responda OMITIDO."
                        ),
                    )
                    for key in usage:
                        usage[key] += response[key]
                    observed = response["content"].strip().strip('\"“”`')
                    if not observed or not occurs_in_text(target, {normalize_key(observed)}):
                        continue
                proposals.append(TermProposal(chapter.number, "termo", term.source, observed))

        return TranslationResponse(translated, proposals, **usage)

    async def translate_chapter(self, chapter: ChapterDocument, context_directive: str) -> TranslationResponse:
        initial = await super().translate_chapter(chapter, context_directive)
        review = await self.collect_proposals(chapter, initial.translated_text)
        return replace(
            initial,
            proposals=review.proposals,
            prompt_tokens=initial.prompt_tokens + review.prompt_tokens,
            completion_tokens=initial.completion_tokens + review.completion_tokens,
            latency_ms=initial.latency_ms + review.latency_ms,
        )

    async def reconcile_chapter(
        self, chapter: ChapterDocument, provisional_text: str,
        resolutions: list[ConflictResolution], context_directive: str,
    ) -> TranslationResponse:
        original = BLOCK_SEPARATOR.split(chapter.content)
        translated = BLOCK_SEPARATOR.split(provisional_text)
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "latency_ms": 0.0}
        snapshot = self.store.snapshot()
        for index in range(0, len(original), 2):
            if original[index].lstrip().startswith("#"):
                continue
            active_terms = {
                normalize_key(term.source)
                for term in self.store.retrieve_chapter_context(snapshot, original[index])[0]
            }
            applicable = [
                item for item in resolutions if item.normalized_source in active_terms and any(
                    occurs_in_text(translated[index], portuguese_inflections(form))
                    for form in item.rejected_targets
                )
            ]
            if not applicable:
                continue
            corrections = "\n".join(
                f"- Use exatamente {item.canonical_target!r} para {item.normalized_source!r}."
                for item in applicable
            )
            response = await super().translate_chapter(
                replace(chapter, content=original[index]), f"{context_directive}\n{corrections}",
            )
            for key in usage:
                usage[key] += getattr(response, key)
            if all(occurs_in_text(response.translated_text, portuguese_inflections(item.canonical_target))
                   for item in applicable):
                translated[index] = response.translated_text
        return TranslationResponse("".join(translated), [], **usage)
