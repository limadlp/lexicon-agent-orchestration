"""Bounded paragraph translation for local models served by LM Studio."""

from __future__ import annotations

import asyncio
import json
import re
import textwrap
import time
import urllib.error
import urllib.request
from difflib import SequenceMatcher

from lexicon.models import ChapterDocument
from lexicon.contracts import TranslationResponse
from lexicon.providers.openai_compatible import OpenAICompatibleProvider


BLOCK_SEPARATOR = re.compile(r"(\n[ \t]*\n(?:[ \t]*\n)*)")


def _words(text: str) -> list[str]:
    return re.findall(r"[^\W\d_]+", text.casefold())


def _check_translation(source: str, target: str) -> None:
    original = _words(source)
    translated = _words(target)
    if not translated:
        if target.strip() != source.strip() or original:
            raise ValueError("bloco vazio ou sem texto")
        return
    if len(original) >= 12 and not 0.55 <= len(translated) / len(original) <= 1.9:
        raise ValueError("comprimento incompatível com o original")
    copied = SequenceMatcher(None, original, translated, autojunk=False).find_longest_match().size
    if copied >= 8 or (len(original) >= 4 and original == translated):
        raise ValueError("sequência longa copiada do inglês")


class LMStudioTranslator(OpenAICompatibleProvider):
    """Translate one paragraph per request while chapters run in parallel."""

    block_budget = 1500

    def __init__(self, **kwargs):
        super().__init__(
            **kwargs, timeout_seconds=180,
        )

    async def _translate_plain(
        self, prompt: str, *, retry: bool = False, system_prompt: str | None = None,
    ) -> dict:
        instruction = system_prompt or (
            "Traduza integralmente do inglês para português brasileiro. "
            "Siga o glossário e preserve a formatação. Responda somente com a tradução."
        )
        if retry:
            instruction += " Confira cada frase e mantenha todo o conteúdo do original."
        body = {
            "model": self.model,
            "system_prompt": instruction,
            "input": prompt,
            "temperature": self.temperature,
            "reasoning": "off",
            "max_output_tokens": 1200,
            "store": False,
        }
        endpoint = self.base_url.removesuffix("/v1") + "/api/v1/chat"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        started = time.perf_counter()
        for attempt in range(self.max_retries):
            request = urllib.request.Request(endpoint, json.dumps(body).encode(), headers)
            try:
                data = await asyncio.to_thread(self._read_response, request)
                break
            except urllib.error.HTTPError as exc:
                if exc.code == 400 and "reasoning" in body:
                    # Some models do not expose a reasoning setting at all.
                    del body["reasoning"]
                elif exc.code < 500 or attempt + 1 == self.max_retries:
                    raise
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt + 1 == self.max_retries:
                    raise RuntimeError("O servidor local não respondeu ao pedido de tradução") from exc
        else:
            raise RuntimeError("O servidor local rejeitou o pedido de tradução")
        stats = data["stats"]
        return {
            "content": "\n".join(item["content"] for item in data["output"] if item["type"] == "message"),
            "prompt_tokens": stats["input_tokens"],
            "completion_tokens": stats["total_output_tokens"],
            "latency_ms": (time.perf_counter() - started) * 1000,
            "finish_reason": "length" if stats["total_output_tokens"] >= body["max_output_tokens"] - 1 else "stop",
        }

    async def translate_chapter(
        self, chapter: ChapterDocument, context_directive: str,
    ) -> TranslationResponse:
        parts = BLOCK_SEPARATOR.split(chapter.content)
        context = f"Glossário e personagens:\n{context_directive}\n\n"
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "latency_ms": 0.0}

        async def translate(source: str, *, retry: bool = False) -> str:
            payload = await self._translate_plain(
                f"{context}Texto de origem:\n{source}", retry=retry,
            )
            for key in usage:
                usage[key] += payload[key]
            try:
                if payload["finish_reason"] == "length":
                    raise ValueError("resposta cortada no limite de saída")
                target = payload["content"].strip()
                _check_translation(source, target)
                return re.sub(r"\n[ \t]*\n+", "\n", target)
            except ValueError as exc:
                if not retry:
                    return await translate(source, retry=True)
                raise RuntimeError(f"{chapter.filename}: {exc}") from exc

        for index in range(0, len(parts), 2):
            paragraph = parts[index].strip()
            if not _words(paragraph):
                continue
            heading = re.match(r"^(#{1,6})\s+", paragraph)
            if heading:
                paragraph = paragraph[heading.end():]
            pieces = (
                textwrap.wrap(paragraph, self.block_budget, break_long_words=False, break_on_hyphens=False)
                if len(paragraph) > self.block_budget else [paragraph]
            )
            target = " ".join([await translate(piece) for piece in pieces])
            if heading:
                target = heading[1] + " " + re.sub(r"^#{1,6}\s+", "", target)
            parts[index] = target

        return TranslationResponse(translated_text="".join(parts), proposals=[], **usage)
