"""Model-agnostic LLM provider abstraction supporting cloud APIs and local inference servers."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import urllib.error
import urllib.request

from lexicon.contracts import TranslationResponse
from lexicon.models import ChapterDocument, ConflictResolution, TermProposal


SYSTEM_TRANSLATE_PROMPT = """Você é um tradutor literário especialista (EN -> PT-BR).
Sua tarefa é traduzir o capítulo integralmente para português brasileiro natural, mantendo a formatação Markdown.

Regras estritas:
1. Siga rigorosamente todas as entradas em 'Restrições de Gênero e Personagens Ativos' e 'Terminologia Canônica Obrigatória'.
2. O inglês frequentemente omite gênero em títulos, cargos e adjetivos ('the director was exhausted', 'the mage'). Em português, flexione artigos, pronomes e adjetivos de acordo com o gênero registrado do personagem.
3. Para qualquer novo termo recorrente do mundo ou personagem relevante que NÃO conste do contexto recebido, registre uma proposta estruturada.
4. Responda EXCLUSIVAMENTE em JSON válido com o formato:
{
  "translation": "<texto completo traduzido em markdown>",
  "proposals": [
    {
      "kind": "termo" | "personagem",
      "source": "<termo original em inglês>",
      "proposed_target": "<tradução proposta em PT-BR>",
      "gender": "M" | "F" | "N" | "",
      "aliases": ["<variantes>"],
      "notes": "<justificativa breve>"
    }
  ]
}"""

SYSTEM_RECONCILE_PROMPT = """Você é um revisor literário especialista (EN -> PT-BR).
Ajuste a tradução provisória do capítulo para substituir os termos rejeitados pelas formas canônicas consolidadas, corrigindo artigos, preposições, flexões de gênero/número e concordância ao redor.
Responda EXCLUSIVAMENTE em JSON válido com o formato:
{
  "translation": "<texto completo revisado em markdown>"
}"""


class OpenAICompatibleProvider:
    """Universal client for OpenAI, Ollama, vLLM, LiteLLM Proxy, and OpenRouter endpoints."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.2,
        timeout_seconds: float = 120.0,
        max_retries: int = 3,
        response_format: dict | None = None,
        max_tokens: int | None = None,
    ) -> None:
        gemini_key = os.environ.get("GEMINI_API_KEY")
        openai_key = os.environ.get("OPENAI_API_KEY")
        generic_key = os.environ.get("LLM_API_KEY")

        resolved_key = api_key or generic_key or openai_key or gemini_key or "local"
        self.api_key = resolved_key

        # Resolve Base URL
        env_base_url = os.environ.get("LLM_BASE_URL")
        if base_url:
            self.base_url = base_url.rstrip("/")
        elif env_base_url:
            self.base_url = env_base_url.rstrip("/")
        elif gemini_key and not openai_key:
            self.base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
        else:
            self.base_url = "https://api.openai.com/v1"

        # Resolve Model
        env_model = os.environ.get("LLM_MODEL")
        if model:
            self.model = model
        elif env_model:
            self.model = env_model
        elif gemini_key and not openai_key:
            self.model = "gemini-2.5-flash"
        else:
            self.model = "gpt-4o-mini"

        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.response_format = response_format if response_format is not None else {"type": "json_object"}
        self.max_tokens = max_tokens

    async def translate_chapter(
        self,
        chapter: ChapterDocument,
        context_directive: str,
    ) -> TranslationResponse:
        user_prompt = (
            f"## Contexto Lexical e de Gênero do Capítulo\n"
            f"{context_directive or 'Nenhuma restrição prévia.'}\n\n"
            f"## Capítulo Original ({chapter.filename})\n"
            f"{chapter.content}"
        )
        payload = await self._chat_completion(
            system_prompt=SYSTEM_TRANSLATE_PROMPT,
            user_prompt=user_prompt,
        )
        parsed = self._parse_json_content(payload["content"])
        raw_proposals = parsed.get("proposals", [])
        proposals = [
            TermProposal(
                chapter_number=chapter.number,
                kind="personagem" if p.get("kind") == "personagem" else "termo",
                source=str(p.get("source", "")).strip(),
                proposed_target=str(p.get("proposed_target", "")).strip(),
                gender=str(p.get("gender", "")).strip(),
                aliases=tuple(p.get("aliases", [])),
                notes=str(p.get("notes", "")).strip(),
            )
            for p in raw_proposals
            if p.get("source") and p.get("proposed_target")
        ]
        return TranslationResponse(
            translated_text=str(parsed.get("translation", "")),
            proposals=proposals,
            prompt_tokens=payload["prompt_tokens"],
            completion_tokens=payload["completion_tokens"],
            latency_ms=payload["latency_ms"],
        )

    async def reconcile_chapter(
        self,
        chapter: ChapterDocument,
        provisional_text: str,
        resolutions: list[ConflictResolution],
        context_directive: str,
    ) -> TranslationResponse:
        resolution_lines = [
            f"- '{r.normalized_source}': substituir {list(r.rejected_targets)} pela forma canônica '{r.canonical_target}'"
            for r in resolutions
        ]
        user_prompt = (
            f"## Resoluções de Conflito Obrigatórias\n"
            + "\n".join(resolution_lines)
            + f"\n\n## Contexto Consolidado\n{context_directive}\n\n"
            f"## Tradução Provisória\n{provisional_text}"
        )
        payload = await self._chat_completion(
            system_prompt=SYSTEM_RECONCILE_PROMPT,
            user_prompt=user_prompt,
        )
        parsed = self._parse_json_content(payload["content"])
        return TranslationResponse(
            translated_text=str(parsed.get("translation", provisional_text)),
            proposals=[],
            prompt_tokens=payload["prompt_tokens"],
            completion_tokens=payload["completion_tokens"],
            latency_ms=payload["latency_ms"],
        )

    async def _chat_completion(self, system_prompt: str, user_prompt: str) -> dict:
        body = json.dumps(
            {
                "model": self.model,
                "temperature": self.temperature,
                "response_format": self.response_format,
                **({"max_tokens": self.max_tokens} if self.max_tokens is not None else {}),
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
        ).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            start = time.perf_counter()
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=body,
                headers=headers,
                method="POST",
            )
            try:
                data = await asyncio.to_thread(self._read_response, req)
                latency_ms = (time.perf_counter() - start) * 1000.0
                choice = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                return {
                    "content": choice,
                    "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                    "completion_tokens": int(usage.get("completion_tokens", 0)),
                    "latency_ms": latency_ms,
                    "finish_reason": data["choices"][0].get("finish_reason"),
                }
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt + 1 < self.max_retries:
                    await asyncio.sleep(0.5 * (2**attempt))

        raise RuntimeError(f"LLM completion failed after {self.max_retries} retries: {last_error}")

    def _read_response(self, request: urllib.request.Request) -> dict:
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            return json.load(response)

    @staticmethod
    def _parse_json_content(raw_content: str) -> dict:
        cleaned = raw_content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.DOTALL)
        return json.loads(cleaned)
