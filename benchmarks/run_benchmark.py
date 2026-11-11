"""Compare translation configurations on three Oz-inspired test passages.

The default simulated benchmark compares:
1. Stateless Zero-Shot (Baseline)
2. Static Context (No Reconciliation)
3. Lexicon 4-Phase Consensus — Pure Asyncio Engine
4. Lexicon 4-Phase Consensus — LangGraph StateGraph Engine (Send API)
5. Lexicon 4-Phase Consensus — Google ADK Engine (SequentialAgent + ParallelAgent)

Pass --lm-studio-model to run the glossary-guided configuration with a loaded local model.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lexicon.dataset import build_oz_dataset
from lexicon.evals import TranslationEvaluator
from lexicon.glossary import GlossaryStore
from lexicon.models import (
    ChapterDocument,
    ConflictResolution,
    TermProposal,
)
from lexicon.pipeline import TranslationCoordinator
from lexicon.config import load_dotenv
from lexicon.providers import OpenAICompatibleProvider, TranslationResponse


class OzUnifiedBenchmarkProvider:
    """Deterministic simulation provider modeling stateless drift vs context-guided consensus."""

    async def translate_chapter(
        self,
        chapter: ChapterDocument,
        context_directive: str,
    ) -> TranslationResponse:
        has_context = "Terminologia Canônica Obrigatória" in context_directive
        if not has_context:
            stateless = {
                1: (
                    "# Capítulo 1: O Conselho com os Munchkins\n\n"
                    "O conselheiro do Norte cumprimentou Dorothy perto da estrada de tijolos amarelos. "
                    "O conselheiro foi cauteloso após observar os símios voadores se espalharem pelas "
                    "colinas do leste, alertando que apenas o grande bruxo poderia ajudar.\n\n"
                    "Antes de partir, o jovem viajante examinou o antigo gorrete de ouro. "
                    "O viajante estava exausto do ciclone, mas ajustou as lentes verdes "
                    "guardadas dentro da cesta."
                ),
                2: (
                    "# Capítulo 2: O Portão da Cidade das Esmeraldas\n\n"
                    "No limiar do palácio, Dorothy segurou o místico chapéu dourado com firmeza. "
                    "O jovem viajante estava determinado a falar com o sábio antes que os macacos com asas "
                    "retornassem da floresta ocidental.\n\n"
                    "O governante do Sul enviou uma mensagem pela corte. O governante foi paciente, "
                    "lembrando cada cidadão de usar os visores de esmeralda trancados dentro da cidadela."
                ),
                3: (
                    "# Capítulo 3: O Chamado dos Macacos Alados\n\n"
                    "O governante do Sul inspecionou as inscrições entalhadas dentro da coroa áurea. "
                    "O governante estava confiante de que um único feiticeiro poderia comandar os macacos alados "
                    "três vezes sem quebrar os óculos de esmeralda.\n\n"
                    "O conselheiro do Norte assentiu em concordância. O conselheiro ficou grato "
                    "por o jovem viajante estar seguro."
                ),
            }
            return TranslationResponse(
                translated_text=stateless[chapter.number],
                proposals=[],
                prompt_tokens=430,
                completion_tokens=115,
                latency_ms=180.0,
            )

        provisional = {
            1: (
                "# Capítulo 1: O Conselho com os Munchkins\n\n"
                "A conselheira do Norte cumprimentou Dorothy perto da estrada de tijolos amarelos. "
                "A conselheira foi cautelosa após observar os macacos alados se espalharem pelas "
                "colinas do leste, alertando que apenas o grande mágico poderia ajudar.\n\n"
                "Antes de partir, a jovem viajante examinou o antigo gorrete de ouro. "
                "A viajante estava exausta do ciclone, mas ajustou os óculos de esmeralda "
                "guardados dentro da cesta."
            ),
            2: (
                "# Capítulo 2: O Portão da Cidade das Esmeraldas\n\n"
                "No limiar do palácio, Dorothy segurou o místico chapéu dourado com firmeza. "
                "A jovem viajante estava determinada a falar com o mágico antes que os macacos alados "
                "retornassem da floresta ocidental.\n\n"
                "A governante do Sul enviou uma mensagem pela corte. A governante foi paciente, "
                "lembrando cada cidadão de usar os óculos de esmeralda trancados dentro da cidadela."
            ),
            3: (
                "# Capítulo 3: O Chamado dos Macacos Alados\n\n"
                "A governante do Sul inspecionou as inscrições entalhadas dentro da coroa áurea. "
                "A governante estava confiante de que um único mágico poderia comandar os macacos alados "
                "três vezes sem quebrar os óculos de esmeralda.\n\n"
                "A conselheira do Norte assentiu em concordância. A conselheira ficou grata "
                "por a jovem viajante estar segura."
            ),
        }
        proposed_target = {1: "gorrete de ouro", 2: "chapéu dourado", 3: "coroa áurea"}[
            chapter.number
        ]
        return TranslationResponse(
            translated_text=provisional[chapter.number],
            proposals=[
                TermProposal(
                    chapter_number=chapter.number,
                    kind="termo",
                    source="golden cap",
                    proposed_target=proposed_target,
                    notes="Artefato de invocação em Oz.",
                )
            ],
            prompt_tokens=560,
            completion_tokens=165,
            latency_ms=210.0,
        )

    async def reconcile_chapter(
        self,
        chapter: ChapterDocument,
        provisional_text: str,
        resolutions: list[ConflictResolution],
        context_directive: str,
    ) -> TranslationResponse:
        revised = provisional_text
        for res in resolutions:
            if res.normalized_source == "golden cap":
                revised = revised.replace(
                    "o místico chapéu dourado", f"o místico {res.canonical_target}"
                )
                revised = revised.replace(
                    "dentro da coroa áurea", f"dentro do {res.canonical_target}"
                )
        return TranslationResponse(
            translated_text=revised,
            proposals=[],
            prompt_tokens=310,
            completion_tokens=115,
            latency_ms=130.0,
        )


async def run_all(lm_studio_model: str | None = None) -> int:
    chapters, references, seed_terms, seed_characters, ground_truth_terms = build_oz_dataset()
    provider = OzUnifiedBenchmarkProvider()
    ground_truth_store = GlossaryStore(terms=ground_truth_terms, characters=seed_characters)
    evaluator = TranslationEvaluator(ground_truth_store.snapshot())

    benchmark_runs = [
        ("1. Stateless Zero-Shot (Baseline)", "asyncio", False, False),
        ("2. Static Context (Sem Reconciliação)", "asyncio", True, False),
        ("3. Lexicon 4-Phase [Pure Asyncio]", "asyncio", True, True),
        ("4. Lexicon 4-Phase [LangGraph StateGraph]", "langgraph", True, True),
        ("5. Lexicon 4-Phase [Google ADK Hierarchy]", "adk", True, True),
    ]
    if lm_studio_model:
        provider = OpenAICompatibleProvider(
            model=lm_studio_model,
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            timeout_seconds=300,
            max_retries=1,
            max_tokens=3072,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "translation",
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "translation": {"type": "string"},
                            "proposals": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        key: {"type": "string"}
                                        for key in ("kind", "source", "proposed_target", "gender", "notes")
                                    } | {"aliases": {"type": "array", "items": {"type": "string"}}},
                                    "required": ["kind", "source", "proposed_target"],
                                },
                            },
                        },
                        "required": ["translation"],
                    },
                },
            },
        )
        benchmark_runs = [benchmark_runs[1]]
        print(f"LM Studio: {lm_studio_model}")

    label_width = max(len(label) for label, *_ in benchmark_runs) + len(" (fallback)")
    header = (
        f"{'Configuração & Engine':<{label_width}} | {'Aderência':>9} | {'Drift (H)':>9} | "
        f"{'Gênero':>8} | {'chrF':>6} | {'Tokens':>7} | {'Tempo (s)':>9} | {'Conflitos':>9}"
    )
    table_width = len(header)
    print("=" * table_width)
    print("BENCHMARK MULTI-ENGINE (THE WONDERFUL WIZARD OF OZ, 1900) — CONSISTÊNCIA, GÊNERO & CHRF")
    print("=" * table_width)
    print(header)
    print("-" * table_width)

    for label, engine_type, use_ctx, use_rec in benchmark_runs:
        store = GlossaryStore(terms=list(seed_terms), characters=list(seed_characters))
        coordinator = TranslationCoordinator(
            store=store,
            provider=provider,
            engine=engine_type,
            enable_context_injection=use_ctx,
            enable_reconciliation=use_rec,
            max_concurrency=1 if lm_studio_model else 4,
        )
        if "fallback" in coordinator.runtime:
            label += " (fallback)"
        started = perf_counter()
        summary = await coordinator.translate_batch(chapters)
        if len(summary.results) != len(chapters) or any(
            not result.final_translation.strip() for result in summary.results
        ):
            raise ValueError("Benchmark returned missing or empty translations")
        elapsed = perf_counter() - started
        translations = {r.chapter.number: r.final_translation for r in summary.results}
        metrics = evaluator.evaluate(chapters, translations, references)
        tokens = summary.total_prompt_tokens + summary.total_completion_tokens

        print(
            f"{label:<{label_width}} | "
            f"{metrics.glossary_adherence_rate:>8.1f}% | "
            f"{metrics.lexical_drift_bits:>6.4f} b | "
            f"{metrics.gender_agreement_accuracy:>7.1f}% | "
            f"{(metrics.chrf_score or 0.0):>6.2f} | "
            f"{tokens:>7d} | {elapsed:>9.2f} | {len(summary.resolutions):>9d}",
            flush=True,
        )

    print("=" * table_width)
    return 0


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lm-studio-model",
        help="Run the glossary-guided configuration with this loaded local model",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run_all(args.lm_studio_model)))
