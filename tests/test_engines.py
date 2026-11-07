"""Integration tests validating all three orchestration engines: Asyncio, LangGraph, and Google ADK."""

from __future__ import annotations

import asyncio
import unittest

from benchmarks.run_benchmark import OzUnifiedBenchmarkProvider
from lexicon.dataset import build_oz_dataset
from lexicon.evals import TranslationEvaluator
from lexicon.glossary import GlossaryStore
from lexicon.pipeline import TranslationCoordinator


class TestAllOrchestrationEngines(unittest.TestCase):
    def setUp(self) -> None:
        (
            self.chapters,
            self.references,
            self.seed_terms,
            self.seed_characters,
            self.ground_truth_terms,
        ) = build_oz_dataset()
        self.provider = OzUnifiedBenchmarkProvider()
        self.evaluator = TranslationEvaluator(
            GlossaryStore(
                terms=self.ground_truth_terms,
                characters=self.seed_characters,
            ).snapshot()
        )

    def test_engines_produce_the_same_results(self) -> None:
        async def run() -> None:
            expected = None
            for engine in ("asyncio", "langgraph", "adk"):
                with self.subTest(engine=engine):
                    store = GlossaryStore(terms=self.seed_terms, characters=self.seed_characters)
                    coordinator = TranslationCoordinator(store, self.provider, engine=engine)
                    summary = await coordinator.translate_batch(list(reversed(self.chapters)))
                    translations = {
                        result.chapter.number: result.final_translation for result in summary.results
                    }
                    metrics = self.evaluator.evaluate(self.chapters, translations, self.references)

                    self.assertEqual(len(summary.resolutions), 1)
                    self.assertEqual(summary.resolutions[0].canonical_target, "gorrete de ouro")
                    self.assertEqual(metrics.glossary_adherence_rate, 100.0)
                    self.assertEqual(metrics.lexical_drift_bits, 0.0)
                    self.assertEqual(metrics.gender_agreement_accuracy, 100.0)
                    self.assertEqual(metrics.chrf_score, 100.0)
                    self.assertEqual(summary.final_glossary_version, store.version)
                    if expected is None:
                        expected = summary
                    self.assertEqual(summary, expected)

                    empty = await coordinator.translate_batch([])
                    self.assertEqual(empty.results, [])
                    self.assertEqual(empty.initial_glossary_version, store.version)
                    self.assertEqual(empty.final_glossary_version, store.version)

        asyncio.run(run())

    def test_reconciliation_failure_preserves_glossary_and_allows_retry(self) -> None:
        class FailingProvider(OzUnifiedBenchmarkProvider):
            fail = True

            async def reconcile_chapter(self, chapter, provisional_text, resolutions, context_directive):
                if self.fail:
                    raise RuntimeError("Falha simulada na reconciliação")
                return await super().reconcile_chapter(
                    chapter, provisional_text, resolutions, context_directive,
                )

        async def run(engine) -> None:
            provider = FailingProvider()
            store = GlossaryStore(terms=self.seed_terms, characters=self.seed_characters)
            coordinator = TranslationCoordinator(store, provider, engine=engine, max_concurrency=1)
            initial = store.snapshot()
            with self.assertRaisesRegex(RuntimeError, "Falha simulada"):
                await coordinator.translate_batch(self.chapters)
            self.assertEqual(store.snapshot(), initial)

            provider.fail = False
            summary = await coordinator.translate_batch(self.chapters)
            self.assertEqual(summary.initial_glossary_version, initial.version)
            self.assertGreater(summary.final_glossary_version, initial.version)
            self.assertEqual(summary.final_glossary_version, store.version)
            self.assertEqual(len(summary.results), len(self.chapters))

        for engine in ("asyncio", "langgraph", "adk"):
            with self.subTest(engine=engine):
                asyncio.run(run(engine))

    def test_concurrency_limits_both_translation_and_reconciliation(self) -> None:
        class CountingProvider(OzUnifiedBenchmarkProvider):
            def __init__(self) -> None:
                self.active = 0
                self.peak = 0

            async def measure(self, operation, *args):
                self.active += 1
                self.peak = max(self.peak, self.active)
                try:
                    await asyncio.sleep(0)
                    return await operation(*args)
                finally:
                    self.active -= 1

            async def translate_chapter(self, chapter, context_directive):
                return await self.measure(super().translate_chapter, chapter, context_directive)

            async def reconcile_chapter(self, chapter, provisional_text, resolutions, context_directive):
                return await self.measure(
                    super().reconcile_chapter, chapter, provisional_text, resolutions, context_directive,
                )

        for engine in ("asyncio", "langgraph", "adk"):
            for limit in (1, 2):
                with self.subTest(engine=engine, limit=limit):
                    provider = CountingProvider()
                    store = GlossaryStore(terms=self.seed_terms, characters=self.seed_characters)
                    coordinator = TranslationCoordinator(
                        store, provider, engine=engine, max_concurrency=limit,
                    )
                    summary = asyncio.run(coordinator.translate_batch(self.chapters))
                    self.assertGreater(summary.reconciled_chapters_count, 0)
                    self.assertEqual(provider.peak, limit)
            with self.subTest(engine=engine, limit=0):
                with self.assertRaises(ValueError):
                    TranslationCoordinator(GlossaryStore(), self.provider, engine=engine, max_concurrency=0)


if __name__ == "__main__":
    unittest.main()
