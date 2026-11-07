"""Unit and integration tests for Lexicon pipeline, consensus protocol, and evaluators.

All test fixtures use public domain material from L. Frank Baum's
The Wonderful Wizard of Oz (1900).
"""

from __future__ import annotations

from collections import Counter
import unittest

from lexicon.contracts import TranslationResponse
from lexicon.pipeline import TranslationCoordinator
from lexicon.evals import TranslationEvaluator, compute_chrf, shannon_entropy_bits
from lexicon.glossary import GlossaryStore
from lexicon.models import (
    CharacterEntry,
    ChapterDocument,
    TermEntry,
    TermProposal,
)


class TestGlossaryAndConsensus(unittest.TestCase):
    def test_symbolic_retrieval_matches_inflections(self) -> None:
        store = GlossaryStore(
            terms=[
                TermEntry(source="winged monkey", target="macaco alado"),
                TermEntry(source="wizard", target="mágico"),
                TermEntry(source="silver shoe", target="sapato de prata"),
            ],
            characters=[
                CharacterEntry(
                    name="Glinda",
                    target_name="Glinda",
                    gender="F",
                    english_referents=("the ruler of the south", "the ruler"),
                )
            ],
        )
        snapshot = store.snapshot()
        chapter_text = "The ruler watched the winged monkeys while two wizards waited."
        terms, chars = store.retrieve_chapter_context(snapshot, chapter_text)

        self.assertEqual([t.source for t in terms], ["winged monkey", "wizard"])
        self.assertEqual([c.name for c in chars], ["Glinda"])

    def test_single_writer_earliest_chapter_wins_regardless_of_order(self) -> None:
        store = GlossaryStore()
        # Even if chapter 8 finishes first and submits its proposal before chapter 3,
        # chapter 3 must win deterministically.
        proposals = [
            TermProposal(
                chapter_number=8,
                kind="termo",
                source="Golden Cap",
                proposed_target="chapéu dourado",
            ),
            TermProposal(
                chapter_number=3,
                kind="termo",
                source="golden cap",
                proposed_target="gorrete de ouro",
            ),
            TermProposal(
                chapter_number=5,
                kind="termo",
                source="Golden Cap",
                proposed_target="coroa áurea",
            ),
        ]
        resolutions = store.consolidate_proposals(proposals)
        self.assertEqual(len(resolutions), 1)
        res = resolutions[0]
        self.assertEqual(res.canonical_target, "gorrete de ouro")
        self.assertEqual(res.affected_chapters, (5, 8))
        self.assertIn("chapéu dourado", res.rejected_targets)
        self.assertIn("coroa áurea", res.rejected_targets)
        self.assertEqual(store.version, 2)

    def test_canonical_entry_always_overrides_new_proposals(self) -> None:
        store = GlossaryStore(
            terms=[TermEntry(source="wizard", target="mágico")]
        )
        proposals = [
            TermProposal(
                chapter_number=1,
                kind="termo",
                source="Wizard",
                proposed_target="bruxo",
            )
        ]
        resolutions = store.consolidate_proposals(proposals)
        self.assertEqual(len(resolutions), 1)
        self.assertEqual(resolutions[0].canonical_target, "mágico")
        self.assertEqual(resolutions[0].affected_chapters, (1,))
        self.assertEqual(resolutions[0].winning_reason, "established_in_canonical_glossary")


class TestBatchValidation(unittest.IsolatedAsyncioTestCase):
    async def test_duplicate_chapter_numbers_fail_before_translation(self):
        class UnusedProvider:
            async def translate_chapter(self, chapter, context_directive):
                raise AssertionError("Duplicate numbers must fail before calling the provider")

        coordinator = TranslationCoordinator(GlossaryStore(), UnusedProvider())
        chapters = [
            ChapterDocument(1, "one.md", "One", "The wizard"),
            ChapterDocument(1, "two.md", "Two", "The wizard"),
        ]
        with self.assertRaisesRegex(ValueError, "únicos"):
            await coordinator.translate_batch(chapters)

    async def test_failed_reconciliation_preserves_glossary_for_retry(self):
        class Provider:
            fail = True

            async def translate_chapter(self, chapter, context_directive):
                target = "mágico" if chapter.number == 1 else "bruxo"
                proposal = TermProposal(chapter.number, "termo", "wizard", target)
                return TranslationResponse(target, [proposal])

            async def reconcile_chapter(self, chapter, provisional_text, resolutions, context_directive):
                if self.fail:
                    raise RuntimeError("revision failed")
                return TranslationResponse(resolutions[0].canonical_target, [])

        store = GlossaryStore()
        provider = Provider()
        coordinator = TranslationCoordinator(store, provider)
        chapters = [
            ChapterDocument(1, "one.md", "One", "The wizard"),
            ChapterDocument(2, "two.md", "Two", "The wizard"),
        ]
        original = store.snapshot()
        with self.assertRaisesRegex(RuntimeError, "revision failed"):
            await coordinator.translate_batch(chapters)
        self.assertEqual(store.snapshot(), original)

        provider.fail = False
        result = await coordinator.translate_batch(chapters)
        self.assertEqual(result.results[1].final_translation, "mágico")
        self.assertEqual(store.snapshot().terms[0].target, "mágico")
        self.assertEqual(store.version, original.version + 1)


class TestEvaluationMetrics(unittest.TestCase):
    def test_shannon_entropy_for_lexical_drift(self) -> None:
        consistent = Counter({"mágico": 10})
        self.assertEqual(shannon_entropy_bits(consistent), 0.0)

        drifted = Counter({"mágico": 5, "bruxo": 5})
        self.assertAlmostEqual(shannon_entropy_bits(drifted), 1.0, places=4)

        three_way_drift = Counter({"mágico": 4, "bruxo": 4, "sábio": 4})
        self.assertAlmostEqual(shannon_entropy_bits(three_way_drift), 1.585, places=3)

    def test_chrf_computation(self) -> None:
        ref = "A governante do Sul estava exausta após comandar os macacos alados."
        identical = compute_chrf(ref, ref)
        self.assertEqual(identical, 100.0)

        close = compute_chrf(
            "O governante do Sul estava exausto após comandar os símios voadores.",
            ref,
        )
        self.assertGreater(close, 50.0)
        self.assertLess(close, 95.0)

    def test_evaluator_detects_term_and_gender_violations(self) -> None:
        store = GlossaryStore(
            terms=[
                TermEntry(
                    source="wizard",
                    target="mágico",
                    forbidden_synonyms=("bruxo", "mago", "sábio"),
                )
            ],
            characters=[
                CharacterEntry(
                    name="Glinda",
                    target_name="Glinda",
                    gender="F",
                    english_referents=("the ruler of the south", "the ruler"),
                    expected_markers_pt=("a governante",),
                    forbidden_markers_pt=("o governante",),
                )
            ],
        )
        chapter = ChapterDocument(
            number=1,
            filename="001.md",
            title="Ch 1",
            content="The ruler of the South spoke to the wizard.",
        )
        bad_translation = {1: "O governante do Sul falou com o bruxo."}
        good_translation = {1: "A governante do Sul falou com o mágico."}

        evaluator = TranslationEvaluator(store.snapshot())
        bad_metrics = evaluator.evaluate([chapter], bad_translation)
        self.assertEqual(bad_metrics.glossary_adherence_rate, 0.0)
        self.assertEqual(bad_metrics.gender_agreement_accuracy, 0.0)
        self.assertEqual(len(bad_metrics.term_violations), 1)
        self.assertEqual(len(bad_metrics.gender_violations), 1)

        good_metrics = evaluator.evaluate([chapter], good_translation)
        self.assertEqual(good_metrics.glossary_adherence_rate, 100.0)
        self.assertEqual(good_metrics.gender_agreement_accuracy, 100.0)


if __name__ == "__main__":
    unittest.main()
