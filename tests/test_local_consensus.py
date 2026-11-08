import unittest
from unittest.mock import AsyncMock

from lexicon.glossary import GlossaryStore
from lexicon.local_consensus import LMStudioConsensusTranslator
from lexicon.models import ChapterDocument, TermEntry


class TestLocalConsensus(unittest.IsolatedAsyncioTestCase):
    async def test_only_observed_proposals_trigger_selective_revision(self):
        store = GlossaryStore(terms=[TermEntry("Silver Shoes", "Sapatos de Prata")])
        provider = LMStudioConsensusTranslator(store=store, model="test")
        chapter = ChapterDocument(1, "test.md", "Test", "She took the Silver Shoes.\n\nShe left.")
        initial = "Ela pegou os sapatos prateados.\n\nEla partiu."
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "latency_ms": 1, "finish_reason": "stop"}
        provider._translate_plain = AsyncMock(side_effect=[
            {**usage, "content": "Sapatos Prateados"},
            {**usage, "content": "Ela pegou os Sapatos de Prata."},
        ])
        extraction = await provider.collect_proposals(chapter, initial)
        conflicts = store.consolidate_proposals(extraction.proposals)
        self.assertEqual(len(conflicts), 1)
        revised = await provider.reconcile_chapter(chapter, initial, conflicts, "")
        self.assertEqual(revised.translated_text, "Ela pegou os Sapatos de Prata.\n\nEla partiu.")
        self.assertEqual(provider._translate_plain.await_count, 2)
