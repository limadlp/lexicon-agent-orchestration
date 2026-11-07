import unittest
from unittest.mock import AsyncMock

from lexicon.local_translation import LMStudioTranslator
from lexicon.models import ChapterDocument


SOURCE = "Dorothy looked out of the window and saw the yellow road stretching into the distance."
TARGET = "Dorothy olhou pela janela e viu a estrada amarela se estender ao longe."


def response(text, finish_reason="stop"):
    return {
        "content": text,
        "finish_reason": finish_reason,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "latency_ms": 1,
    }


class TestLocalTranslation(unittest.IsolatedAsyncioTestCase):
    async def test_retries_a_copied_paragraph_and_preserves_order(self):
        translator = LMStudioTranslator(model="test")
        translator._translate_plain = AsyncMock(side_effect=[
            response("Capítulo Um"),
            response(SOURCE),
            response(TARGET),
        ])
        chapter = ChapterDocument(1, "01.md", "Chapter One", f"## Chapter One\n\n{SOURCE}")
        result = await translator.translate_chapter(chapter, "")
        self.assertEqual(result.translated_text, f"## Capítulo Um\n\n{TARGET}")
        self.assertEqual(result.prompt_tokens, 30)
        self.assertEqual(translator._translate_plain.await_args.kwargs, {"retry": True})

    async def test_recovers_a_truncated_response(self):
        translator = LMStudioTranslator(model="test")
        translator._translate_plain = AsyncMock(side_effect=[
            response("Dorothy olhou", finish_reason="length"),
            response(TARGET),
        ])
        result = await translator.translate_chapter(ChapterDocument(1, "01.md", "", SOURCE), "")
        self.assertEqual(result.translated_text, TARGET)

    async def test_does_not_accept_untranslated_output_after_retry(self):
        translator = LMStudioTranslator(model="test")
        translator._translate_plain = AsyncMock(return_value=response(SOURCE))
        with self.assertRaisesRegex(RuntimeError, "01.md"):
            await translator.translate_chapter(ChapterDocument(1, "01.md", "", SOURCE), "")
        self.assertEqual(translator._translate_plain.await_count, 2)
