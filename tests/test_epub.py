"""Unit tests for Lexicon's native standalone EPUB extraction and packaging."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ebooklib import epub
from bs4 import BeautifulSoup

from lexicon.cli import build_parser
from lexicon.contracts import TranslationResponse
from lexicon.epub import extract_epub, build_epub
from lexicon.epub.builder import replace_paragraphs_in_xhtml


class TestEpubStandalone(unittest.TestCase):
    """Verify that Lexicon extracts and builds EPUBs without any external dependencies."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.work_path = Path(self.temp_dir.name)

        # Create a minimal test EPUB
        self.book = epub.EpubBook()
        self.book.set_identifier("lexicon-test-book-001")
        self.book.set_title("Test Book")
        self.book.set_language("en")

        c1 = epub.EpubHtml(title="Chapter 1", file_name="ch01.xhtml", lang="en")
        c1.content = "<h1>Chapter 1</h1><p>The Wonderful Wizard was famous.</p>"
        self.book.add_item(c1)

        self.book.toc = (epub.Link("ch01.xhtml", "Chapter 1", "ch1"),)
        self.book.add_item(epub.EpubNcx())
        self.book.add_item(epub.EpubNav())
        self.book.spine = [c1]

        self.source_epub_path = self.work_path / "source.epub"
        epub.write_epub(str(self.source_epub_path), self.book)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_extract_and_build_epub(self):
        en_dir = self.work_path / "en"
        chapters = extract_epub(self.source_epub_path, en_dir)

        self.assertGreaterEqual(len(chapters), 1)
        chapter = [c for c in chapters if "ch01" in c.filename or "Chapter" in c.title][0]
        self.assertIn("Wizard", chapter.content)

        # Translate in-memory simulation
        pt_dir = self.work_path / "pt"
        pt_dir.mkdir(parents=True, exist_ok=True)
        pt_file = pt_dir / chapter.filename
        pt_file.write_text("# Chapter 1\n\nO Mágico Maravilhoso era famoso.", encoding="utf-8")

        # Copy _chapters.json mapping
        mapping_file = en_dir / "_chapters.json"
        if mapping_file.exists():
            (pt_dir / "_chapters.json").write_text(mapping_file.read_text(encoding="utf-8"), encoding="utf-8")

        output_epub = self.work_path / "output_pt.epub"
        build_epub(
            markdown_dir=str(pt_dir),
            output_path=str(output_epub),
            title="Livro de Teste",
            source_epub_path=str(self.source_epub_path),
            language="pt-BR",
        )

        self.assertTrue(output_epub.exists())
        self.assertGreater(output_epub.stat().st_size, 1000)

    def test_translate_epub_command_builds_translated_book(self):
        class FakeProvider:
            async def translate_chapter(self, chapter, context_directive):
                return TranslationResponse("# Capítulo 1\n\nO Mágico era famoso.", [])

        output = self.work_path / "translated.epub"
        work_dir = self.work_path / "workspace"
        args = build_parser().parse_args([
            "translate-epub", str(self.source_epub_path), "--lm-studio",
            "--mode", "glossary", "--sections", "1", "1",
            "--work-dir", str(work_dir), "-o", str(output),
        ])
        with patch("lexicon.cli._translation_provider", return_value=FakeProvider()):
            self.assertEqual(args.func(args), 0)

        translated = epub.read_epub(str(output))
        chapter = translated.get_item_with_href("ch01.xhtml")
        self.assertIn("O Mágico era famoso.", chapter.get_content().decode())
        self.assertEqual(translated.get_metadata("DC", "language")[0][0], "pt-BR")
        self.assertTrue((work_dir / "pt" / "_chapters.json").is_file())

    def test_translate_epub_reflows_chapters_sharing_one_xhtml(self):
        chapter = self.book.get_item_with_href("ch01.xhtml")
        chapter.content = (
            '<p id="first">Chapter One</p><p>First scene.</p>'
            '<p id="second">Chapter Two</p><p>Second scene.</p>'
        )
        self.book.toc = (
            epub.Link("ch01.xhtml#first", "Chapter One", "first"),
            epub.Link("ch01.xhtml#second", "Chapter Two", "second"),
        )
        epub.write_epub(str(self.source_epub_path), self.book)

        class FakeProvider:
            async def translate_chapter(self, chapter, context_directive):
                text = chapter.content.replace("Chapter Two", "Capítulo Dois")
                return TranslationResponse(text.replace("Second scene.", "Segunda cena."), [])

        output = self.work_path / "reflowed.epub"
        work_dir = self.work_path / "workspace"
        args = build_parser().parse_args([
            "translate-epub", str(self.source_epub_path), "--lm-studio",
            "--sections", "2", "2", "--work-dir", str(work_dir), "-o", str(output),
        ])
        with patch("lexicon.cli._translation_provider", return_value=FakeProvider()):
            self.assertEqual(args.func(args), 0)

        translated = epub.read_epub(str(output))
        body = translated.get_item_with_href("chapter_001.xhtml").get_content().decode()
        self.assertIn("Segunda cena.", body)
        self.assertNotIn("First scene.", body)
        self.assertEqual(len(translated.toc), 1)

        # The standalone builder can consume the workspace artifacts directly.
        rebuilt = self.work_path / "rebuilt.epub"
        build_epub(str(work_dir / "pt"), str(rebuilt), "Livro", str(self.source_epub_path))
        book = epub.read_epub(str(rebuilt))
        self.assertIn(
            "Segunda cena.", book.get_item_with_href("chapter_001.xhtml").get_content().decode(),
        )

    def test_block_mapping_replaces_heading_without_losing_anchor(self):
        original = b'<html><body><h2><a id="chap01"/>Chapter I<br/>The Cyclone</h2><p>Text.</p></body></html>'
        result = replace_paragraphs_in_xhtml(original, [["Texto."]], "pt-BR", ["Capítulo I O Ciclone"])
        soup = BeautifulSoup(result, "lxml-xml")
        self.assertEqual(soup.h2.get_text(strip=True), "Capítulo I O Ciclone")
        self.assertIsNotNone(soup.h2.find(id="chap01"))
        self.assertEqual(soup.p.get_text(strip=True), "Texto.")


if __name__ == "__main__":
    unittest.main()
