"""Translate an EPUB and publish the reconstructed book and chapter files."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from lexicon.contracts import BatchRunSummary
from lexicon.epub.builder import build_epub
from lexicon.epub.chapter_mapping import read_chapter_mapping, write_chapter_mapping
from lexicon.epub.extractor import extract_epub
from lexicon.pipeline import TranslationCoordinator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EpubTranslationOptions:
    source: Path
    output: Path | None = None
    work_dir: Path | None = None
    sections: tuple[int, int] | None = None
    title: str | None = None
    language: str = "pt-BR"
    paragraph_blocks: bool = False

    @property
    def workspace(self) -> Path:
        return self.work_dir or Path.cwd() / ".lexicon_cache" / self.source.stem


@dataclass(frozen=True)
class EpubTranslationResult:
    output: Path
    summary: BatchRunSummary


async def translate_epub(
    options: EpubTranslationOptions, coordinator: TranslationCoordinator,
) -> EpubTranslationResult:
    if not options.source.is_file():
        raise FileNotFoundError(f"Arquivo EPUB não encontrado: {options.source}")

    en_dir = options.workspace / "en"
    pt_dir = options.workspace / "pt"
    output = options.output or Path.cwd() / f"{options.source.stem}_pt.epub"

    logger.info("Extraindo seções de %s", options.source.name)
    chapters = extract_epub(options.source, en_dir, paragraph_blocks=options.paragraph_blocks)
    if options.sections:
        first, last = options.sections
        if not 1 <= first <= last <= len(chapters):
            raise ValueError("Intervalo de seções fora do EPUB")
        chapters = chapters[first - 1:last]
    if not chapters:
        raise ValueError("Nenhum texto extraído do EPUB")

    logger.info("Traduzindo %d seções com %s", len(chapters), coordinator.runtime)
    summary = await coordinator.translate_batch(chapters)
    mapping = read_chapter_mapping(en_dir)
    selected = {chapter.filename: mapping[chapter.filename] for chapter in chapters}

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".lexicon-", dir=output.parent) as staging:
        stage = Path(staging)
        staged_mapping = write_chapter_mapping(stage, selected)
        for result in summary.results:
            (stage / result.chapter.filename).write_text(result.final_translation, encoding="utf-8")

        logger.info("Reconstruindo EPUB")
        candidate = stage / "translated.epub"
        build_epub(
            markdown_dir=str(stage),
            output_path=str(candidate),
            title=options.title or options.source.stem,
            source_epub_path=str(options.source),
            language=options.language,
            only_translated=options.sections is not None,
            keep_opening_documents=False,
            source_markdown_dir=en_dir,
        )
        candidate.replace(output)
        pt_dir.mkdir(parents=True, exist_ok=True)
        for path in (*stage.glob("*.md"), staged_mapping):
            shutil.copyfile(path, pt_dir / path.name)

    return EpubTranslationResult(output, summary)
