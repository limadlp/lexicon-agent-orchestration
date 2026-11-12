"""Apply local lexical consensus to the saved translation from Static Context."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lexicon.cli import add_epub_arguments, add_translation_arguments, section_range
from lexicon.config import load_dotenv
from lexicon.glossary import GlossaryStore
from lexicon.local_consensus import LMStudioConsensusTranslator
from lexicon.pipeline import TranslationCoordinator
from lexicon.translate_epub import EpubTranslationOptions, translate_epub


class ReplayTranslator(LMStudioConsensusTranslator):
    def __init__(self, translated_dir: Path, **kwargs):
        super().__init__(**kwargs)
        self.translated_dir = translated_dir

    async def translate_chapter(self, chapter, context_directive):
        prior = (self.translated_dir / chapter.filename).read_text(encoding="utf-8")
        return await self.collect_proposals(chapter, prior)


def main() -> int:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("static_translation_dir", type=Path)
    add_epub_arguments(parser)
    add_translation_arguments(parser)
    args = parser.parse_args()
    if args.engine != "asyncio" or args.mode == "glossary" or not args.glossary or not args.glossary.is_file():
        parser.error("Use asyncio e passe o mesmo glossário da tradução inicial.")
    store = GlossaryStore.from_markdown(args.glossary)
    provider = ReplayTranslator(
        args.static_translation_dir, store=store, model=args.model,
        base_url=args.base_url or "http://localhost:1234/v1",
        api_key=args.api_key or "lm-studio", temperature=args.temperature,
    )
    options = EpubTranslationOptions(
        source=args.epub, output=args.output, work_dir=args.work_dir,
        sections=section_range(args.sections),
        title=args.title, language=args.language, paragraph_blocks=True,
    )
    coordinator = TranslationCoordinator(store, provider, max_concurrency=args.concurrency)
    result = asyncio.run(translate_epub(options, coordinator))
    print(f"EPUB gerado em: {result.output}")
    print(f"Conflitos detectados: {len(result.summary.resolutions)}")
    print(f"Tokens: {result.summary.total_prompt_tokens + result.summary.total_completion_tokens}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
