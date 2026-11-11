"""Command-line interface for the Lexicon translation and evaluation pipeline."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from lexicon.config import load_dotenv
from lexicon.contracts import LLMProvider
from lexicon.evals import TranslationEvaluator
from lexicon.glossary import GlossaryStore
from lexicon.local_consensus import LMStudioConsensusTranslator
from lexicon.models import ChapterDocument
from lexicon.pipeline import TranslationCoordinator
from lexicon.providers import OpenAICompatibleProvider
from lexicon.providers.local import LMStudioTranslator


def _translation_provider(args: argparse.Namespace, store: GlossaryStore) -> LLMProvider:
    options = {
        "model": args.model,
        "base_url": args.base_url,
        "api_key": args.api_key,
        "temperature": args.temperature,
    }
    if args.lm_studio:
        options["base_url"] = args.base_url or "http://localhost:1234/v1"
        options["api_key"] = args.api_key or "lm-studio"
        if args.mode == "reconcile":
            return LMStudioConsensusTranslator(store=store, **options)
        return LMStudioTranslator(**options)
    return OpenAICompatibleProvider(**options)


def load_markdown_chapters(directory: Path) -> list[ChapterDocument]:
    """Load numbered Markdown chapter files from a directory."""
    chapters: list[ChapterDocument] = []
    for idx, path in enumerate(sorted(directory.glob("*.md")), start=1):
        if path.name.startswith(("glossary", "_")):
            continue
        match = re.match(r"^(\d+)", path.stem)
        ch_num = int(match.group(1)) if match else idx
        content = path.read_text(encoding="utf-8")
        first_line = content.partition("\n")[0].lstrip("# ").strip() if content else path.stem
        chapters.append(
            ChapterDocument(
                number=ch_num,
                filename=path.name,
                title=first_line,
                content=content,
                source_path=path,
            )
        )
    return chapters


def _coordinator(args: argparse.Namespace, store: GlossaryStore) -> TranslationCoordinator:
    reconcile = args.mode == "reconcile" or (args.mode is None and not args.lm_studio)
    return TranslationCoordinator(
        store=store,
        provider=_translation_provider(args, store),
        engine=args.engine,
        max_concurrency=args.concurrency,
        enable_reconciliation=reconcile,
    )


def cmd_translate(args: argparse.Namespace) -> int:
    store = (
        GlossaryStore.from_markdown(args.glossary)
        if args.glossary.exists()
        else GlossaryStore()
    )
    chapters = load_markdown_chapters(args.source_dir)
    if not chapters:
        print(f"Nenhum capítulo encontrado em {args.source_dir}", file=sys.stderr)
        return 1

    coordinator = _coordinator(args, store)
    print(f"Motor: {coordinator.runtime}", flush=True)
    summary = asyncio.run(coordinator.translate_batch(chapters))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for res in summary.results:
        out_path = args.output_dir / res.chapter.filename
        out_path.write_text(res.final_translation, encoding="utf-8")

    if not args.lm_studio:
        args.glossary.parent.mkdir(parents=True, exist_ok=True)
        args.glossary.write_text(store.to_markdown(), encoding="utf-8")

    print(f"Capítulos traduzidos: {len(summary.results)}")
    print(f"Conflitos lexicais reconciliados: {len(summary.resolutions)}")
    print(f"Capítulos revisados na Fase 4: {summary.reconciled_chapters_count}")
    print(
        f"Versão do glossário: v{summary.initial_glossary_version} -> v{summary.final_glossary_version}"
    )
    print(
        f"Tokens consumidos: prompt={summary.total_prompt_tokens}, completion={summary.total_completion_tokens}"
    )
    return 0


def cmd_translate_epub(args: argparse.Namespace) -> int:
    from lexicon.translate_epub import EpubTranslationOptions, translate_epub

    options = EpubTranslationOptions(
        source=args.epub,
        output=args.output,
        work_dir=args.work_dir,
        sections=section_range(args.sections),
        title=args.title,
        language=args.language,
        paragraph_blocks=args.lm_studio,
    )
    glossary_path = args.glossary or options.workspace / "glossary.md"
    store = (
        GlossaryStore.from_markdown(glossary_path) if glossary_path.exists() else GlossaryStore()
    )
    try:
        result = asyncio.run(translate_epub(options, _coordinator(args, store)))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1

    if not args.lm_studio:
        glossary_path.parent.mkdir(parents=True, exist_ok=True)
        glossary_path.write_text(store.to_markdown(), encoding="utf-8")

    summary = result.summary
    print(f"EPUB gerado em: {result.output}")
    print(f"Capítulos traduzidos: {len(summary.results)}")
    print(f"Conflitos detectados: {len(summary.resolutions)}")
    print(f"Capítulos revisados: {summary.reconciled_chapters_count}")
    print(f"Tokens: {summary.total_prompt_tokens + summary.total_completion_tokens}")
    return 0


def cmd_extract_epub(args: argparse.Namespace) -> int:
    from lexicon.epub.extractor import extract_epub
    epub_path = Path(args.epub)
    output_dir = Path(args.output_dir)
    chapters = extract_epub(epub_path, output_dir)
    print(f"Extraídos {len(chapters)} capítulos para {output_dir}")
    return 0


def cmd_build_epub(args: argparse.Namespace) -> int:
    from lexicon.epub.builder import build_epub
    build_epub(
        markdown_dir=str(args.source_dir),
        output_path=str(args.output),
        title=args.title or args.source_dir.stem,
        source_epub_path=str(args.source_epub) if args.source_epub else None,
        language=args.language,
    )
    print(f"EPUB gerado em: {args.output}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    store = GlossaryStore.from_markdown(args.glossary)
    chapters = load_markdown_chapters(args.source_dir)
    translations: dict[int, str] = {}
    for ch in chapters:
        trans_path = args.translation_dir / ch.filename
        if trans_path.exists():
            translations[ch.number] = trans_path.read_text(encoding="utf-8")

    references: dict[int, str] | None = None
    if args.reference_dir and args.reference_dir.exists():
        references = {}
        for ch in chapters:
            ref_path = args.reference_dir / ch.filename
            if ref_path.exists():
                references[ch.number] = ref_path.read_text(encoding="utf-8")

    evaluator = TranslationEvaluator(store.snapshot())
    metrics = evaluator.evaluate(chapters, translations, references)

    report = {
        "total_chapters": metrics.total_chapters,
        "glossary_adherence_rate_pct": metrics.glossary_adherence_rate,
        "glossary_violation_rate_pct": metrics.glossary_violation_rate,
        "lexical_drift_bits": metrics.lexical_drift_bits,
        "gender_agreement_accuracy_pct": metrics.gender_agreement_accuracy,
        "structural_parity_rate_pct": metrics.structural_parity_rate,
        "chrf_score": metrics.chrf_score,
        "per_term_entropy_bits": metrics.per_term_entropy,
        "violations_count": {
            "terminology": len(metrics.term_violations),
            "gender": len(metrics.gender_violations),
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def add_translation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--engine",
        choices=["asyncio", "langgraph", "adk"],
        default="asyncio",
        help="Motor de orquestração: asyncio puro, langgraph (StateGraph) ou adk (Google ADK)",
    )
    parser.add_argument("--model", default=None, help="Identificador do modelo (padrão: gpt-4o-mini ou definido no .env)")
    parser.add_argument("--lm-studio", action="store_true", help="Traduz em blocos curtos pelo servidor local do LM Studio")
    parser.add_argument("--base-url", default=None, help="Endpoint compatível com OpenAI API (padrão: OpenAI ou definido no .env)")
    parser.add_argument("--api-key", default=None, help="Chave de API (opcional se configurada via .env)")
    parser.add_argument("--concurrency", type=int, default=4, help="Número máximo de workers paralelos")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperatura de geração")
    parser.add_argument(
        "--mode", choices=["glossary", "reconcile"],
        help="Glossário fixo ou revisão seletiva (padrão: glossary no LM Studio, reconcile nos demais)",
    )


def section_range(sections: Sequence[int] | None) -> tuple[int, int] | None:
    if sections is None:
        return None
    first, last = sections
    return first, last


def add_epub_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("epub", type=Path, help="Arquivo EPUB de origem")
    parser.add_argument("-o", "--output", type=Path, help="Arquivo EPUB de saída")
    parser.add_argument("--work-dir", type=Path, help="Diretório de trabalho intermediário")
    parser.add_argument("--glossary", type=Path, help="Caminho para o glossário")
    parser.add_argument("--sections", nargs=2, type=int, metavar=("FIRST", "LAST"),
                        help="Intervalo inclusivo de seções")
    parser.add_argument("-t", "--title", help="Título do livro traduzido")
    parser.add_argument("-l", "--language", default="pt-BR", help="Idioma do EPUB")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lexicon",
        description="Stateful literary translation pipeline with lexical consensus and evaluation.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    tr = sub.add_parser("translate", help="Executa o pipeline de 4 fases sobre capítulos Markdown")
    tr.add_argument("--source-dir", type=Path, required=True, help="Diretório com capítulos de origem")
    tr.add_argument("--output-dir", type=Path, required=True, help="Diretório de saída das traduções")
    tr.add_argument("--glossary", type=Path, required=True, help="Caminho para o arquivo glossary.md")
    add_translation_arguments(tr)
    tr.set_defaults(func=cmd_translate)

    # Subcomando: translate-epub (end-to-end all-in-one de EPUB para EPUB)
    te = sub.add_parser("translate-epub", help="Traduz diretamente um arquivo .epub completo gerando um novo .epub em português")
    add_translation_arguments(te)
    add_epub_arguments(te)
    te.set_defaults(func=cmd_translate_epub)

    # Subcomando: extract (extrai EPUB para pasta markdown)
    ex = sub.add_parser("extract", help="Extrai capítulos de um EPUB em arquivos Markdown")
    ex.add_argument("epub", type=Path, help="Caminho do arquivo EPUB de origem")
    ex.add_argument("-o", "--output-dir", type=Path, required=True, help="Diretório onde os capítulos markdown serão salvos")
    ex.set_defaults(func=cmd_extract_epub)

    # Subcomando: build-epub (reconstrói EPUB a partir de pasta markdown)
    bp = sub.add_parser("build-epub", help="Reconstrói um arquivo EPUB a partir de capítulos Markdown traduzidos")
    bp.add_argument("source_dir", type=Path, help="Diretório com os capítulos Markdown em português")
    bp.add_argument("-o", "--output", type=Path, required=True, help="Caminho do arquivo EPUB gerado")
    bp.add_argument("-s", "--source-epub", type=Path, default=None, help="EPUB original de referência (para estilos e estrutura)")
    bp.add_argument("-t", "--title", default=None, help="Título do livro")
    bp.add_argument("-l", "--language", default="pt-BR", help="Código de idioma (padrão: pt-BR)")
    bp.set_defaults(func=cmd_build_epub)

    ev = sub.add_parser("evaluate", help="Executa a suíte de avaliação determinística e chrF")
    ev.add_argument("--source-dir", type=Path, required=True)
    ev.add_argument("--translation-dir", type=Path, required=True)
    ev.add_argument("--glossary", type=Path, required=True)
    ev.add_argument("--reference-dir", type=Path, default=None)
    ev.set_defaults(func=cmd_evaluate)

    return parser


def main() -> int:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "concurrency", 1) < 1:
        parser.error("--concurrency deve ser maior que zero")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
