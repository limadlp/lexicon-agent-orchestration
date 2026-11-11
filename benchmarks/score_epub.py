"""Measure paragraph coverage and glossary adherence on matching EPUB sections."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bs4 import BeautifulSoup
from ebooklib import epub

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lexicon.epub.extractor import localizable_elements
from lexicon.glossary import GlossaryStore, occurs_in_text, portuguese_inflections
from lexicon.local_translation import _check_translation, _words


def document_blocks(book, name):
    item = book.get_item_with_href(name)
    if item is None:
        return []
    return [
        (node.name, node.get_text(" ", strip=True))
        for node in localizable_elements(BeautifulSoup(item.content, "lxml-xml"))
    ]


def score(source_path, target_path, mapping_path, glossary_path):
    source, target = (epub.read_epub(str(path)) for path in (source_path, target_path))
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    documents = [name for info in mapping.values() for name in info["xhtml_files"]]
    if not documents or len(set(documents)) != len(documents) or any(
        "heading_counts" not in info for info in mapping.values()
    ):
        raise ValueError("Use o mapa de um EPUB traduzido em modo LM Studio, com documentos distintos.")
    snapshot = GlossaryStore.from_markdown(glossary_path).snapshot()
    results = {}
    for section, info in mapping.items():
        counts = {"paragraphs": 0, "covered": 0, "terms": 0, "adhered": 0}
        for name in info["xhtml_files"]:
            originals = document_blocks(source, name)
            translations = document_blocks(target, name)
            if not originals:
                raise ValueError(f"{name}: source document is missing or empty")
            if translations and [tag for tag, _ in originals] != [tag for tag, _ in translations]:
                raise ValueError(f"{name}: the XHTML blocks do not align")
            for index, (tag, original) in enumerate(originals):
                if tag.startswith("h"):
                    continue
                counts["paragraphs"] += 1
                translated = translations[index][1] if translations else ""
                if _words(original):
                    try:
                        _check_translation(original, translated)
                    except ValueError:
                        pass
                    else:
                        counts["covered"] += 1
                else:
                    counts["covered"] += bool(translated)
                terms, _ = GlossaryStore.retrieve_chapter_context(snapshot, original)
                for term in terms:
                    counts["terms"] += 1
                    if occurs_in_text(translated, portuguese_inflections(term.target)) and not any(
                        occurs_in_text(translated, portuguese_inflections(form))
                        for form in term.forbidden_synonyms
                    ):
                        counts["adhered"] += 1
        results[section] = counts
    total = {key: sum(counts[key] for counts in results.values()) for key in (
        "paragraphs", "covered", "terms", "adhered",
    )}
    return {
        "sections": results,
        "total": total,
        "coverage_pct": round(100 * total["covered"] / total["paragraphs"], 2) if total["paragraphs"] else None,
        "glossary_pct": round(100 * total["adhered"] / total["terms"], 2) if total["terms"] else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("source", "target", "mapping", "glossary"):
        parser.add_argument(argument, type=Path)
    args = parser.parse_args()
    print(json.dumps(score(args.source, args.target, args.mapping, args.glossary), indent=2))
