"""Score complete Oz chapters against an independent Portuguese EPUB."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from score_epub import document_blocks, score
from bs4 import BeautifulSoup
from ebooklib import epub

from lexicon.evals import compute_chrf, shannon_entropy_bits
from lexicon.glossary import GlossaryStore, occurs_in_text

# Full EPUB's chapters I–III. The first numbered file in the Livris EPUB is the introduction.
REFERENCE_FILES = ("cap02.xhtml", "cap03.xhtml", "cap04.xhtml")
VARIANTS = {
    "Wizard of Oz": (
        "mágico de oz", "feiticeiro de oz", "mago de oz", "bruxo de oz",
        "grande mágico", "grande feiticeiro", "grande mago", "grande bruxo",
        "mágico", "feiticeiro", "mago", "bruxo",
    ),
    "Emerald City": (
        "cidade das esmeraldas", "cidade de esmeraldas", "cidade esmeralda",
        "cidade das esmeralda", "cidade da esmeralda",
    ),
    "Silver Shoes": (
        "sapatos de prata", "sapatinhos de prata", "sapatilhas de prata",
        "sapatos prateados", "sapatinhos prateados", "sapatilhas prateadas",
        "calçados de prata", "sapatos de cor prateada",
    ),
    "yellow brick road": (
        "estrada de tijolos amarelos", "estrada dos tijolos amarelos",
        "caminho de tijolos amarelos", "caminho dos tijolos amarelos",
    ),
}
# (chapter, XHTML block index including headings, feminine, masculine)
# The original establishes the referent in every selected paragraph.
GENDER = (
    (1, 5, r"bonita|bela", r"bonito|belo"),
    (1, 14, r"assustada|amedrontada|apavorada|aterrorizada", r"assustado|amedrontado|apavorado|aterrorizado"),
    (1, 21, r"sozinha|solitária", r"sozinho|solitário"),
    (2, 2, r"despertada|acordada|deitada", r"despertado|acordado|deitado"),
    (2, 23, r"assustada|amedrontada|apavorada|aterrorizada", r"assustado|amedrontado|apavorado|aterrorizado"),
    (2, 44, r"sozinha|solitária", r"sozinho|solitário"),
    (3, 13, r"surpresa|espantada|admirada", r"surpreso|espantado|admirado"),
    (3, 14, r"cansada|fatigada|exausta", r"cansado|fatigado|exausto"),
    (3, 30, r"surpresa|espantada|admirada", r"surpreso|espantado|admirado"),
)


def score_oz(source_path, target_path, mapping_path, glossary_path, reference_path):
    result = score(source_path, target_path, mapping_path, glossary_path)
    source, target, reference = (
        epub.read_epub(str(path)) for path in (source_path, target_path, reference_path)
    )
    mapping = list(json.loads(mapping_path.read_text(encoding="utf-8")).values())
    if len(mapping) != 3 or any(len(item["xhtml_files"]) != 1 for item in mapping):
        raise ValueError("Expected the three complete Oz chapters.")
    store = GlossaryStore.from_markdown(glossary_path)
    snapshot = store.snapshot()
    variants = defaultdict(Counter)
    chrf = []
    gender_correct = gender_wrong = 0
    for chapter, (section, ref_name) in enumerate(zip(mapping, REFERENCE_FILES), 1):
        (name,) = section["xhtml_files"]
        originals = document_blocks(source, name)
        translated = document_blocks(target, name)
        if not translated:
            translated = [(tag, "") for tag, _ in originals]

        # The illustrated reference has one figure caption per chapter.
        ref_item = reference.get_item_with_href(ref_name)
        ref_html = BeautifulSoup(ref_item.content, "lxml-xml")
        reference_body = [p.get_text(" ", strip=True) for p in ref_html.select("p")
                          if not p.find_parent("figure") and "legenda" not in p.get("class", "").split()]
        target_body = [text for tag, text in translated if not tag.startswith("h")]
        if len(reference_body) != len(target_body):
            raise ValueError(f"Reference chapter {chapter} has a different paragraph count.")
        chrf.append(compute_chrf("\n".join(target_body), "\n".join(reference_body)))

        for (tag, original), (_, translation) in zip(originals, translated, strict=True):
            if tag.startswith("h"):
                continue
            for term in store.retrieve_chapter_context(snapshot, original)[0]:
                forms = sorted(VARIANTS.get(term.source, ()), key=len, reverse=True)
                observed = next((form for form in forms if occurs_in_text(translation, {form})), None)
                if observed:
                    variants[term.source][observed] += 1

        for _, index, feminine, masculine in (item for item in GENDER if item[0] == chapter):
            text = translated[index - 1][1]
            f = bool(re.search(rf"\b(?:{feminine})\b", text, re.IGNORECASE))
            m = bool(re.search(rf"\b(?:{masculine})\b", text, re.IGNORECASE))
            gender_correct += f and not m
            gender_wrong += m and not f

    matched = sum(sum(counter.values()) for counter in variants.values())
    result.update(
        drift_bits=round(sum(map(shannon_entropy_bits, variants.values())) / len(variants), 4) if variants else None,
        drift_matched=matched,
        variants=dict(variants),
        gender_pct=round(100 * gender_correct / (gender_correct + gender_wrong), 2) if gender_correct + gender_wrong else None,
        gender_correct=gender_correct,
        gender_marked=gender_correct + gender_wrong,
        gender_total=len(GENDER),
        chrf=round(sum(chrf) / len(chrf), 2),
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("source", "target", "mapping", "glossary", "reference"):
        parser.add_argument(argument, type=Path)
    print(json.dumps(score_oz(**{key + "_path": value for key, value in vars(parser.parse_args()).items()}),
                     ensure_ascii=False, indent=2))
