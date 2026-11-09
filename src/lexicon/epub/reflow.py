"""Reconstruct chapters split across pages in PDF-derived EPUBs."""

import html
import re
from pathlib import Path

from bs4 import BeautifulSoup
from ebooklib import epub

from lexicon.epub.alignment import _paragraph_length, _strip_md_formatting, get_md_paragraphs
from lexicon.epub.chapter_mapping import ChapterMapping, ChapterMappingEntry
from lexicon.epub.xhtml import _markdown_inline_xhtml


def _is_false_paragraph_break(previous: str, following: str) -> bool:
    """Identify paragraph boundaries introduced by PDF page layout."""
    previous = _strip_md_formatting(previous).rstrip()
    following = _strip_md_formatting(following).lstrip()
    if not previous or not following:
        return False

    # A page can split dialogue even after a sentence-ending period. The
    # unmatched opening quote in the accumulated paragraph identifies it.
    if previous.count("“") > previous.count("”"):
        return True
    if previous.count('"') % 2:
        return True

    # Commas, dashes, and bare words cannot end a logical prose paragraph.
    if not re.search(r"[.!?…][”’\"')\]*_]*$", previous):
        return True

    # A lowercase continuation after a sentence boundary is also a page split.
    first_letter = next((character for character in following if character.isalpha()), "")
    return bool(first_letter and first_letter.islower())


def _source_scene_breaks(source_book: epub.EpubBook, chapter_info: ChapterMappingEntry) -> set[int]:
    """Return source paragraph boundaries separated by blank layout paragraphs."""
    items = {item.get_name(): item for item in source_book.get_items()}
    boundaries: set[int] = set()
    chapter_offset = 0
    paragraph_offsets = chapter_info.get("paragraph_offsets") or []
    for name, count, offsets in zip(
        chapter_info["xhtml_files"],
        chapter_info["para_counts"],
        paragraph_offsets,
    ):
        item = items.get(name)
        if item is None:
            chapter_offset += count
            continue
        soup = BeautifulSoup(item.content, "lxml-xml")
        all_paragraphs = soup.find_all("p")
        nonempty_positions = [
            index
            for index, paragraph in enumerate(all_paragraphs)
            if paragraph.get_text(strip=True)
        ]
        for local_index in range(min(count, len(offsets)) - 1):
            left_offset = offsets[local_index]
            right_offset = offsets[local_index + 1]
            if left_offset >= len(nonempty_positions) or right_offset >= len(nonempty_positions):
                continue
            left = nonempty_positions[left_offset]
            right = nonempty_positions[right_offset]
            if any(
                not paragraph.get_text(strip=True)
                for paragraph in all_paragraphs[left + 1:right]
            ):
                boundaries.add(chapter_offset + local_index)
        chapter_offset += count
    return boundaries


def _reflow_chapter_paragraphs(
    source: list[str],
    translated: list[str],
    scene_breaks: set[int],
) -> list[str | None]:
    """Conservatively merge Portuguese continuations and retain scene breaks."""
    if not translated:
        return []

    merged = [translated[0]]
    for paragraph in translated[1:]:
        previous = _strip_md_formatting(merged[-1]).rstrip()
        following = _strip_md_formatting(paragraph).lstrip()
        first_letter = next(
            (character for character in following if character.isalpha()), ""
        )
        starts_dialogue = following.startswith(("—", "–", "-", "“", '"'))
        previous_starts_dialogue = previous.startswith(("—", "–", "-"))
        is_standalone_dash = following in {"—", "–", "-"}
        has_terminal_punctuation = bool(
            re.search(r"[.!?…][”’\"')\]]*_*?$", previous)
        )
        is_unambiguous_continuation = (
            bool(first_letter and first_letter.islower())
            or is_standalone_dash
            or (
                not has_terminal_punctuation
                and not previous.endswith(":")
                and not (previous_starts_dialogue and starts_dialogue)
            )
        )
        if len(merged) > 1 and is_unambiguous_continuation:
            merged[-1] = f"{merged[-1]} {paragraph}"
        else:
            merged.append(paragraph)

    # Blank paragraphs in the source mark scene changes, but source and
    # translation paragraph indexes can drift. Map them by cumulative text
    # position onto the nearest boundary in the conservatively merged text.
    real_scene_breaks = [
        index
        for index in sorted(scene_breaks)
        if 0 < index < len(source) - 1
        and not _is_false_paragraph_break(source[index], source[index + 1])
    ]
    if not real_scene_breaks or len(merged) < 3:
        return merged

    source_lengths = [_paragraph_length(paragraph) + 1 for paragraph in source]
    translated_lengths = [_paragraph_length(paragraph) + 1 for paragraph in merged]
    source_total = sum(source_lengths)
    translated_total = sum(translated_lengths)
    source_cumulative = []
    running = 0
    for length in source_lengths:
        running += length
        source_cumulative.append(running / source_total)
    translated_cumulative = []
    running = 0
    for length in translated_lengths:
        running += length
        translated_cumulative.append(running / translated_total)

    mapped_breaks = []
    previous_boundary = 0
    for scene_index in real_scene_breaks:
        target = source_cumulative[scene_index]
        candidates = range(previous_boundary + 1, len(merged) - 1)
        boundary = min(
            candidates,
            key=lambda index: abs(translated_cumulative[index] - target),
            default=None,
        )
        if boundary is None:
            break
        mapped_breaks.append(boundary)
        previous_boundary = boundary

    blocks: list[str | None] = []
    for index, paragraph in enumerate(merged):
        blocks.append(paragraph)
        if index in mapped_breaks:
            blocks.append(None)
    return blocks


def _render_reflowed_document(
    document_title: str,
    blocks: list[str | None],
    language: str,
) -> bytes:
    rendered = []
    for block in blocks:
        if block is None:
            rendered.append('<div class="scene-break" aria-hidden="true">&#160;</div>')
        else:
            rendered.append(
                f'<p class="book-paragraph">{_markdown_inline_xhtml(block)}</p>'
            )
    body = "\n".join(rendered)
    return f'''<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="{language}" lang="{language}">
<head>
  <meta charset="utf-8"/>
  <title>{html.escape(document_title)}</title>
  <link rel="stylesheet" type="text/css" href="chapter_styles.css"/>
</head>
<body class="calibre">
  <section epub:type="chapter">
    <h1 class="chapter-title">{html.escape(document_title)}</h1>
{body}
  </section>
</body>
</html>
'''.encode("utf-8")


def build_reflowed_chapters(
    md_dir: Path,
    mapping: ChapterMapping,
    source_book: epub.EpubBook,
    out_book: epub.EpubBook,
    language: str,
    source_dir: Path,
) -> tuple[list, list, int]:
    """Create one XHTML spine item per Markdown chapter."""
    chapters = []
    toc = []
    merged_breaks = 0
    chapter_files = sorted(
        path for path in md_dir.glob("[0-9][0-9][0-9]_Chapter_*.md") if path.name in mapping
    )
    for index, md_file in enumerate(chapter_files, start=1):
        source_file = source_dir / md_file.name
        if not source_file.exists():
            raise ValueError(f"Source Markdown not found for {md_file.name}")
        translated = get_md_paragraphs(md_file.read_text(encoding="utf-8"))
        source = get_md_paragraphs(source_file.read_text(encoding="utf-8"))
        chapter_info = mapping[md_file.name]
        scene_breaks = _source_scene_breaks(source_book, chapter_info)
        # Blank source paragraphs after the title supplied heading spacing in
        # the PDF conversion; the chapter-title CSS now handles that spacing.
        scene_breaks.discard(0)
        blocks = _reflow_chapter_paragraphs(source, translated, scene_breaks)
        if not blocks or blocks[0] is None:
            raise ValueError(f"Chapter title not found in {md_file.name}")
        chapter_title = _strip_md_formatting(blocks[0])
        chapter_body = blocks[1:]
        merged_breaks += len(source) - sum(block is not None for block in blocks)
        file_name = f"chapter_{index:03d}.xhtml"
        chapter = epub.EpubItem(
            uid=f"chapter-{index:03d}",
            file_name=file_name,
            media_type="application/xhtml+xml",
            content=_render_reflowed_document(chapter_title, chapter_body, language),
        )
        out_book.add_item(chapter)
        chapters.append(chapter)
        toc.append(epub.Link(file_name, chapter_title, f"toc-chapter-{index:03d}"))
    return chapters, toc, merged_breaks


def build_reflow_frontmatter(
    md_dir: Path, mapping: ChapterMapping, out_book: epub.EpubBook, language: str,
):
    """Keep translated title and copyright text, leaving navigation to the EPUB TOC."""
    frontmatter_files = sorted(md_dir.glob("[0-9][0-9][0-9]_*.md"))
    frontmatter_files = [
        path for path in frontmatter_files if path.name in mapping and "_Chapter_" not in path.name
    ]
    if not frontmatter_files:
        return []
    paragraphs = get_md_paragraphs(frontmatter_files[0].read_text(encoding="utf-8"))
    paragraphs = [p for p in paragraphs if not re.match(r"^\[[^]]+\]\([^)]+\)$", p)]
    if not paragraphs:
        return []
    body = "\n".join(f'<p class="book-paragraph">{_markdown_inline_xhtml(p)}</p>' for p in paragraphs[1:])
    content = f'''<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="{language}" lang="{language}">
<head><meta charset="utf-8"/><title>Folha de rosto</title><link rel="stylesheet" type="text/css" href="chapter_styles.css"/></head>
<body><section><h1 class="chapter-title">{html.escape(paragraphs[0])}</h1>{body}</section></body>
</html>'''.encode("utf-8")
    item = epub.EpubItem(uid="translated-frontmatter", file_name="frontmatter.xhtml", media_type="application/xhtml+xml", content=content)
    out_book.add_item(item)
    return [item]
