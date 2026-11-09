"""EPUB chapter extractor for Lexicon.

Extracts EPUB documents to markdown chapters and generates ChapterDocument instances.
"""

from __future__ import annotations

import os
import posixpath
import re
from pathlib import Path

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub
from markdownify import markdownify

from lexicon.epub.chapter_mapping import ChapterMapping, write_chapter_mapping
from lexicon.models import ChapterDocument

HTML_PARSER = "html.parser"

SECTION_START_RE = re.compile(
    r"^\s*(Chapter\s+(?:[IVXLC\d]+|One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|Eleven|Twelve|Thirteen|Fourteen|Fifteen|Sixteen|Seventeen|Eighteen|Nineteen|Twenty|Thirty|Forty|Fifty|Sixty)\b|\d+\s*:|Prologue|Epilogue|Appendix|Acknowledgements?|Dedication|Introduction|Preface)",
    re.IGNORECASE,
)

SKIP_RE = re.compile(r"royallib\.com", re.IGNORECASE)


def text_preview(html_str: str) -> str:
    """Return first 120 chars of plain text from HTML."""
    soup = BeautifulSoup(html_str, HTML_PARSER)
    content = soup.body if soup.body is not None else soup
    return content.get_text(separator=" ").strip()[:120]


def _toc_chapters(book: epub.EpubBook) -> list[dict]:
    """Return flat chapter entries that point to an XHTML anchor in the TOC."""
    chapters = []
    for entry in book.toc:
        href = getattr(entry, "href", "")
        title = getattr(entry, "title", "")
        if not href or "#" not in href or not SECTION_START_RE.match(title):
            continue
        xhtml_name, anchor = href.split("#", 1)
        xhtml_name = posixpath.normpath(xhtml_name)
        document_names = [item.get_name() for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT)]
        xhtml_name = next(
            (name for name in document_names if xhtml_name == name or xhtml_name.endswith("/" + name)),
            xhtml_name,
        )
        chapters.append({"title": title, "xhtml_name": xhtml_name, "anchor": anchor})
    return chapters


def _toc_titles_by_href(book: epub.EpubBook) -> dict[str, str]:
    """Return TOC titles keyed by their XHTML document path."""
    titles = {}

    def visit(entries) -> None:
        for entry in entries:
            if isinstance(entry, (list, tuple)):
                visit(entry)
                continue
            href = getattr(entry, "href", "")
            title = getattr(entry, "title", "")
            if href and title:
                titles.setdefault(href.split("#", 1)[0], title)

    visit(book.toc)
    return titles


def _extract_toc_anchored_epub(book: epub.EpubBook) -> list[dict] | None:
    """Extract sections from TOC anchors when several chapters share XHTML files."""
    chapters = _toc_chapters(book)
    if not chapters:
        return None

    chapter_documents = [chapter["xhtml_name"] for chapter in chapters]
    if len(chapter_documents) == len(set(chapter_documents)):
        return None

    documents = []
    positions = {}
    flat_paragraphs = []

    for idref, _linear in book.spine:
        item = book.get_item_with_id(idref)
        if item is None:
            continue
        html_str = item.get_content().decode("utf-8", errors="replace")
        soup = BeautifulSoup(html_str, HTML_PARSER)
        paragraphs = [p for p in soup.find_all("p") if p.get_text(strip=True)]
        if not paragraphs:
            continue

        xhtml_name = item.get_name()
        paragraph_positions = {id(paragraph): index for index, paragraph in enumerate(paragraphs)}
        for chapter in chapters:
            if chapter["xhtml_name"] != xhtml_name:
                continue
            anchor = soup.find(id=chapter["anchor"]) or soup.find(attrs={"name": chapter["anchor"]})
            parent = (anchor if anchor.name == "p" else anchor.find_parent("p")) if anchor else None
            if parent is None or id(parent) not in paragraph_positions:
                return None
            positions[(xhtml_name, chapter["anchor"])] = (
                len(flat_paragraphs) + paragraph_positions[id(parent)]
            )

        documents.append((xhtml_name, paragraphs))
        flat_paragraphs.extend((xhtml_name, paragraph, index) for index, paragraph in enumerate(paragraphs))

    if len(positions) != len(chapters):
        return None

    starts = [positions[(chapter["xhtml_name"], chapter["anchor"])] for chapter in chapters]
    if starts != sorted(starts):
        return None

    def make_section(title: str, start: int, end: int) -> dict:
        selected = flat_paragraphs[start:end]
        if not selected:
            return {"title": title, "parts": [], "xhtml_names": [], "para_counts": []}

        parts = []
        xhtml_names = []
        para_counts = []
        paragraph_offsets = []
        current_name = None
        current_parts = []
        current_indices = []

        for xhtml_name, paragraph, paragraph_index in selected:
            if xhtml_name != current_name:
                if current_name is not None:
                    parts.append("\n".join(current_parts))
                    xhtml_names.append(current_name)
                    para_counts.append(len(current_parts))
                    paragraph_offsets.append(current_indices)
                current_name = xhtml_name
                current_parts = []
                current_indices = []
            current_parts.append(str(paragraph))
            current_indices.append(paragraph_index)

        parts.append("\n".join(current_parts))
        xhtml_names.append(current_name)
        para_counts.append(len(current_parts))
        paragraph_offsets.append(current_indices)
        return {
            "title": title,
            "parts": parts,
            "xhtml_names": xhtml_names,
            "para_counts": para_counts,
            "paragraph_offsets": paragraph_offsets,
        }

    first_start = starts[0]
    front_title = text_preview("\n".join(str(paragraph) for _, paragraph, _ in flat_paragraphs[:first_start]))
    sections = [make_section(front_title, 0, first_start)] if first_start else []
    for index, chapter in enumerate(chapters):
        end = starts[index + 1] if index + 1 < len(starts) else len(flat_paragraphs)
        sections.append(make_section(chapter["title"], starts[index], end))
    return sections


def localizable_elements(soup):
    """Text blocks supported by the EPUB builder, in document order."""
    return [
        node for node in soup.find_all(["p", "li", "h1", "h2", "h3", "h4", "h5", "h6"])
        if node.get_text(strip=True)
        and not (node.name == "li" and node.find("p"))
        and not node.find_parent(class_=re.compile(r"(?:^|\s)pg-boilerplate(?:\s|$)"))
    ]


def extract_epub(
    epub_path: str | Path, output_dir: str | Path, *, paragraph_blocks: bool = False,
) -> list[ChapterDocument]:
    """Extract epub chapters into markdown files and return list of ChapterDocuments."""
    epub_path = Path(epub_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    book = epub.read_epub(str(epub_path))
    toc_titles = _toc_titles_by_href(book)

    sections = _extract_toc_anchored_epub(book)
    if sections is None:
        sections = []

    for idref, _linear in ([] if sections else book.spine):
        item = book.get_item_with_id(idref)
        if item is None:
            continue

        html_str = item.get_content().decode("utf-8", errors="replace")
        preview = text_preview(html_str)

        if len(preview) < 5 or SKIP_RE.search(preview):
            continue

        soup = BeautifulSoup(html_str, HTML_PARSER)
        para_count = len([p for p in soup.find_all("p") if p.get_text(strip=True)])
        xhtml_name = item.get_name()
        toc_title = toc_titles.get(xhtml_name, "")
        document_title = soup.title.get_text(separator=" ").strip() if soup.title else ""
        chapter_match = re.search(r'id=["\']chapter-(\d+)["\']', html_str, re.IGNORECASE)
        chapter_file_match = re.search(r'(?:^|/)chapter[-_]?(\d+)\.x?html?$', xhtml_name, re.IGNORECASE)
        semantic_section_match = re.search(r'\belement-type-(chapter|prologue|epilogue)\b', html_str, re.IGNORECASE)
        section_start = document_title if SECTION_START_RE.match(document_title) else preview

        is_calibre_chapter = chapter_match is not None and xhtml_name.endswith("_split_000.html")
        is_chapter = bool(
            is_calibre_chapter
            or chapter_file_match
            or semantic_section_match
            or toc_title
        )

        if is_chapter or SECTION_START_RE.match(section_start):
            heading = soup.find(re.compile(r"^h[1-6]$"))
            if toc_title:
                title = toc_title
            elif is_chapter and (chapter_match or chapter_file_match):
                number_match = chapter_match or chapter_file_match
                heading_text = heading.get_text(separator=" ").strip() if heading else ""
                title = f"Chapter {number_match.group(1)}"
                if heading_text:
                    title += f": {heading_text}"
            elif is_chapter and heading:
                title = heading.get_text(separator=" ").strip()
            elif document_title:
                title = document_title
            elif heading:
                title = heading.get_text(separator=" ").strip()
            else:
                title = preview.split(".")[0].strip()[:80]

            sections.append({
                "title": title,
                "parts": [html_str],
                "xhtml_names": [xhtml_name],
                "para_counts": [para_count],
            })
        elif sections:
            sections[-1]["parts"].append(html_str)
            sections[-1]["xhtml_names"].append(xhtml_name)
            sections[-1]["para_counts"].append(para_count)
        else:
            heading = soup.find(re.compile(r"^h[1-6]$"))
            title = heading.get_text().strip() if heading else preview[:60]
            sections.append({
                "title": title,
                "parts": [html_str],
                "xhtml_names": [xhtml_name],
                "para_counts": [para_count],
            })

    chapters: list[ChapterDocument] = []
    chapters_mapping: ChapterMapping = {}

    for chapter_num, section in enumerate(sections, start=1):
        fragments = [BeautifulSoup(part, HTML_PARSER) for part in section["parts"]]
        combined_html = "\n".join(
            fragment.body.decode_contents() if fragment.body else str(fragment)
            for fragment in fragments
        )
        heading_counts = None
        if paragraph_blocks and "paragraph_offsets" not in section:
            blocks = [localizable_elements(fragment.body or fragment) for fragment in fragments]
            heading_counts = [sum(node.name.startswith("h") for node in group) for group in blocks]
            section["para_counts"] = [len(group) - count for group, count in zip(blocks, heading_counts)]
            markdown_blocks = []
            for group in blocks:
                for node in group:
                    # A list item is one prose block; the original XHTML keeps its marker.
                    html = node.decode_contents() if node.name == "li" else str(node)
                    block = markdownify(html, heading_style="ATX", strip=["img"]).strip()
                    markdown_blocks.append(re.sub(r"\n[ \t]*\n+", "\n", block))
            if not markdown_blocks:
                continue
            markdown_text = "\n\n".join(markdown_blocks)
        else:
            markdown_text = markdownify(combined_html, heading_style="ATX", strip=["img"])

        slug = re.sub(r"[^\w\s-]", "", section["title"])
        slug = re.sub(r"[\s]+", "_", slug).strip("_")[:60]

        filename = f"{chapter_num:03d}_{slug}.md"
        filepath = output_dir / filename
        filepath.write_text(markdown_text, encoding="utf-8")

        chapters_mapping[filename] = {
            "xhtml_files": section["xhtml_names"],
            "para_counts": section["para_counts"],
        }
        if "paragraph_offsets" in section:
            chapters_mapping[filename]["paragraph_offsets"] = section["paragraph_offsets"]
        if heading_counts is not None:
            chapters_mapping[filename]["heading_counts"] = heading_counts

        chapters.append(
            ChapterDocument(
                number=chapter_num,
                filename=filename,
                title=section["title"],
                content=markdown_text,
                source_path=filepath,
            )
        )

    write_chapter_mapping(output_dir, chapters_mapping)

    return chapters
