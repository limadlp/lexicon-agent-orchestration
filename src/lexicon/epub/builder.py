"""Build a translated epub from markdown files, preserving source formatting."""

import argparse
import logging
import posixpath
import re
import uuid
from pathlib import Path

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub

from lexicon.epub.alignment import (
    _align_translated_paragraphs,
    _expand_markdown_table_cells,
    _extract_headings,
    _strip_md_formatting,
    get_md_paragraphs,
)
from lexicon.epub.chapter_mapping import ChapterMapping, find_chapter_mapping
from lexicon.epub.extractor import localizable_elements
from lexicon.epub.reflow import build_reflow_frontmatter, build_reflowed_chapters
from lexicon.epub.xhtml import (
    replace_paragraphs_at_offsets,
    replace_paragraphs_in_xhtml,
)

logger = logging.getLogger(__name__)


CHAPTER_STYLES = r"""

/* Normalize translated novel text to match the typography used by ARC 2. */
.book-paragraph {
    display: block;
    font-size: 1em;
    line-height: 1.2;
    text-align: justify;
    text-indent: 1.5em;
    margin: 0;
}
.chapter-title {
    break-before: page;
    display: block;
    font-size: 1.5em;
    font-weight: normal;
    line-height: 1.2;
    page-break-after: avoid;
    page-break-before: always;
    text-align: center;
    text-indent: 0;
    margin: 1.85% 0 1em;
}
.scene-break {
    break-inside: avoid;
    height: 1.2em;
    margin: 0.6em 0;
}
"""


def _copy_items(
    source_book: epub.EpubBook,
    out_book: epub.EpubBook,
    md_translations: dict,
    md_headings: dict,
    chapter_documents: set[str],
    language: str,
    title: str,
    only_translated: bool = False,
    keep_opening_documents: bool = True,
) -> dict:
    """Copy all items from source to output book; return dict of name -> EpubHtml for documents."""
    leading_documents: set[str] = set()
    if only_translated:
        # Cover and title-page documents are not consistently named in source
        # EPUBs (this book calls them c0.xhtml and c9.xhtml). Preserve every
        # spine document before the first translated document so partial books
        # retain their opening pages without pulling in untranslated chapters.
        items_by_id = {item.get_id(): item for item in source_book.get_items()}
        for item_id, _linear in source_book.spine:
            item = items_by_id.get(item_id)
            if item is None:
                continue
            if item.get_name() in md_translations or item.get_name() in md_headings:
                break
            if keep_opening_documents or not localizable_elements(BeautifulSoup(item.content, "lxml-xml")):
                leading_documents.add(item.get_name())

    item_by_name = {}
    for item in source_book.get_items():
        item_type = item.get_type()
        if item_type == ebooklib.ITEM_DOCUMENT:
            if isinstance(item, epub.EpubNav) or "nav" in getattr(item, "properties", []):
                continue
            if (
                only_translated
                and item.get_name() not in md_translations
                and item.get_name() not in md_headings
                and item.get_name() not in leading_documents
                and not item.get_name().lower().endswith(("cover.xhtml", "titlepage.xhtml"))
            ):
                continue
            new_item = _copy_document(
                item,
                out_book,
                md_translations,
                md_headings,
                chapter_documents,
                language,
                title,
            )
            item_by_name[item.get_name()] = new_item
        elif item_type == ebooklib.ITEM_IMAGE:
            _copy_image(item, out_book)
        elif item_type != ebooklib.ITEM_NAVIGATION:
            content = item.get_content()
            if item.media_type == "text/css" and item.get_name().endswith("stylesheet.css"):
                content += CHAPTER_STYLES.encode("utf-8")
            out_book.add_item(epub.EpubItem(
                uid=item.get_id(),
                file_name=item.get_name(),
                media_type=item.media_type,
                content=content,
            ))
    if only_translated:
        _prune_missing_document_links(item_by_name)
    return item_by_name


def _prune_missing_document_links(item_by_name: dict) -> None:
    """Remove navigation entries for documents omitted from a partial EPUB.

    Partial translations can retain a source contents page that mentions later
    chapters or back matter. Those entries must not point to documents that are
    absent from the package.
    """
    included_documents = set(item_by_name)
    for name, item in item_by_name.items():
        soup = BeautifulSoup(item.content, "lxml-xml")
        changed = False
        for link in soup.find_all("a", href=True):
            if link.attrs is None:
                continue
            href = link["href"]
            if not href or href.startswith(("#", "http:", "https:", "mailto:")):
                continue
            target = href.split("#", 1)[0]
            resolved = posixpath.normpath(posixpath.join(posixpath.dirname(name), target))
            if resolved.lower().endswith((".html", ".xhtml")) and resolved not in included_documents:
                entry = (
                    link.find_parent("li")
                    or link.find_parent("div", class_="toc-item")
                    or link.find_parent("p")
                )
                if entry is not None:
                    entry.decompose()
                else:
                    link.unwrap()
                changed = True
        if changed:
            for entry in soup.select("div.toc-item"):
                if entry.find("a", href=True) is None:
                    entry.decompose()
            item.content = str(soup).encode("utf-8")


def _build_toc(source_book: epub.EpubBook, item_by_name: dict, md_titles: dict) -> list:
    """Rebuild TOC from source links, substituting translated titles."""
    toc = []
    documents = {}
    for index, entry in enumerate(source_book.toc, start=1):
        if isinstance(entry, epub.Link):
            xhtml_name, _, fragment = entry.href.partition("#")
            item = item_by_name.get(xhtml_name)
            if item is None:
                continue
            title = md_titles.get(xhtml_name, entry.title)
            href = item.file_name
            if fragment:
                if xhtml_name not in documents:
                    documents[xhtml_name] = BeautifulSoup(item.content, "lxml-xml")
                target = documents[xhtml_name].find(id=fragment)
                if target is not None:
                    href += f"#{fragment}"
                    if xhtml_name in md_titles:
                        heading = target if re.fullmatch(r"h[1-6]", target.name) else target.find(re.compile(r"^h[1-6]$"))
                        if heading:
                            title = heading.get_text(" ", strip=True)
            # Some source EPUBs omit NCX identifiers. EbookLib 0.20 requires
            # every generated navPoint id to be a string.
            toc.append(epub.Link(href, title, entry.uid or f"toc-{index:03d}"))
        else:
            toc.append(entry)
    return toc


def _build_spine(source_book: epub.EpubBook, item_by_name: dict) -> list:
    """Rebuild spine preserving source linear attributes."""
    spine = []
    for item_id, linear in source_book.spine:
        # Find item by id in item_by_name values
        item = next((v for v in item_by_name.values() if v.id == item_id), None)
        if item:
            spine.append((item, linear))
    return spine


def _add_css_class(tag, class_name: str) -> None:
    classes = tag.get("class", [])
    if isinstance(classes, str):
        classes = classes.split()
    if class_name not in classes:
        tag["class"] = [*classes, class_name]


def _is_chapter_title(paragraph) -> bool:
    text = " ".join(paragraph.get_text(" ", strip=True).split())
    return bool(
        re.match(r"^Capítulo\s+\d{3}\s+-", text, re.IGNORECASE)
        and not paragraph.find(["a", "em", "strong"])
    )


def _copy_reflow_support_items(
    source_book: epub.EpubBook,
    out_book: epub.EpubBook,
    language: str,
    title: str,
) -> list:
    """Copy cover, styles, images, and other non-chapter assets."""
    opening_documents = []
    for item in source_book.get_items():
        item_type = item.get_type()
        if item_type == ebooklib.ITEM_DOCUMENT:
            if item.get_name().lower().endswith(("cover.xhtml", "titlepage.xhtml")):
                opening_documents.append(
                    _copy_document(item, out_book, {}, {}, set(), language, title)
                )
        elif item_type == ebooklib.ITEM_IMAGE:
            _copy_image(item, out_book)
        elif item_type != ebooklib.ITEM_NAVIGATION:
            content = item.get_content()
            if item.media_type == "text/css" and item.get_name().endswith("stylesheet.css"):
                content += CHAPTER_STYLES.encode("utf-8")
            out_book.add_item(epub.EpubItem(
                uid=item.get_id(),
                file_name=item.get_name(),
                media_type=item.media_type,
                content=content,
            ))
    return opening_documents


def _style_chapter_document(soup: BeautifulSoup) -> None:
    """Apply ARC 2-like paragraph typography without restyling front matter."""
    paragraphs = soup.find_all("p")
    chapter_titles = [i for i, paragraph in enumerate(paragraphs) if _is_chapter_title(paragraph)]
    if not chapter_titles:
        first_body_paragraph = 0
    else:
        first_title = chapter_titles[0]
        has_front_matter = any(
            paragraph.get_text(" ", strip=True).casefold() == "sumário"
            for paragraph in paragraphs[:first_title]
        )
        first_body_paragraph = first_title if has_front_matter else 0

    for paragraph in paragraphs[first_body_paragraph:]:
        _add_css_class(paragraph, "book-paragraph")
        if _is_chapter_title(paragraph):
            _add_css_class(paragraph, "chapter-title")


def _copy_document(
    item,
    out_book,
    md_translations,
    md_headings,
    chapter_documents,
    language,
    title,
):
    name = item.get_name()
    # EbookLib's EpubHtml.get_content() regenerates the document and drops the
    # original head. Read the stored XHTML bytes directly to retain its CSS.
    content = item.content
    if name in md_translations:
        logger.info("Traduzindo XHTML: %s", name)
        if isinstance(md_translations[name], dict):
            content = replace_paragraphs_at_offsets(content, md_translations[name], language)
        else:
            content = replace_paragraphs_in_xhtml(
                content,
                md_translations[name],
                language,
                md_headings.get(name),
            )
    else:
        logger.info("Copiando XHTML sem tradução: %s", name)
    soup = BeautifulSoup(content, "lxml-xml")
    if name in chapter_documents:
        _style_chapter_document(soup)
    for meta in soup.find_all("meta"):
        if meta.get("http-equiv", "").lower() == "content-style-type":
            meta.decompose()
    for tag in soup.find_all(True):
        for attribute in list(tag.attrs):
            if attribute.startswith("data-") and attribute != attribute.lower():
                value = tag.attrs.pop(attribute)
                tag.attrs.setdefault(attribute.lower(), value)
    html_tag = soup.find("html")
    if html_tag:
        html_tag["lang"] = language
        html_tag["xml:lang"] = language
    title_tag = soup.find("title")
    document_heading = md_headings.get(name)
    if isinstance(document_heading, list):
        document_heading = document_heading[-1] if document_heading else None
    if title_tag and document_heading:
        title_tag.string = document_heading
    elif not title_tag:
        head = soup.find("head")
        if head:
            title_tag = soup.new_tag("title")
            title_tag.string = document_heading or title
            head.append(title_tag)

    # Opening documents are copied outside the Markdown paragraph mapping, but
    # their visible/accessibility title still needs to match the localized book
    # metadata. The cover artwork itself remains the source publisher asset.
    lower_name = name.lower()
    if lower_name.endswith(("title-page.xhtml", "titlepage.xhtml")):
        visible_title = soup.select_one(".title-page-title") or soup.find("h1")
        if visible_title:
            visible_title.string = title
    if lower_name.endswith("cover.xhtml"):
        if title_tag:
            title_tag.string = title
        cover_image = soup.find("img")
        if cover_image:
            cover_image["alt"] = title
    content = str(soup).encode("utf-8")

    # Use a raw XHTML item so EbookLib does not discard the original head,
    # stylesheet links, metadata, or the inline cover-page CSS.
    new_item = epub.EpubItem(
        uid=item.get_id(),
        file_name=name,
        media_type=item.media_type,
        content=content,
    )
    new_item.properties = list(getattr(item, "properties", []))
    if soup.find("svg") and "svg" not in new_item.properties:
        new_item.properties.append("svg")
    out_book.add_item(new_item)
    return new_item


def _copy_image(item, out_book):
    img = epub.EpubImage()
    img.id = item.get_id()
    img.file_name = item.get_name()
    img.media_type = item.media_type
    img.content = item.get_content()
    img.properties = list(getattr(item, "properties", []))
    if item.get_id() == "cover" and "cover-image" not in img.properties:
        img.properties.append("cover-image")
    out_book.add_item(img)


def _distribute_paragraphs(
    paragraphs: list[list[str]],
    xhtml_files: list[str],
    para_counts: list[int],
) -> dict[str, list[list[str]]]:
    """Slice a flat paragraph list across multiple xhtml files using original para counts."""
    result = {}
    offset = 0
    for i, (xhtml_name, count) in enumerate(zip(xhtml_files, para_counts)):
        chunk = paragraphs[offset:] if i == len(xhtml_files) - 1 else paragraphs[offset: offset + count]
        offset += count
        if chunk:
            result[xhtml_name] = chunk
    return result


def _build_translations(
    md_dir: Path, mapping: ChapterMapping | None, source_dir: Path
) -> tuple[dict, dict, dict, set[str]]:
    """Build translated paragraph groups, headings, and TOC titles."""
    md_translations: dict[str, list[list[str]] | dict[int, list[str]]] = {}
    md_headings: dict[str, str | list[str]] = {}
    md_titles: dict[str, str] = {}
    chapter_documents: set[str] = set()
    for md_file in sorted(md_dir.glob("*.md")):
        chapter_info = mapping.get(md_file.name) if mapping else None
        if not chapter_info:
            continue
        content = md_file.read_text(encoding="utf-8")
        heading_counts = chapter_info.get("heading_counts")
        if heading_counts is not None:
            blocks = [block.strip() for block in re.split(r"\n[ \t]*\n+", content) if block.strip()]
            headings, body = _extract_headings(blocks)
            if len(headings) != sum(heading_counts) or len(body) != sum(chapter_info["para_counts"]):
                raise ValueError(f"{md_file.name}: blocos traduzidos não correspondem ao EPUB")
            heading_offset = body_offset = 0
            for name, heading_count, body_count in zip(
                chapter_info["xhtml_files"], heading_counts, chapter_info["para_counts"],
            ):
                document_headings = headings[heading_offset:heading_offset + heading_count]
                md_headings[name] = document_headings
                if document_headings:
                    md_titles[name] = document_headings[-1]
                md_translations[name] = [[block] for block in body[body_offset:body_offset + body_count]]
                heading_offset += heading_count
                body_offset += body_count
            continue
        paragraphs = get_md_paragraphs(content)
        if not paragraphs:
            continue
        if "_Chapter_" in md_file.name:
            chapter_documents.update(chapter_info["xhtml_files"])
        headings, body = _extract_headings(paragraphs)
        body = _expand_markdown_table_cells(body)
        source_md = source_dir / md_file.name
        if source_md.exists():
            source_paragraphs = get_md_paragraphs(source_md.read_text(encoding="utf-8"))
            source_headings, source_body = _extract_headings(source_paragraphs)
            source_body = _expand_markdown_table_cells(source_body)
            if len(headings) != len(source_headings):
                raise ValueError(
                    f"{md_file.name}: {len(headings)} translated headings, "
                    f"expected {len(source_headings)}"
                )
            aligned = _align_translated_paragraphs(source_body, body)
        else:
            expected = sum(chapter_info["para_counts"])
            if len(body) != expected:
                raise ValueError(
                    f"{md_file.name}: {len(body)} translated paragraphs, expected {expected}"
                )
            aligned = [[paragraph] for paragraph in body]
        paragraph_offsets = chapter_info.get("paragraph_offsets")
        if paragraph_offsets:
            aligned_offset = 0
            for xhtml_name, count, offsets in zip(
                chapter_info["xhtml_files"],
                chapter_info["para_counts"],
                paragraph_offsets,
            ):
                chunk = aligned[aligned_offset:aligned_offset + count]
                aligned_offset += count
                if len(chunk) != len(offsets):
                    raise ValueError(
                        f"{md_file.name}: paragraph offsets do not match the chapter mapping"
                    )
                document_translations = md_translations.setdefault(xhtml_name, {})
                if not isinstance(document_translations, dict):
                    raise ValueError(
                        f"{xhtml_name}: cannot mix offset and whole-document mappings"
                    )
                overlap = set(document_translations).intersection(offsets)
                if overlap:
                    raise ValueError(f"{xhtml_name}: duplicate paragraph offsets {sorted(overlap)}")
                document_translations.update(zip(offsets, chunk))
        else:
            chunks = _distribute_paragraphs(
                aligned,
                chapter_info["xhtml_files"],
                chapter_info["para_counts"],
            )
            md_translations.update(chunks)
        first_xhtml = chapter_info["xhtml_files"][0]
        if headings:
            for xhtml_name, heading in zip(chapter_info["xhtml_files"], headings):
                md_titles[xhtml_name] = heading
                md_headings[xhtml_name] = heading
        elif first_xhtml in md_translations:
            md_titles[first_xhtml] = _strip_md_formatting(paragraphs[0])
    return md_translations, md_headings, md_titles, chapter_documents


def build_epub(
    markdown_dir: str,
    output_path: str,
    title: str,
    source_epub_path: str,
    language: str = "pt-BR",
    only_translated: bool = False,
    keep_opening_documents: bool = True,
    *,
    source_markdown_dir: Path | None = None,
):
    """Build a translated epub using source epub structure and translated markdown text."""
    source_book = epub.read_epub(source_epub_path)

    md_dir = Path(markdown_dir)
    mapping = find_chapter_mapping(md_dir)
    source_dir = source_markdown_dir
    if source_dir is None:
        sibling_source = md_dir.parent / "en"
        legacy_source = Path(__file__).resolve().parent.parent / "en" / md_dir.name
        source_dir = sibling_source if sibling_source.is_dir() else legacy_source
    md_translations, md_headings, md_titles, chapter_documents = _build_translations(
        md_dir, mapping, source_dir
    )

    out_book = epub.EpubBook()
    publication_id = uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"codex-translation:{title}:{language}",
    )
    out_book.set_identifier(f"urn:uuid:{publication_id}")
    out_book.set_title(title)
    out_book.set_language(language)

    # Copy metadata from source, except fields explicitly localized above.
    dc_namespace = "http://purl.org/dc/elements/1.1/"
    opf_namespace = "http://www.idpf.org/2007/opf"
    for namespace, values in source_book.metadata.items():
        for name, ns_values in values.items():
            if namespace == dc_namespace and name in {"title", "language", "identifier"}:
                continue
            for value, opts in ns_values:
                if (
                    namespace == opf_namespace
                    and name == "meta"
                    and opts.get("name") == "calibre:title_sort"
                ):
                    continue
                try:
                    # EPUB 2 used opf:role, opf:file-as, and opf:scheme on DC
                    # elements. They are invalid in EPUB 3 unless expressed as
                    # refinements, so preserve the values without legacy attrs.
                    safe_opts = {} if namespace == dc_namespace else opts
                    out_book.add_metadata(namespace, name, value, safe_opts)
                except Exception:
                    pass
    out_book.add_metadata(
        opf_namespace,
        "meta",
        None,
        {"name": "calibre:title_sort", "content": title},
    )

    # PDF-derived books map several logical chapters into the same XHTML files.
    # Rebuild those chapters as independent spine items so reading systems (and
    # Kindle's chapter-duration estimator) can recognize each one separately.
    if mapping and any(chapter.get("paragraph_offsets") for chapter in mapping.values()):
        opening_documents = _copy_reflow_support_items(
            source_book, out_book, language, title
        )
        out_book.add_item(epub.EpubItem(
            uid="chapter-styles",
            file_name="chapter_styles.css",
            media_type="text/css",
            content=CHAPTER_STYLES.encode("utf-8"),
        ))
        opening_documents.extend(build_reflow_frontmatter(md_dir, mapping, out_book, language))
        chapters, toc, merged_breaks = build_reflowed_chapters(
            md_dir, mapping, source_book, out_book, language, source_dir
        )
        out_book.toc = toc
        out_book.add_item(epub.EpubNcx())
        out_book.add_item(epub.EpubNav())
        out_book.spine = [*opening_documents, *chapters]
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        epub.write_epub(output_path, out_book)
        logger.info("Capítulos reconstruídos: %d", len(chapters))
        logger.info("Quebras indevidas removidas: %d", merged_breaks)
        return

    item_by_name = _copy_items(
        source_book,
        out_book,
        md_translations,
        md_headings,
        chapter_documents,
        language,
        title,
        only_translated,
        keep_opening_documents,
    )

    # Rebuild TOC from source, using translated titles where available
    out_book.toc = _build_toc(source_book, item_by_name, md_titles)

    out_book.add_item(epub.EpubNcx())
    out_book.add_item(epub.EpubNav())

    # Preserve spine order and linear attributes from source
    out_book.spine = _build_spine(source_book, item_by_name)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    epub.write_epub(output_path, out_book)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Build epub from translated markdown")
    parser.add_argument("markdown_dir", help="Directory with translated markdown files")
    parser.add_argument("-o", "--output", default=None, help="Output epub path")
    parser.add_argument("-t", "--title", default=None, help="Book title")
    parser.add_argument("-s", "--source", default=None, help="Source epub path")
    parser.add_argument("-l", "--language", default="pt-BR", help="EPUB language tag")
    parser.add_argument(
        "--only-translated",
        action="store_true",
        help="Include only translated chapters and the cover page",
    )
    args = parser.parse_args()

    md_dir = Path(args.markdown_dir)
    book_name = md_dir.name
    output_path = args.output or str(
        Path(__file__).resolve().parent.parent / "output" / f"{book_name}_pt.epub"
    )
    title = args.title or book_name

    source_epub = args.source
    if not source_epub:
        source_dir = Path(__file__).resolve().parent.parent / "source"
        candidates = list(source_dir.glob(f"{book_name}.epub"))
        if not candidates:
            parser.error(
                "Source epub not found in source/. Use -s to specify it."
            )
        source_epub = str(candidates[0])

    print(f"Building from: {md_dir}")
    print(f"Source epub:   {source_epub}")
    build_epub(
        str(md_dir), output_path, title, source_epub, args.language,
        args.only_translated,
    )
    print(f"EPUB gerado em: {output_path}")


if __name__ == "__main__":
    main()
