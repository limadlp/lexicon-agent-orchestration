"""Replace translated text in EPUB XHTML while preserving anchors."""

import copy
import html
import logging
import re

import markdown
from bs4 import BeautifulSoup

from lexicon.epub.alignment import _strip_md_formatting
from lexicon.epub.extractor import localizable_elements

logger = logging.getLogger(__name__)


def _markdown_inline_xhtml(text: str) -> str:
    """Convert markdown inline formatting to XHTML-safe inline HTML."""
    rendered_html = markdown.markdown(text, output_format="xhtml")
    fragment = BeautifulSoup(rendered_html, "html.parser")
    p_tag = fragment.find("p")
    if p_tag:
        return p_tag.decode_contents()
    # A line beginning with ``1.`` is parsed as a one-item Markdown list,
    # even when it came from a single paragraph in an EPUB contents page.
    list_item = fragment.find("li")
    if list_item:
        marker = re.match(r"^\s*(?:\d+[.)]|[-+*])\s+", text)
        prefix = html.escape(marker.group(0)) if marker else ""
        return prefix + list_item.decode_contents()
    return html.escape(text)


def _set_paragraph_text(p, text: str) -> None:
    rendered = _markdown_inline_xhtml(text)
    rendered_fragment = BeautifulSoup(f"<root>{rendered}</root>", "lxml-xml")
    nodes = list(rendered_fragment.root.contents)

    # Source paragraphs can contain a span at a print-page boundary. Replacing
    # only that span leaves the English text on either side in the output.
    # Clear the complete paragraph and retain page anchors as empty markers;
    # inline emphasis and links are rebuilt from the translated Markdown.
    page_anchors = []
    for tag in p.find_all(attrs={"id": True}):
        if re.match(r"^page[_-]", str(tag.get("id", "")), re.IGNORECASE):
            anchor = copy.deepcopy(tag)
            anchor.clear()
            page_anchors.append(anchor)

    p.clear()
    for anchor in page_anchors:
        p.append(anchor)
    for node in nodes:
        p.append(node)


def _replace_paragraph_group(p, translated_group: list[str]) -> None:
    if not translated_group:
        p.decompose()
        return
    _set_paragraph_text(p, translated_group[0])
    insertion_point = p
    for extra_paragraph in translated_group[1:]:
        extra = copy.deepcopy(p)
        _set_paragraph_text(extra, extra_paragraph)
        insertion_point.insert_after(extra)
        insertion_point = extra


def replace_paragraphs_in_xhtml(
    xhtml_bytes: bytes,
    translated: list[list[str]],
    lang: str,
    heading: str | list[str] | None = None,
) -> bytes:
    """Replace text paragraph content in source xhtml with translated text."""
    soup = BeautifulSoup(xhtml_bytes, "lxml-xml")

    html_tag = soup.find("html")
    if html_tag:
        html_tag["lang"] = lang
        html_tag["xml:lang"] = lang

    if isinstance(heading, list):
        elements = localizable_elements(soup)
        headings = [node for node in elements if node.name.startswith("h")]
        paragraphs = [node for node in elements if not node.name.startswith("h")]
        if len(headings) != len(heading) or len(paragraphs) != len(translated):
            raise ValueError("O XHTML não corresponde ao mapa de blocos da tradução")
        for node, text in zip(headings, heading):
            anchors = [copy.deepcopy(tag) for tag in node.find_all(id=True)]
            anchors.extend(copy.deepcopy(tag) for tag in node.find_all("a", attrs={"name": True}) if not tag.has_attr("id"))
            node.clear()
            for anchor in anchors:
                anchor.clear()
                node.append(anchor)
            node.append(text)
        for node, group in zip(paragraphs, translated):
            _set_paragraph_text(node, group[0])
        return str(soup).encode("utf-8")

    text_paras = [
        element
        for element in soup.find_all(["p", "li"])
        if element.get_text(strip=True)
        and not (element.name == "li" and element.find("p"))
    ]
    translated = list(translated)

    # WebToEpub stores author-note headings in a visual span, while the
    # extracted Markdown represents those headings as standalone paragraphs.
    # Remove the extra translated heading from paragraph alignment and use it
    # to localize the visual label instead of overwriting the note body.
    author_note_portlets = soup.select(".author-note-portlet")
    if author_note_portlets and len(translated) > len(text_paras):
        removed = 0
        for portlet in author_note_portlets:
            note_paragraphs = [
                p for p in portlet.find_all("p") if p.get_text(strip=True)
            ]
            if not note_paragraphs:
                continue
            first_note_paragraph = note_paragraphs[0]
            source_index = text_paras.index(first_note_paragraph)
            translated_index = source_index + removed
            if translated_index >= len(translated):
                break
            label_group = translated.pop(translated_index)
            if label_group:
                label = portlet.select_one(".caption-subject")
                if label:
                    label.clear()
                    label.append(_strip_md_formatting(label_group[0]))
            removed += 1

    # Markdown extracted from some EPUBs represents the chapter number and an
    # ornamental break as standalone blocks even though the source XHTML uses
    # non-paragraph elements for them. Keep those source elements in place,
    # but do not let the extracted markers shift every prose paragraph.
    if len(translated) > len(text_paras):
        without_structural_markers = [
            group
            for group in translated
            if not (
                len(group) == 1
                and (
                    _strip_md_formatting(group[0]).isdigit()
                    or _strip_md_formatting(group[0]) == "---"
                )
            )
        ]
        if len(without_structural_markers) == len(text_paras):
            translated = without_structural_markers

    if len(text_paras) != len(translated):
        logger.warning("XHTML com %d parágrafos de origem e %d traduzidos", len(text_paras), len(translated))

    if heading:
        heading_tag = soup.find(re.compile(r"^h[1-6]$"))
        if heading_tag:
            target = heading_tag.find("span") or heading_tag.find("a") or heading_tag
            target.clear()
            target.append(heading)

    for i, p in enumerate(text_paras):
        if i >= len(translated):
            break
        translated_group = translated[i]
        _replace_paragraph_group(p, translated_group)
    return str(soup).encode("utf-8")


def replace_paragraphs_at_offsets(
    xhtml_bytes: bytes,
    translated: dict[int, list[str]],
    lang: str,
) -> bytes:
    """Replace selected paragraphs in an XHTML document by their positions.

    This supports EPUBs where multiple logical chapters occupy one XHTML file.
    ``translated`` is keyed by the index among non-empty source paragraphs.
    """
    soup = BeautifulSoup(xhtml_bytes, "lxml-xml")
    html_tag = soup.find("html")
    if html_tag:
        html_tag["lang"] = lang
        html_tag["xml:lang"] = lang

    text_paras = [
        element
        for element in soup.find_all(["p", "li"])
        if element.get_text(strip=True)
        and not (element.name == "li" and element.find("p"))
    ]
    for paragraph_index, translated_group in translated.items():
        if paragraph_index >= len(text_paras):
            raise ValueError(
                f"Paragraph index {paragraph_index} is outside the XHTML document "
                f"({len(text_paras)} non-empty paragraphs)"
            )
        p = text_paras[paragraph_index]
        _replace_paragraph_group(p, translated_group)
    return str(soup).encode("utf-8")
