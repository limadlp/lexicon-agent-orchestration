"""Markdown blocks and paragraph alignment for EPUB reconstruction."""

import math
import re


def _strip_md_formatting(text: str) -> str:
    """Strip markdown heading markers and bold/italic markers from a string."""
    text = re.sub(r"^#+\s*", "", text)
    text = re.sub(r"\*+([^*]+)\*+", r"\1", text)
    text = re.sub(r"_+([^_]+)_+", r"\1", text)
    return text.strip()


def get_md_paragraphs(md_content: str) -> list[str]:
    """Extract logical markdown paragraphs, joining wrapped lines inside each block."""
    paragraphs = []
    # A quote-only Markdown line separates paragraphs inside a blockquote. It
    # is semantically blank even though it contains a ``>`` marker.
    md_content = re.sub(r"(?m)^[ \t]*>[ \t]*$", "", md_content)
    blocks = re.split(r"\n\s*\n+", md_content)
    for block in blocks:
        stripped = block.strip()
        if not stripped:
            continue
        # Skip xml declaration artifacts that can appear in extracted markdown
        if re.match(r"^xml version", stripped, re.IGNORECASE):
            continue
        stripped_lines = [line.strip() for line in stripped.splitlines()]
        if len(stripped_lines) > 1 and all(
            re.match(r"^\d+[.)]\s+", line) for line in stripped_lines
        ):
            paragraphs.extend(stripped_lines)
            continue
        if len(stripped_lines) > 1 and all(line.startswith("|") for line in stripped_lines):
            normalized = "\n".join(stripped_lines)
        else:
            normalized = " ".join(stripped_lines)
        if normalized:
            paragraphs.append(normalized)
    return paragraphs


def _split_heading(paragraphs: list[str]) -> tuple[str | None, list[str]]:
    """Separate an initial Markdown heading from the chapter body."""
    if paragraphs and re.match(r"^#{1,6}\s+", paragraphs[0]):
        return _strip_md_formatting(paragraphs[0]), paragraphs[1:]
    return None, paragraphs


def _extract_headings(paragraphs: list[str]) -> tuple[list[str], list[str]]:
    """Extract Markdown headings while retaining the order of body paragraphs."""
    headings = []
    body = []
    for paragraph in paragraphs:
        if re.match(r"^#{1,6}\s+", paragraph):
            headings.append(_strip_md_formatting(paragraph))
        else:
            body.append(paragraph)
    return headings, body


def _expand_markdown_table_cells(paragraphs: list[str]) -> list[str]:
    """Expand Markdown table blocks to match cell paragraphs in source XHTML."""
    expanded = []
    separator_cell = re.compile(r"^:?-{3,}:?$")
    for paragraph in paragraphs:
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if len(lines) < 2 or not all(line.startswith("|") and line.endswith("|") for line in lines):
            expanded.append(paragraph)
            continue

        rows = []
        is_table = False
        for line in lines:
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if cells and all(not cell or separator_cell.fullmatch(cell) for cell in cells):
                is_table = True
                continue
            rows.append(cells)
        if not is_table:
            expanded.append(paragraph)
            continue
        expanded.extend(cell for row in rows for cell in row if cell)
    return expanded


def _paragraph_length(text: str) -> int:
    """Return a formatting-independent length useful for paragraph alignment."""
    return len(_strip_md_formatting(re.sub(r"\[([^]]+)]\([^)]+\)", r"\1", text)))


def _align_translated_paragraphs(source: list[str], translated: list[str]) -> list[list[str]]:
    """Align translated blocks to source paragraphs, retaining translated splits.

    Most chapters are one-to-one. A few translations intentionally split a source
    paragraph into several blocks. In that case, dynamic programming uses relative
    text lengths to decide where the additional blocks belong.
    """
    if len(source) == len(translated):
        return [[paragraph] for paragraph in translated]
    if not source:
        return []
    if len(translated) < len(source):
        return _align_merged_translated_paragraphs(source, translated)

    source_lengths = [_paragraph_length(text) + 1 for text in source]
    translated_lengths = [_paragraph_length(text) + 1 for text in translated]
    scale = sum(translated_lengths) / sum(source_lengths)
    source_count = len(source)
    translated_count = len(translated)
    extra = translated_count - source_count
    max_group = min(extra + 1, 8)

    infinity = float("inf")
    costs = [[infinity] * (translated_count + 1) for _ in range(source_count + 1)]
    previous: list[list[tuple[int, int] | None]] = [
        [None] * (translated_count + 1) for _ in range(source_count + 1)
    ]
    costs[0][0] = 0.0

    for source_index in range(source_count):
        remaining_source = source_count - source_index - 1
        for translated_index in range(translated_count + 1):
            if not math.isfinite(costs[source_index][translated_index]):
                continue
            max_here = min(
                max_group,
                translated_count - translated_index - remaining_source,
            )
            for group_size in range(1, max_here + 1):
                next_index = translated_index + group_size
                group_length = sum(translated_lengths[translated_index:next_index])
                ratio = group_length / source_lengths[source_index]
                length_cost = abs(math.log(max(ratio / scale, 1e-9)))
                split_penalty = 0.12 * (group_size - 1)
                new_cost = costs[source_index][translated_index] + length_cost + split_penalty
                if new_cost < costs[source_index + 1][next_index]:
                    costs[source_index + 1][next_index] = new_cost
                    previous[source_index + 1][next_index] = (translated_index, group_size)

    if not math.isfinite(costs[source_count][translated_count]):
        raise ValueError("Could not align translated paragraphs to the source")

    aligned: list[list[str]] = []
    source_index = source_count
    translated_index = translated_count
    while source_index:
        step = previous[source_index][translated_index]
        if step is None:
            raise ValueError("Incomplete paragraph alignment")
        start, group_size = step
        aligned.append(translated[start:start + group_size])
        translated_index = start
        source_index -= 1
    aligned.reverse()
    return aligned


def _align_merged_translated_paragraphs(
    source: list[str], translated: list[str]
) -> list[list[str]]:
    """Align translations that merged adjacent source paragraphs.

    The result retains one entry per source paragraph. An empty entry marks a
    source paragraph absorbed into the preceding translated paragraph, allowing
    the XHTML replacement step to remove that now-redundant paragraph node.
    """
    source_lengths = [_paragraph_length(text) + 1 for text in source]
    translated_lengths = [_paragraph_length(text) + 1 for text in translated]
    scale = sum(translated_lengths) / sum(source_lengths)
    source_count = len(source)
    translated_count = len(translated)
    extra = source_count - translated_count
    max_group = min(extra + 1, 8)

    infinity = float("inf")
    costs = [[infinity] * (source_count + 1) for _ in range(translated_count + 1)]
    previous: list[list[tuple[int, int] | None]] = [
        [None] * (source_count + 1) for _ in range(translated_count + 1)
    ]
    costs[0][0] = 0.0

    for translated_index in range(translated_count):
        remaining_translated = translated_count - translated_index - 1
        for source_index in range(source_count + 1):
            if not math.isfinite(costs[translated_index][source_index]):
                continue
            max_here = min(
                max_group,
                source_count - source_index - remaining_translated,
            )
            for group_size in range(1, max_here + 1):
                next_index = source_index + group_size
                group_length = sum(source_lengths[source_index:next_index])
                ratio = translated_lengths[translated_index] / group_length
                length_cost = abs(math.log(max(ratio / scale, 1e-9)))
                merge_penalty = 0.12 * (group_size - 1)
                new_cost = costs[translated_index][source_index] + length_cost + merge_penalty
                if new_cost < costs[translated_index + 1][next_index]:
                    costs[translated_index + 1][next_index] = new_cost
                    previous[translated_index + 1][next_index] = (source_index, group_size)

    if not math.isfinite(costs[translated_count][source_count]):
        raise ValueError("Could not align merged translated paragraphs to the source")

    groups: list[tuple[int, int, str]] = []
    translated_index = translated_count
    source_index = source_count
    while translated_index:
        step = previous[translated_index][source_index]
        if step is None:
            raise ValueError("Incomplete merged paragraph alignment")
        start, group_size = step
        groups.append((start, group_size, translated[translated_index - 1]))
        source_index = start
        translated_index -= 1
    groups.reverse()

    aligned: list[list[str]] = [[] for _ in source]
    for start, _group_size, paragraph in groups:
        aligned[start] = [paragraph]
    return aligned


def _align_paragraphs_flexibly(
    source: list[str], translated: list[str]
) -> list[list[str]]:
    """Align local split/merge shifts even when total paragraph counts match."""
    if not source:
        return []
    if not translated:
        return [[] for _ in source]

    # Chapter titles are a reliable anchor and should never participate in a
    # prose split/merge decision.
    anchored = [[translated[0]]]
    source = source[1:]
    translated = translated[1:]
    if not source:
        return anchored

    source_lengths = [_paragraph_length(text) + 1 for text in source]
    translated_lengths = [_paragraph_length(text) + 1 for text in translated]
    scale = sum(translated_lengths) / sum(source_lengths)
    source_count = len(source)
    translated_count = len(translated)
    infinity = float("inf")
    costs = [[infinity] * (translated_count + 1) for _ in range(source_count + 1)]
    previous: list[list[tuple[int, int] | None]] = [
        [None] * (translated_count + 1) for _ in range(source_count + 1)
    ]
    costs[0][0] = 0.0
    max_group = 4

    for source_index in range(source_count + 1):
        for translated_index in range(translated_count + 1):
            if not math.isfinite(costs[source_index][translated_index]):
                continue
            for source_group in range(1, min(max_group, source_count - source_index) + 1):
                for translated_group in range(
                    1, min(max_group, translated_count - translated_index) + 1
                ):
                    if source_group > 1 and translated_group > 1:
                        continue
                    next_source = source_index + source_group
                    next_translated = translated_index + translated_group
                    source_length = sum(source_lengths[source_index:next_source])
                    translated_length = sum(
                        translated_lengths[translated_index:next_translated]
                    )
                    ratio = translated_length / source_length
                    length_cost = abs(math.log(max(ratio / scale, 1e-9)))
                    grouping_cost = 0.14 * (source_group + translated_group - 2)
                    boundary_cost = 0.0
                    for offset in range(source_group - 1):
                        left = source_index + offset
                        if not _is_false_paragraph_break(source[left], source[left + 1]):
                            boundary_cost += 0.35
                    for offset in range(translated_group - 1):
                        left = translated_index + offset
                        if not _is_false_paragraph_break(
                            translated[left], translated[left + 1]
                        ):
                            boundary_cost += 0.35
                    new_cost = (
                        costs[source_index][translated_index]
                        + length_cost
                        + grouping_cost
                        + boundary_cost
                    )
                    if new_cost < costs[next_source][next_translated]:
                        costs[next_source][next_translated] = new_cost
                        previous[next_source][next_translated] = (
                            source_group,
                            translated_group,
                        )

    if not math.isfinite(costs[source_count][translated_count]):
        raise ValueError("Could not flexibly align translated paragraphs")

    steps = []
    source_index = source_count
    translated_index = translated_count
    while source_index or translated_index:
        step = previous[source_index][translated_index]
        if step is None:
            raise ValueError("Incomplete flexible paragraph alignment")
        source_group, translated_group = step
        steps.append(
            (
                source_index - source_group,
                source_group,
                translated_index - translated_group,
                translated_group,
            )
        )
        source_index -= source_group
        translated_index -= translated_group
    steps.reverse()

    aligned: list[list[str]] = [[] for _ in source]
    for source_start, _source_group, translated_start, translated_group in steps:
        aligned[source_start] = translated[
            translated_start:translated_start + translated_group
        ]
    return anchored + aligned
