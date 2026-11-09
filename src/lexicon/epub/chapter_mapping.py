"""Read and write the chapter map shared by EPUB extraction and reconstruction."""

import json
from pathlib import Path
from typing import TypedDict, cast

_FILENAME = "_chapters.json"


class ChapterLocation(TypedDict):
    xhtml_files: list[str]
    para_counts: list[int]


class ChapterMappingEntry(ChapterLocation, total=False):
    paragraph_offsets: list[list[int]]
    heading_counts: list[int]


ChapterMapping = dict[str, ChapterMappingEntry]


def read_chapter_mapping(directory: Path) -> ChapterMapping:
    path = directory / _FILENAME
    contents = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(contents, dict):
        raise ValueError(f"Mapa de capítulos inválido: {path}")
    return cast(ChapterMapping, contents)


def write_chapter_mapping(directory: Path, mapping: ChapterMapping) -> Path:
    path = directory / _FILENAME
    path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def find_chapter_mapping(directory: Path) -> ChapterMapping | None:
    for candidate in (directory, directory.parent / "en", directory.parent):
        try:
            return read_chapter_mapping(candidate)
        except (OSError, ValueError):
            continue
    return None
