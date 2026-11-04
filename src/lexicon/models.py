"""Domain models for the translation pipeline and evaluation suite."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


GenderCode = Literal["M", "F", "N", "M+F"]
EntryKind = Literal["personagem", "termo"]


@dataclass(frozen=True)
class TermEntry:
    """Canonical glossary mapping for worldbuilding, magic, or domain terms."""

    source: str
    target: str
    aliases: tuple[str, ...] = ()
    forbidden_synonyms: tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True)
class CharacterEntry:
    """Canonical character profile used to enforce grammatical gender in target prose."""

    name: str
    target_name: str
    gender: GenderCode
    english_referents: tuple[str, ...] = ()
    expected_markers_pt: tuple[str, ...] = ()
    forbidden_markers_pt: tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True)
class GlossarySnapshot:
    """Immutable read-only view of the canonical lexicon passed to isolated chapter workers."""

    version: int
    terms: tuple[TermEntry, ...]
    characters: tuple[CharacterEntry, ...]


@dataclass(frozen=True)
class TermProposal:
    """Candidate lexical entry proposed by an isolated chapter translator."""

    chapter_number: int
    kind: EntryKind
    source: str
    proposed_target: str
    gender: str = ""
    aliases: tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True)
class ConflictResolution:
    """Record of a terminology collision resolved during Phase 3 consolidation."""

    normalized_source: str
    canonical_target: str
    rejected_targets: tuple[str, ...]
    affected_chapters: tuple[int, ...]
    winning_reason: str


@dataclass(frozen=True)
class ChapterDocument:
    """Input chapter representation."""

    number: int
    filename: str
    title: str
    content: str
    source_path: Path | None = None


@dataclass
class ChapterResult:
    """Output of a single chapter translation lifecycle."""

    chapter: ChapterDocument
    provisional_translation: str
    final_translation: str
    proposals: list[TermProposal] = field(default_factory=list)
    active_terms: list[TermEntry] = field(default_factory=list)
    active_characters: list[CharacterEntry] = field(default_factory=list)
    reconciled: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0


@dataclass(frozen=True)
class TermViolation:
    """Detected violation of canonical terminology in a translated chapter."""

    chapter_number: int
    source_term: str
    expected_target: str
    found_forbidden: str | None = None


@dataclass(frozen=True)
class GenderViolation:
    """Detected grammatical gender mismatch for a known character."""

    chapter_number: int
    character_name: str
    expected_gender: GenderCode
    evidence: str


@dataclass(frozen=True)
class EvalMetrics:
    """Quantitative evaluation metrics across a batch of translated chapters."""

    total_chapters: int
    glossary_adherence_rate: float
    glossary_violation_rate: float
    lexical_drift_bits: float
    gender_agreement_accuracy: float
    structural_parity_rate: float
    chrf_score: float | None = None
    term_violations: tuple[TermViolation, ...] = ()
    gender_violations: tuple[GenderViolation, ...] = ()
    per_term_entropy: dict[str, float] = field(default_factory=dict)
