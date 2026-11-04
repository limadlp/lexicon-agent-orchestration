"""Single-writer glossary store, snapshot isolation, and symbolic lexical retrieval."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from lexicon.models import (
    CharacterEntry,
    ConflictResolution,
    GenderCode,
    GlossarySnapshot,
    TermEntry,
    TermProposal,
)


def normalize_key(value: str) -> str:
    """Normalize text for case-insensitive, accent-normalized lexical matching."""
    cleaned = re.sub(r"[`*_\"“”]", "", value)
    folded = unicodedata.normalize("NFKC", cleaned).casefold()
    folded = folded.translate(str.maketrans({"’": "'", "–": "-", "—": "-"}))
    return re.sub(r"\s+", " ", folded).strip()


def english_inflections(phrase: str) -> set[str]:
    """Generate common English singular/plural/possessive surface forms."""
    base = normalize_key(phrase)
    if not base:
        return set()
    forms = {base, f"{base}'s"}
    words = base.split()
    last = words[-1]
    if len(last) > 2 and last[-1] == "y" and last[-2] not in "aeiou":
        plural_last = last[:-1] + "ies"
    elif last.endswith(("s", "x", "z", "ch", "sh")):
        plural_last = last + "es"
    else:
        plural_last = last + "s"
    forms.add(" ".join([*words[:-1], plural_last]))
    return forms


def portuguese_inflections(phrase: str) -> set[str]:
    """Generate common Portuguese singular/plural and gender surface forms."""
    base = normalize_key(phrase)
    if not base:
        return set()
    forms = {base}
    for part in re.split(r"\s*/\s*", base):
        part = part.strip()
        if not part:
            continue
        forms.add(part)
        words = part.split()
        last = words[-1]
        plurals = {last + "s"}
        if last.endswith("ão"):
            plurals.update({last[:-2] + "ões", last[:-2] + "ãos", last[:-2] + "ães"})
        elif last.endswith(("al", "el", "ol", "ul")):
            plurals.add(last[:-1] + "is")
        elif last.endswith("m"):
            plurals.add(last[:-1] + "ns")
        elif last.endswith(("r", "z")):
            plurals.add(last + "es")
        for plural in plurals:
            forms.add(" ".join([*words[:-1], plural]))
    return forms


def occurs_in_text(text: str, variants: set[str]) -> bool:
    """Check if any surface form occurs as a whole token/phrase in text."""
    norm_text = normalize_key(text)
    for variant in variants:
        if len(variant) < 2:
            continue
        pattern = rf"(?<!\w){re.escape(variant)}(?!\w)"
        if re.search(pattern, norm_text):
            return True
    return False


class GlossaryStore:
    """Canonical lexicon enforcing single-writer updates and snapshot isolation."""

    def __init__(
        self,
        terms: list[TermEntry] | None = None,
        characters: list[CharacterEntry] | None = None,
    ) -> None:
        self._version = 1
        self._terms: dict[str, TermEntry] = {}
        self._characters: dict[str, CharacterEntry] = {}

        for term in terms or []:
            self._terms[normalize_key(term.source)] = term
        for char in characters or []:
            self._characters[normalize_key(char.name)] = char

    @property
    def version(self) -> int:
        return self._version

    def snapshot(self) -> GlossarySnapshot:
        """Return an immutable snapshot for isolated parallel workers."""
        return GlossarySnapshot(
            version=self._version,
            terms=tuple(sorted(self._terms.values(), key=lambda t: normalize_key(t.source))),
            characters=tuple(sorted(self._characters.values(), key=lambda c: normalize_key(c.name))),
        )

    @classmethod
    def from_snapshot(cls, snapshot: GlossarySnapshot) -> GlossaryStore:
        store = cls(terms=list(snapshot.terms), characters=list(snapshot.characters))
        store._version = snapshot.version
        return store

    def commit(self, staged: GlossaryStore, *, expected_version: int) -> None:
        """Publish a completed batch's glossary changes in one synchronous step."""
        if self._version != expected_version:
            raise RuntimeError("O glossário mudou durante a tradução do lote")
        self._terms = staged._terms.copy()
        self._characters = staged._characters.copy()
        self._version = staged._version

    @staticmethod
    def retrieve_chapter_context(
        snapshot: GlossarySnapshot,
        chapter_text: str,
    ) -> tuple[list[TermEntry], list[CharacterEntry]]:
        """Symbolic RAG: retrieve only terms and characters present in the source chapter."""
        matched_terms: list[TermEntry] = []
        for term in snapshot.terms:
            candidates = set()
            for token in (term.source, *term.aliases):
                candidates.update(english_inflections(token))
            if occurs_in_text(chapter_text, candidates):
                matched_terms.append(term)

        matched_characters: list[CharacterEntry] = []
        for char in snapshot.characters:
            candidates = set()
            for referent in (char.name, *char.english_referents):
                candidates.update(english_inflections(referent))
            if occurs_in_text(chapter_text, candidates):
                matched_characters.append(char)

        return matched_terms, matched_characters

    @staticmethod
    def format_context_directive(
        terms: list[TermEntry],
        characters: list[CharacterEntry],
    ) -> str:
        """Format retrieved entities into a compact prompt constraint block."""
        lines: list[str] = []
        if characters:
            lines.append("### Restrições de Gênero e Personagens Ativos")
            for char in characters:
                gender_label = {
                    "F": "Feminino (usar 'a', 'ela', adjetivos no feminino)",
                    "M": "Masculino (usar 'o', 'ele', adjetivos no masculino)",
                    "N": "Neutro / Sem gênero marcado",
                    "M+F": "Plural misto",
                }.get(char.gender, char.gender)
                referents = (
                    f" [referências EN: {', '.join(char.english_referents)}]"
                    if char.english_referents
                    else ""
                )
                lines.append(
                    f"- {char.name} -> {char.target_name} | Gênero: {gender_label}{referents}"
                )
            lines.append("")

        if terms:
            lines.append("### Terminologia Canônica Obrigatória")
            for term in terms:
                forbidden = (
                    f" (PROIBIDO usar: {', '.join(term.forbidden_synonyms)})"
                    if term.forbidden_synonyms
                    else ""
                )
                lines.append(f"- {term.source} -> {term.target}{forbidden}")
            lines.append("")

        return "\n".join(lines).strip()

    def consolidate_proposals(
        self,
        proposals: list[TermProposal],
    ) -> list[ConflictResolution]:
        """Single-writer Reduce phase: resolve conflicting worker proposals deterministically.

        Resolution precedence:
        1. Canonical glossary entry already established prior to the batch always wins.
        2. For genuinely new entries with competing proposals across parallel chapters,
           the proposal from the numerically earliest chapter wins (never completion time).
        """
        grouped: dict[str, list[TermProposal]] = defaultdict(list)
        for proposal in proposals:
            key = normalize_key(proposal.source)
            if key:
                grouped[key].append(proposal)

        resolutions: list[ConflictResolution] = []
        mutated = False

        for key in sorted(grouped):
            group = sorted(
                grouped[key],
                key=lambda p: (p.chapter_number, normalize_key(p.proposed_target)),
            )
            sample = group[0]

            if sample.kind == "personagem":
                existing_char = self._characters.get(key)
                if existing_char is not None:
                    canonical_target = existing_char.target_name
                    reason = "established_in_canonical_glossary"
                else:
                    canonical_target = sample.proposed_target
                    reason = f"earliest_chapter_appearance (ch.{sample.chapter_number})"
                    gender_code: GenderCode = (
                        sample.gender if sample.gender in ("M", "F", "N", "M+F") else "M"
                    )
                    all_aliases = tuple(
                        sorted({alias for p in group for alias in p.aliases})
                    )
                    self._characters[key] = CharacterEntry(
                        name=sample.source,
                        target_name=canonical_target,
                        gender=gender_code,
                        english_referents=all_aliases,
                        notes=sample.notes,
                    )
                    mutated = True
            else:
                existing_term = self._terms.get(key)
                if existing_term is not None:
                    canonical_target = existing_term.target
                    reason = "established_in_canonical_glossary"
                else:
                    canonical_target = sample.proposed_target
                    reason = f"earliest_chapter_appearance (ch.{sample.chapter_number})"
                    all_aliases = tuple(
                        sorted({alias for p in group for alias in p.aliases})
                    )
                    rejected_forms = tuple(
                        sorted(
                            {
                                p.proposed_target
                                for p in group
                                if normalize_key(p.proposed_target)
                                != normalize_key(canonical_target)
                            }
                        )
                    )
                    self._terms[key] = TermEntry(
                        source=sample.source,
                        target=canonical_target,
                        aliases=all_aliases,
                        forbidden_synonyms=rejected_forms,
                        notes=sample.notes,
                    )
                    mutated = True

            rejected: list[str] = []
            affected_chapters: list[int] = []
            for prop in group:
                if normalize_key(prop.proposed_target) != normalize_key(canonical_target):
                    if prop.proposed_target not in rejected:
                        rejected.append(prop.proposed_target)
                    if prop.chapter_number not in affected_chapters:
                        affected_chapters.append(prop.chapter_number)

            if affected_chapters:
                resolutions.append(
                    ConflictResolution(
                        normalized_source=key,
                        canonical_target=canonical_target,
                        rejected_targets=tuple(rejected),
                        affected_chapters=tuple(sorted(affected_chapters)),
                        winning_reason=reason,
                    )
                )

        if mutated:
            self._version += 1

        return resolutions

    def to_markdown(self) -> str:
        """Export the canonical glossary to Markdown tables."""
        lines = [
            "# Glossário Canônico (EN -> PT-BR)",
            "",
            "## Personagens",
            "",
            "| Nome original | Forma estabelecida | Gênero/sexo | Aliases | Notas |",
            "| --- | --- | --- | --- | --- |",
        ]
        for char in sorted(self._characters.values(), key=lambda c: normalize_key(c.name)):
            aliases = "; ".join(char.english_referents)
            lines.append(
                f"| {char.name} | {char.target_name} | {char.gender} | {aliases} | {char.notes} |"
            )

        lines.extend(
            [
                "",
                "## Termos do mundo",
                "",
                "| Original | Tradução estabelecida | Aliases | Proibidos | Notas |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for term in sorted(self._terms.values(), key=lambda t: normalize_key(t.source)):
            aliases = "; ".join(term.aliases)
            forbidden = "; ".join(term.forbidden_synonyms)
            lines.append(
                f"| {term.source} | {term.target} | {aliases} | {forbidden} | {term.notes} |"
            )

        return "\n".join(lines) + "\n"

    @classmethod
    def from_markdown(cls, path: Path) -> GlossaryStore:
        """Load a GlossaryStore from a Markdown glossary file."""
        text = path.read_text(encoding="utf-8")
        terms: list[TermEntry] = []
        characters: list[CharacterEntry] = []
        section = ""
        headers: list[str] | None = None

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if line.startswith("## "):
                section = line[3:].strip()
                headers = None
                continue
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(re.fullmatch(r":?-{3,}:?", c) for c in cells if c):
                continue
            if headers is None:
                headers = [normalize_key(c) for c in cells]
                continue
            row = dict(zip(headers, cells))
            if "personagens" in normalize_key(section):
                name = row.get("nome original") or row.get("nome", "")
                target = row.get("forma estabelecida") or name
                gender_raw = (row.get("gênero/sexo") or row.get("genero/sexo") or row.get("gênero") or "M").strip()
                gender: GenderCode = gender_raw if gender_raw in ("M", "F", "N", "M+F") else "M"
                aliases = tuple(
                    a.strip() for a in re.split(r"[;,]", row.get("aliases", "")) if a.strip()
                )
                notes = row.get("notas", "")
                if name:
                    characters.append(
                        CharacterEntry(
                            name=name,
                            target_name=target,
                            gender=gender,
                            english_referents=aliases,
                            notes=notes,
                        )
                    )
            elif "termos" in normalize_key(section):
                source = row.get("original") or row.get("english", "")
                target = row.get("tradução estabelecida") or row.get("português", "")
                aliases = tuple(
                    a.strip() for a in re.split(r"[;,]", row.get("aliases", "")) if a.strip()
                )
                forbidden = tuple(
                    f.strip() for f in re.split(r"[;,]", row.get("proibidos", "")) if f.strip()
                )
                notes = row.get("notas") or row.get("notes", "")
                if source and target:
                    terms.append(
                        TermEntry(
                            source=source,
                            target=target,
                            aliases=aliases,
                            forbidden_synonyms=forbidden,
                            notes=notes,
                        )
                    )
        return cls(terms=terms, characters=characters)
