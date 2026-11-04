"""Quantitative evaluation suite for literary translation quality and lexical consistency."""

from __future__ import annotations

import math
import re
from collections import Counter

from lexicon.glossary import (
    GlossaryStore,
    english_inflections,
    normalize_key,
    occurs_in_text,
    portuguese_inflections,
)
from lexicon.models import (
    ChapterDocument,
    EvalMetrics,
    GenderViolation,
    GlossarySnapshot,
    TermViolation,
)


def compute_chrf(
    hypothesis: str,
    reference: str,
    max_n: int = 6,
    beta: float = 2.0,
) -> float:
    """Compute standard character n-gram F-score (chrF, Popovic 2015) on [0, 100] scale.

    chrF is particularly effective for morphologically rich target languages like Portuguese
    because it captures stem and inflection overlap without requiring external tokenizers.
    """
    hyp = re.sub(r"\s+", "", normalize_key(hypothesis))
    ref = re.sub(r"\s+", "", normalize_key(reference))
    if not hyp and not ref:
        return 100.0
    if not hyp or not ref:
        return 0.0

    total_precision = 0.0
    total_recall = 0.0
    valid_orders = 0

    for n in range(1, max_n + 1):
        if len(hyp) < n or len(ref) < n:
            continue
        hyp_ngrams = Counter(hyp[i : i + n] for i in range(len(hyp) - n + 1))
        ref_ngrams = Counter(ref[i : i + n] for i in range(len(ref) - n + 1))

        overlap = sum((hyp_ngrams & ref_ngrams).values())
        total_hyp = sum(hyp_ngrams.values())
        total_ref = sum(ref_ngrams.values())

        prec = overlap / total_hyp if total_hyp else 0.0
        rec = overlap / total_ref if total_ref else 0.0
        total_precision += prec
        total_recall += rec
        valid_orders += 1

    if valid_orders == 0:
        return 0.0

    avg_p = total_precision / valid_orders
    avg_r = total_recall / valid_orders
    if avg_p + avg_r == 0.0:
        return 0.0

    beta_sq = beta**2
    score = ((1 + beta_sq) * avg_p * avg_r) / (beta_sq * avg_p + avg_r)
    return round(score * 100.0, 2)


def shannon_entropy_bits(counts: Counter[str]) -> float:
    """Compute Shannon entropy H(X) in bits for a frequency distribution."""
    total = sum(counts.values())
    if total <= 1 or len(counts) <= 1:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return round(entropy, 4)


class TranslationEvaluator:
    """Evaluates translated chapters on terminology consistency, gender agreement, and chrF."""

    def __init__(self, snapshot: GlossarySnapshot) -> None:
        self.snapshot = snapshot

    def evaluate(
        self,
        chapters: list[ChapterDocument],
        translations: dict[int, str],
        references: dict[int, str] | None = None,
    ) -> EvalMetrics:
        term_checks_total = 0
        term_checks_passed = 0
        term_violations: list[TermViolation] = []

        gender_checks_total = 0
        gender_checks_passed = 0
        gender_violations: list[GenderViolation] = []

        structural_matches = 0
        term_variant_counts: dict[str, Counter[str]] = {
            t.source: Counter() for t in self.snapshot.terms
        }

        for ch in chapters:
            translated = translations.get(ch.number, "")
            active_terms, active_chars = GlossaryStore.retrieve_chapter_context(
                self.snapshot, ch.content
            )

            # 1. Terminology Adherence & Drift Tracking
            for term in active_terms:
                term_checks_total += 1
                expected_forms = portuguese_inflections(term.target)
                has_canonical = occurs_in_text(translated, expected_forms)

                found_forbidden: str | None = None
                for forbidden in term.forbidden_synonyms:
                    if occurs_in_text(translated, portuguese_inflections(forbidden)):
                        found_forbidden = forbidden
                        break

                if has_canonical:
                    term_variant_counts[term.source][normalize_key(term.target)] += 1
                if found_forbidden:
                    term_variant_counts[term.source][normalize_key(found_forbidden)] += 1

                if has_canonical and found_forbidden is None:
                    term_checks_passed += 1
                else:
                    term_violations.append(
                        TermViolation(
                            chapter_number=ch.number,
                            source_term=term.source,
                            expected_target=term.target,
                            found_forbidden=found_forbidden,
                        )
                    )

            # 2. Character Gender Agreement
            for char in active_chars:
                if not char.expected_markers_pt and not char.forbidden_markers_pt:
                    continue
                gender_checks_total += 1
                found_bad_marker: str | None = None
                for bad_marker in char.forbidden_markers_pt:
                    if occurs_in_text(translated, {normalize_key(bad_marker)}):
                        found_bad_marker = bad_marker
                        break

                has_good_marker = True
                if char.expected_markers_pt:
                    has_good_marker = any(
                        occurs_in_text(translated, {normalize_key(good)})
                        for good in char.expected_markers_pt
                    )

                if has_good_marker and found_bad_marker is None:
                    gender_checks_passed += 1
                else:
                    gender_violations.append(
                        GenderViolation(
                            chapter_number=ch.number,
                            character_name=char.name,
                            expected_gender=char.gender,
                            evidence=found_bad_marker or f"ausência de marcador esperado ({char.expected_markers_pt})",
                        )
                    )

            # 3. Structural Parity (Paragraph Count Alignment)
            src_paragraphs = [p for p in ch.content.strip().split("\n\n") if p.strip()]
            tgt_paragraphs = [p for p in translated.strip().split("\n\n") if p.strip()]
            if len(src_paragraphs) == len(tgt_paragraphs):
                structural_matches += 1

        adherence_rate = (
            term_checks_passed / term_checks_total if term_checks_total > 0 else 1.0
        )
        violation_rate = 1.0 - adherence_rate

        per_term_entropy = {
            src: shannon_entropy_bits(counts)
            for src, counts in term_variant_counts.items()
            if sum(counts.values()) > 0
        }
        mean_drift_bits = (
            sum(per_term_entropy.values()) / len(per_term_entropy)
            if per_term_entropy
            else 0.0
        )

        gender_accuracy = (
            gender_checks_passed / gender_checks_total
            if gender_checks_total > 0
            else 1.0
        )
        structural_rate = structural_matches / len(chapters) if chapters else 1.0

        chrf_avg: float | None = None
        if references:
            scores = [
                compute_chrf(translations.get(ch.number, ""), references[ch.number])
                for ch in chapters
                if ch.number in references
            ]
            if scores:
                chrf_avg = round(sum(scores) / len(scores), 2)

        return EvalMetrics(
            total_chapters=len(chapters),
            glossary_adherence_rate=round(adherence_rate * 100.0, 2),
            glossary_violation_rate=round(violation_rate * 100.0, 2),
            lexical_drift_bits=round(mean_drift_bits, 4),
            gender_agreement_accuracy=round(gender_accuracy * 100.0, 2),
            structural_parity_rate=round(structural_rate * 100.0, 2),
            chrf_score=chrf_avg,
            term_violations=tuple(term_violations),
            gender_violations=tuple(gender_violations),
            per_term_entropy=per_term_entropy,
        )
