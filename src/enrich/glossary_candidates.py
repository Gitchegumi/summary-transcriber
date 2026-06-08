from __future__ import annotations

import re
import string
from collections import defaultdict

from src.config.manifest import GlossaryConfig, SessionManifest


COMMON_WORDS = {
    "about",
    "after",
    "again",
    "also",
    "and",
    "because",
    "before",
    "being",
    "could",
    "every",
    "first",
    "going",
    "great",
    "have",
    "just",
    "like",
    "little",
    "maybe",
    "okay",
    "really",
    "right",
    "something",
    "the",
    "there",
    "their",
    "then",
    "they",
    "thing",
    "think",
    "this",
    "that",
    "well",
    "what",
    "when",
    "where",
    "which",
    "will",
    "with",
    "would",
    "you",
    "your",
}


def build_glossary_candidates(
    manifest: SessionManifest,
    turns: list[dict],
    words: list[dict],
    corrections: list[dict],
) -> list[dict]:
    known_terms = {term.lower() for term in manifest.glossary.all_terms()}
    buckets: dict[tuple[str, str], dict] = {}

    for word in words:
        confidence = word.get("confidence")
        term = _clean_term(word.get("word", ""))
        if (
            confidence is None
            or confidence >= 0.75
            or not _useful_term(term)
            or term.lower() in known_terms
        ):
            continue
        turn = _turn_by_id(turns, word["turn_id"])
        _add_candidate(
            buckets,
            session_id=manifest.session.id,
            term=term,
            candidate_type="low_confidence_term",
            suggested_bucket="custom_terms",
            reason="Repeated low-confidence term that may need campaign glossary coverage.",
            example_turn=turn,
            confidence=confidence,
        )

    for turn in turns:
        for term in _capitalized_terms(turn["text_cleaned"]):
            if term.lower() in known_terms or not _useful_term(term):
                continue
            _add_candidate(
                buckets,
                session_id=manifest.session.id,
                term=term,
                candidate_type="possible_name_or_place",
                suggested_bucket="custom_terms",
                reason="Unfamiliar capitalized transcript term; review for names, places, factions, or items.",
                example_turn=turn,
                confidence=turn.get("confidence_avg"),
            )

    for correction in corrections:
        original, corrected = _parse_replacement(correction.get("reason", ""))
        if not original or not corrected or original.lower() in known_terms:
            continue
        turn = _turn_by_id(turns, correction["turn_id"])
        _add_candidate(
            buckets,
            session_id=manifest.session.id,
            term=original,
            candidate_type="correction_source_phrase",
            suggested_bucket=_bucket_for_term(corrected, manifest.glossary),
            reason=f"Rule corrected this phrase to '{corrected}'. Consider adding related campaign terms.",
            example_turn=turn,
            confidence=correction.get("confidence"),
        )

    candidates = [value for value in buckets.values() if value["occurrence_count"] > 1 or value["candidate_type"] == "correction_source_phrase"]
    candidates.sort(key=lambda row: (row["candidate_type"], row["term"].lower()))
    for index, row in enumerate(candidates, start=1):
        row["candidate_id"] = f"{manifest.session.id}-glossary-candidate-{index:05d}"
    return candidates


def _add_candidate(
    buckets: dict[tuple[str, str], dict],
    session_id: str,
    term: str,
    candidate_type: str,
    suggested_bucket: str,
    reason: str,
    example_turn: dict | None,
    confidence: float | None,
) -> None:
    key = (candidate_type, term.lower())
    if key not in buckets:
        buckets[key] = {
            "session_id": session_id,
            "candidate_id": "",
            "term": term,
            "candidate_type": candidate_type,
            "suggested_glossary_bucket": suggested_bucket,
            "reason": reason,
            "example_text": example_turn.get("text_cleaned", "") if example_turn else "",
            "speaker_id": example_turn.get("speaker_id") if example_turn else None,
            "turn_id": example_turn.get("turn_id") if example_turn else None,
            "start_seconds": example_turn.get("start_seconds") if example_turn else None,
            "end_seconds": example_turn.get("end_seconds") if example_turn else None,
            "confidence": confidence,
            "occurrence_count": 0,
        }
    row = buckets[key]
    row["occurrence_count"] += 1
    if confidence is not None:
        existing = row.get("confidence")
        row["confidence"] = confidence if existing is None else min(existing, confidence)


def _clean_term(value: str) -> str:
    return value.strip().strip(string.punctuation + "“”‘’").strip()


def _useful_term(term: str) -> bool:
    normalized = term.lower()
    return (
        len(term) >= 3
        and any(char.isalpha() for char in term)
        and normalized not in COMMON_WORDS
        and not normalized.isdigit()
    )


def _capitalized_terms(text: str) -> list[str]:
    matches = re.findall(r"\b[A-Z][A-Za-z'’-]{2,}(?:\s+[A-Z][A-Za-z'’-]{2,}){0,2}\b", text)
    return [_clean_term(match) for match in matches]


def _parse_replacement(reason: str) -> tuple[str | None, str | None]:
    match = re.search(r":\s*(.+?)\s*->\s*(.+)$", reason)
    if not match:
        return None, None
    return match.group(1).strip(), match.group(2).strip()


def _turn_by_id(turns: list[dict], turn_id: str) -> dict | None:
    return next((turn for turn in turns if turn["turn_id"] == turn_id), None)


def _bucket_for_term(term: str, glossary: GlossaryConfig) -> str:
    term_lower = term.lower()
    if glossary and hasattr(glossary, "entries"):
        for bucket, entries in glossary.entries.items():
            for entry in entries:
                if entry.canonical.lower() == term_lower or any(alias.lower() == term_lower for alias in entry.aliases):
                    return bucket
    if "save" in term_lower or "check" in term_lower:
        return "rules_terms"
    return "custom_terms"
