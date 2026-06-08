from __future__ import annotations

import csv
import re
from pathlib import Path

from src.config.manifest import SessionManifest


# Base combat keywords always included regardless of glossary contents.
_BASE_KEYWORDS = [
    # core mechanics
    "initiative", "attack", "damage", "hit", "miss", "kill", "dead", "crit",
    r"saving\s+throw", "spell", "heal", "hp",
    # common spells / features
    r"guiding\s+bolt", r"eldritch\s+blast", "fireball",
    r"cone\s+of\s+silence", r"bubble\s+of\s+silence",
    r"time\s+stop", r"plane\s+shift",
    r"divine\s+smite", r"spiritual\s+weapon", r"sneak\s+attack",
    r"shield\s+of\s+faith", r"charm\s+person", "hexblade", r"wild\s+shape",
    # action economy
    r"action\s+economy", r"bonus\s+action", "reaction",
    # class features (fallback when not in glossary)
    r"starry\s+form", r"giant's\s+might", r"cosmic\s+omen",
    # creature types
    "zombie", "skeleton", "undead", "specter", "wraith", "cultist", "bandit",
    # combat structure
    r"combat\s+round", r"round\s+\d+", r"initiative\s+order",
]

# Columns to keep from a transcript turn for downstream review.
COMBAT_FIELDS = [
    "session_id",
    "turn_id",
    "speaker_id",
    "speaker_name",
    "character_name",
    "start_seconds",
    "end_seconds",
    "duration_seconds",
    "text_raw",
    "text_cleaned",
    "confidence_avg",
    "source_file",
    "backend",
    "model",
]


def _build_combat_pattern(glossary_terms: list[str]) -> re.Pattern:
    """Compile a regex from base keywords + glossary terms.

    Glossary terms are escaped and wrapped with word boundaries so multi-word
    phrases (e.g. ``Hunger of Hadar``) and single-word terms (e.g. ``Fireball``)
    match precisely without colliding with substring false positives.
    """
    parts = list(_BASE_KEYWORDS)
    for term in glossary_terms:
        term = term.strip()
        if not term:
            continue
        parts.append(rf"\b{re.escape(term)}\b")
    return re.compile("|".join(parts), re.IGNORECASE)


def export_combat_extract(
    output_dir: Path,
    manifest: SessionManifest,
    turns: list[dict],
    pattern: re.Pattern | None = None,
) -> Path:
    """Write a CSV of turns whose cleaned or raw text matches combat keywords.

    If *pattern* is not supplied, the regex is built automatically from the
    manifest's loaded glossary (all canonical names and aliases across every
    glossary source) merged with the built-in base keyword set.

    Parameters
    ----------
    output_dir:
        Directory where ``combat_extract.csv`` will be written.
    manifest:
        Session manifest.  Its ``glossary`` field drives the keyword set when
        *pattern* is ``None``.
    turns:
        Normalized transcript turns (same shape as ``export_nocodb`` input).
    pattern:
        Optional compiled regex overriding the default combat keyword set.

    Returns
    -------
    Path to the written CSV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    matcher = pattern or _build_combat_pattern(manifest.glossary.all_terms())

    rows: list[dict] = []
    for turn in turns:
        text = (turn.get("text_cleaned") or "") + " " + (turn.get("text_raw") or "")
        if not text.strip():
            continue
        if matcher.search(text):
            rows.append({k: turn.get(k) for k in COMBAT_FIELDS})

    csv_path = output_dir / "combat_extract.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COMBAT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return csv_path
