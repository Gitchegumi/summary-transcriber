from __future__ import annotations

import csv
import json
from pathlib import Path
import yaml


GLOSSARY_CANDIDATE_FIELDS = [
    "session_id",
    "candidate_id",
    "term",
    "candidate_type",
    "suggested_glossary_bucket",
    "reason",
    "example_text",
    "speaker_id",
    "turn_id",
    "start_seconds",
    "end_seconds",
    "confidence",
    "occurrence_count",
]


def export_glossary_candidates(output_dir: Path, candidates: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "glossary_candidates.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=GLOSSARY_CANDIDATE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(candidates)

    json_path = output_dir / "glossary_candidates.json"
    with json_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(candidates, f, ensure_ascii=False, indent=2)


def export_draft_outputs(
    output_dir: Path,
    candidates: list[dict],
    turns: list[dict],
    words: list[dict],
) -> None:
    draft_dir = output_dir / "draft"
    draft_dir.mkdir(parents=True, exist_ok=True)

    # 1. transcript_turns.draft.csv
    turns_path = draft_dir / "transcript_turns.draft.csv"
    from src.exports.nocodb import TURN_FIELDS
    with turns_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TURN_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(turns)

    # 2. transcript_words.draft.csv
    words_path = draft_dir / "transcript_words.draft.csv"
    from src.exports.nocodb import WORD_FIELDS
    with words_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=WORD_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(words)

    # 3. unknown_terms_report.csv
    unknown_path = draft_dir / "unknown_terms_report.csv"
    with unknown_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=GLOSSARY_CANDIDATE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(candidates)

    # 4. candidate_glossary.yaml
    candidate_yaml_path = draft_dir / "candidate_glossary.yaml"
    glossary_data: dict[str, list[dict]] = {}
    for candidate in candidates:
        bucket = candidate.get("suggested_glossary_bucket") or "custom_terms"
        if bucket not in glossary_data:
            glossary_data[bucket] = []
        
        # Build structured entry
        evidence = f"Turn {candidate.get('turn_id', '')}: \"{candidate.get('example_text', '')}\""
        entry = {
            "canonical": candidate["term"],
            "type": bucket.rstrip("s"),  # e.g., "pcs" -> "pc"
            "aliases": [],
            "description": f"Reason: {candidate['reason']}. Evidence: {evidence}"
        }
        glossary_data[bucket].append(entry)

    with candidate_yaml_path.open("w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump({"glossary": glossary_data}, f, default_flow_style=False, sort_keys=True)
