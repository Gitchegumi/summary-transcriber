from __future__ import annotations

import csv
from pathlib import Path
import yaml


GLOSSARY_CANDIDATE_FIELDS = [
    "session_id",
    "candidate_id",
    "term",
    "normalized_term",
    "candidate_type",
    "suggested_glossary_bucket",
    "score",
    "positive_signals",
    "negative_signals",
    "reason",
    "occurrence_count",
    "speaker_count",
    "example_turn_ids",
    "example_text",
    "speaker_id",
    "turn_id",
    "start_seconds",
    "end_seconds",
    "confidence",
]

NOISE_REPORT_FIELDS = [
    "term",
    "reason_filtered",
    "occurrence_count",
    "example_text",
]


def export_noise_report(output_dir: Path, noise_candidates: list[dict], enabled: bool) -> None:
    if not enabled:
        return
    draft_dir = output_dir / "draft"
    draft_dir.mkdir(parents=True, exist_ok=True)
    noise_path = draft_dir / "noise_report.csv"
    with noise_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=NOISE_REPORT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(noise_candidates)


def export_draft_outputs(
    output_dir: Path,
    candidates: list[dict],
    turns: list[dict],
    words: list[dict],
) -> None:
    draft_dir = output_dir / "draft"
    draft_dir.mkdir(parents=True, exist_ok=True)

    # Remove the redundant detailed YAML produced by older versions. The CSV
    # remains the review artifact; candidate_glossary.yaml is import-ready.
    legacy_candidates = draft_dir / "glossary_candidates.yaml"
    if legacy_candidates.exists():
        legacy_candidates.unlink()

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
        
        evidence = f"Turn {candidate.get('turn_id', '')}: \"{candidate.get('example_text', '')}\""
        reason_parts = [f"Score: {candidate.get('score', 0)}", f"Reason: {candidate.get('reason', '')}"]
        if candidate.get("positive_signals"):
            reason_parts.append(f"Positive: {candidate['positive_signals']}")
        if candidate.get("negative_signals"):
            reason_parts.append(f"Negative: {candidate['negative_signals']}")
        reason_str = " | ".join(reason_parts)
        
        entry = {
            "canonical": candidate["term"],
            "type": bucket.rstrip("s"),
            "aliases": [],
            "description": f"{reason_str}. Evidence: {evidence}"
        }
        glossary_data[bucket].append(entry)

    with candidate_yaml_path.open("w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump({"glossary": glossary_data}, f, default_flow_style=False, sort_keys=True)
