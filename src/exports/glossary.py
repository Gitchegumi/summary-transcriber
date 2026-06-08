from __future__ import annotations

import csv
import json
from pathlib import Path


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
