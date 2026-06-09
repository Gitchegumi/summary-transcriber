from __future__ import annotations

import csv
import json
from pathlib import Path

from src.config.manifest import SessionManifest


TURN_FIELDS = [
    "session_id", "turn_id", "speaker_id", "speaker_name", "character_name",
    "start_seconds", "end_seconds", "duration_seconds", "text_raw", "text_cleaned",
    "confidence_avg", "word_count", "source_file", "backend", "model",
]
WORD_FIELDS = [
    "session_id", "turn_id", "word_id", "speaker_id", "word", "start_seconds",
    "end_seconds", "confidence", "is_low_confidence", "backend", "model",
]
CORRECTION_FIELDS = [
    "session_id", "turn_id", "correction_id", "original_text", "corrected_text",
    "correction_type", "reason", "confidence",
]
ENTITY_FIELDS = [
    "session_id", "turn_id", "entity_id", "entity_text", "entity_type",
    "start_seconds", "end_seconds", "source",
]
CHUNK_FIELDS = [
    "session_id", "chunk_id", "chunk_index", "chunk_type", "start_seconds",
    "end_seconds", "turn_count", "speaker_ids", "turn_ids",
]
SPEAKER_FIELDS = [
    "session_id", "speaker_id", "display_name", "role", "character_name", "file",
    "duration_seconds", "sample_rate", "channels", "format", "codec",
]


def export_nocodb(
    output_dir: Path,
    manifest: SessionManifest,
    turns: list[dict],
    words: list[dict],
    entities: list[dict],
    corrections: list[dict],
    chunks: list[dict],
    quality_report: dict,
    audio_reports: dict,
) -> None:
    _write_json(output_dir / "session.json", _session_payload(manifest))
    _write_csv(output_dir / "speakers.csv", SPEAKER_FIELDS, _speaker_rows(manifest, audio_reports))
    _write_csv(output_dir / "transcript_turns.csv", TURN_FIELDS, turns)
    _write_csv(output_dir / "transcript_words.csv", WORD_FIELDS, words)
    _write_csv(output_dir / "transcript_entities.csv", ENTITY_FIELDS, entities)
    _write_csv(output_dir / "transcript_corrections.csv", CORRECTION_FIELDS, corrections)
    _write_csv(output_dir / "transcript_chunks.csv", CHUNK_FIELDS, [_csv_chunk(chunk) for chunk in chunks])
    _write_json(output_dir / "transcript_quality_report.json", quality_report)


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _session_payload(manifest: SessionManifest) -> dict:
    return {
        "session_id": manifest.session.id,
        "campaign": manifest.session.campaign,
        "session_number": manifest.session.session_number,
        "session_date": manifest.session.session_date,
        "source": manifest.session.source,
        "backend": manifest.transcription.backend,
        "model": manifest.transcription.model,
        "device": manifest.transcription.device,
    }


def _speaker_rows(manifest: SessionManifest, audio_reports: dict) -> list[dict]:
    rows = []
    for speaker in manifest.speakers:
        audio = audio_reports.get(speaker.speaker_id, {})
        rows.append(
            {
                "session_id": manifest.session.id,
                "speaker_id": speaker.speaker_id,
                "display_name": speaker.display_name,
                "role": speaker.role,
                "character_name": speaker.character_name,
                "file": speaker.file,
                "duration_seconds": audio.get("duration_seconds"),
                "sample_rate": audio.get("sample_rate"),
                "channels": audio.get("channels"),
                "format": audio.get("format"),
                "codec": audio.get("codec"),
            }
        )
    return rows


def _csv_chunk(chunk: dict) -> dict:
    row = dict(chunk)
    row["speaker_ids"] = json.dumps(row["speaker_ids"], ensure_ascii=False)
    row["turn_ids"] = json.dumps(row["turn_ids"], ensure_ascii=False)
    return row
