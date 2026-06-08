from __future__ import annotations

import json
from pathlib import Path

from src.config.manifest import SessionManifest


def export_agents(
    output_dir: Path,
    manifest: SessionManifest,
    turns: list[dict],
    chunks: list[dict],
    corrections: list[dict],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    corrections_by_turn = {}
    for correction in corrections:
        corrections_by_turn.setdefault(correction["turn_id"], []).append(correction)

    jsonl_path = output_dir / "hermes_turns.jsonl"
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as f:
        for turn in turns:
            f.write(json.dumps(_turn_payload(manifest, turn, corrections_by_turn), ensure_ascii=False) + "\n")

    chunk_dir = output_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    for chunk in chunks:
        chunk_turns = [turn for turn in turns if turn["turn_id"] in chunk["turn_ids"]]
        with (chunk_dir / f"{chunk['chunk_id']}.json").open("w", encoding="utf-8", newline="\n") as f:
            json.dump(
                {
                    "session": _session(manifest),
                    "chunk": chunk,
                    "turns": [_turn_payload(manifest, turn, corrections_by_turn) for turn in chunk_turns],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    _write_summary_input(output_dir / "session_summary_input.md", manifest, chunks)


def _turn_payload(manifest: SessionManifest, turn: dict, corrections_by_turn: dict) -> dict:
    turn_corrections = corrections_by_turn.get(turn["turn_id"], [])
    return {
        "session": _session(manifest),
        "turn_id": turn["turn_id"],
        "speaker": {
            "speaker_id": turn["speaker_id"],
            "display_name": turn["speaker_name"],
            "role": _speaker_role(manifest, turn["speaker_id"]),
        },
        "character": {"name": turn["character_name"]},
        "timestamps": {
            "start_seconds": turn["start_seconds"],
            "end_seconds": turn["end_seconds"],
            "duration_seconds": turn["duration_seconds"],
        },
        "text_raw": turn["text_raw"],
        "text_cleaned": turn["text_cleaned"],
        "confidence": {"average": turn["confidence_avg"]},
        "low_confidence_terms": [],
        "glossary_correction_candidates": turn_corrections,
        "tags": [],
    }


def _session(manifest: SessionManifest) -> dict:
    return {
        "session_id": manifest.session.id,
        "campaign": manifest.session.campaign,
        "session_number": manifest.session.session_number,
        "session_date": manifest.session.session_date,
        "source": manifest.session.source,
    }


def _speaker_role(manifest: SessionManifest, speaker_id: str) -> str | None:
    return next((speaker.role for speaker in manifest.speakers if speaker.speaker_id == speaker_id), None)


def _write_summary_input(path: Path, manifest: SessionManifest, chunks: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# Hermes Session Summary Input\n\n")
        f.write("Use `hermes_turns.jsonl` as the canonical chronological transcript.\n")
        f.write("Use chunk JSON files for bounded summarization passes.\n\n")
        f.write("## Session\n\n")
        f.write(f"- Session ID: {manifest.session.id}\n")
        f.write(f"- Campaign: {manifest.session.campaign}\n")
        f.write(f"- Session number: {manifest.session.session_number}\n")
        f.write(f"- Session date: {manifest.session.session_date}\n\n")
        f.write("## Chunks\n\n")
        for chunk in chunks:
            f.write(
                f"- {chunk['chunk_id']}.json: {chunk['start_seconds']:.3f} to "
                f"{chunk['end_seconds']:.3f}, {chunk['turn_count']} turns\n"
            )
