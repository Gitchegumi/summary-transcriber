from __future__ import annotations

import json
import shutil
import re
from pathlib import Path
from datetime import datetime, timezone
from src.config.manifest import SessionManifest


def export_agents(
    output_dir: Path,
    manifest: SessionManifest,
    turns: list[dict],
    chunks: list[dict],
    corrections: list[dict],
    completeness_report: dict,
    audio_reports: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean stale chunk files
    chunk_dir = output_dir / "chunks"
    if chunk_dir.exists():
        shutil.rmtree(chunk_dir)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    # 1. Write agents/turns.jsonl
    turns_jsonl_path = output_dir / "turns.jsonl"
    with turns_jsonl_path.open("w", encoding="utf-8", newline="\n") as f:
        for payload in _agent_payloads(manifest, turns):
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    # 2. Write agents/chunks/chunk_001.jsonl, etc.
    turns_by_id = {turn["turn_id"]: turn for turn in turns}
    for chunk in chunks:
        chunk_file_name = f"{chunk['chunk_id']}.jsonl"
        chunk_turns = [
            turns_by_id[turn_id]
            for turn_id in chunk["turn_ids"]
            if turn_id in turns_by_id
        ]
        with (chunk_dir / chunk_file_name).open("w", encoding="utf-8", newline="\n") as f:
            for payload in _agent_payloads(manifest, chunk_turns):
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    # 3. Write agents/chunks.jsonl (index)
    chunks_jsonl_path = output_dir / "chunks.jsonl"
    with chunks_jsonl_path.open("w", encoding="utf-8", newline="\n") as f:
        for chunk in chunks:
            f.write(json.dumps({
                "chunk_id": chunk["chunk_id"],
                "session_id": chunk["session_id"],
                "start_seconds": chunk["start_seconds"],
                "end_seconds": chunk["end_seconds"],
                "turn_count": chunk["turn_count"],
                "turn_ids": chunk["turn_ids"],
                "first_turn_id": chunk["first_turn_id"],
                "last_turn_id": chunk["last_turn_id"],
                "speaker_ids": chunk["speaker_ids"],
                "chunk_file": f"agents/chunks/{chunk['chunk_id']}.jsonl",
                "token_estimate": chunk["token_estimate"]
            }, ensure_ascii=False) + "\n")

    # 4. Write agents/session.json
    session_path = output_dir / "session.json"
    
    total_audio_duration = sum(info.get("duration_seconds", 0) for info in audio_reports.values())
    speaker_files = [{"speaker_id": s.speaker_id, "file": s.file, "duration_seconds": audio_reports.get(s.speaker_id, {}).get("duration_seconds", 0)} for s in manifest.speakers]
    
    # Get failed chunks and missing coverage ranges from completeness report
    failed_chunks = []
    missing_ranges = []
    total_missing_seconds = 0.0
    comp_summary = completeness_report.get("completeness_summary", {})
    comp_status = comp_summary.get("status", "complete")
    total_missing_seconds = comp_summary.get("total_missing_seconds", 0.0)
    
    for spk_id, spk_report in completeness_report.get("speakers", {}).items():
        if spk_report.get("chunks_failed", 0) > 0:
            failed_chunks.append({
                "speaker_id": spk_id,
                "failed_count": spk_report["chunks_failed"]
            })
        if spk_report.get("missing_ranges"):
            for r in spk_report["missing_ranges"]:
                missing_ranges.append({
                    "speaker_id": spk_id,
                    "start_seconds": r[0],
                    "end_seconds": r[1]
                })

    session_payload = {
        "session_id": manifest.session.id,
        "campaign": manifest.session.campaign,
        "session_number": manifest.session.session_number,
        "session_date": manifest.session.session_date,
        "source": manifest.session.source,
        "transcription": {
            "backend": manifest.transcription.backend,
            "model": manifest.transcription.model
        },
        "audio": {
            "total_audio_duration_seconds": total_audio_duration,
            "speaker_files": speaker_files
        },
        "outputs": {
            "turns_jsonl": "agents/turns.jsonl",
            "chunks_index_jsonl": "agents/chunks.jsonl",
            "chunks_dir": "agents/chunks",
            "entities_jsonl": "agents/transcript_entities.jsonl",
            "markdown_transcript": "transcript.md"
        },
        "chunking": {
            "transcript_chunk_minutes": 30
        },
        "completeness": {
            "status": comp_status,
            "total_missing_seconds": total_missing_seconds,
            "failed_chunks": failed_chunks,
            "missing_ranges": missing_ranges,
            "warnings": comp_summary.get("warnings", [])
        },
        "generated_at": completeness_report.get("generated_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    }
    
    with session_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(session_payload, f, ensure_ascii=False, indent=2)

    # 5. Write agents/transcript_entities.jsonl
    entities_path = output_dir / "transcript_entities.jsonl"
    
    entity_mentions = {}
    for turn in turns:
        text = turn["text_cleaned"]
        for category, entries in manifest.glossary.entries.items():
            entity_type = category.rstrip("s")
            if entity_type == "rules":
                entity_type = "rules_term"
            elif entity_type == "custom":
                entity_type = "custom_term"
                
            for entry in entries:
                if not entry.canonical:
                    continue
                terms_to_check = [entry.canonical] + entry.aliases
                matched = False
                for term in terms_to_check:
                    if not term:
                        continue
                    if re.search(rf"\b{re.escape(term)}\b", text, flags=re.IGNORECASE):
                        matched = True
                        break
                if matched:
                    entity_mentions.setdefault(entry.canonical, []).append(turn)
                    
    with entities_path.open("w", encoding="utf-8", newline="\n") as f:
        entity_idx = 1
        for category, entries in manifest.glossary.entries.items():
            entity_type = category.rstrip("s")
            if entity_type == "rules":
                entity_type = "rules_term"
            elif entity_type == "custom":
                entity_type = "custom_term"
                
            for entry in entries:
                mentions = entity_mentions.get(entry.canonical, [])
                if not mentions:
                    continue
                
                mentions.sort(key=lambda t: t["start_seconds"])
                first_turn = mentions[0]
                
                related = []
                for r in entry.related_to:
                    if isinstance(r, dict):
                        related.append(r)
                    elif isinstance(r, str):
                        related.append({"name": r})
                
                entity_payload = {
                    "entity_id": f"entity-{entity_type}-{entity_idx:04d}",
                    "canonical name": entry.canonical,
                    "canonical_name": entry.canonical,
                    "type/category": entity_type,
                    "type": entity_type,
                    "category": entity_type,
                    "aliases": entry.aliases,
                    "description": entry.description,
                    "related entities": related,
                    "related_entities": related,
                    "first_seen_turn_id": first_turn["turn_id"],
                    "first_seen_timestamp": first_turn["start_seconds"],
                    "evidence turn IDs where available": [t["turn_id"] for t in mentions],
                    "evidence_turn_ids": [t["turn_id"] for t in mentions],
                }
                f.write(json.dumps(entity_payload, ensure_ascii=False) + "\n")
                entity_idx += 1


def _agent_payloads(
    manifest: SessionManifest,
    turns: list[dict],
    max_gap_seconds: float = 10.0,
) -> list[dict]:
    merged: list[tuple[dict, float]] = []
    for turn in turns:
        payload = _turn_payload(manifest, turn)
        if payload is None:
            continue

        start_seconds = float(turn["start_seconds"])
        end_seconds = float(turn["end_seconds"])
        if (
            merged
            and merged[-1][0]["speaker"] == payload["speaker"]
            and start_seconds - merged[-1][1] <= max_gap_seconds
        ):
            previous, previous_end = merged[-1]
            merged_end = max(previous_end, end_seconds)
            previous["end_time"] = _format_timestamp(merged_end)
            previous["text"] = f'{previous["text"]} {payload["text"]}'
            merged[-1] = (previous, merged_end)
        else:
            merged.append((payload, end_seconds))

    return [payload for payload, _ in merged]

def _turn_payload(manifest: SessionManifest, turn: dict) -> dict | None:
    text = turn["text_cleaned"].strip()
    if not text:
        return None

    character_name = (turn.get("character_name") or "").strip()
    role = _speaker_role(manifest, turn["speaker_id"])
    speaker = character_name or (
        "DM" if role and role.lower() == "dm" else turn["speaker_name"]
    )

    return {
        "start_time": _format_timestamp(turn["start_seconds"]),
        "end_time": _format_timestamp(turn["end_seconds"]),
        "speaker": speaker,
        "text": text,
    }


def _format_timestamp(seconds: float) -> str:
    total_milliseconds = max(0, round(float(seconds) * 1000))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"

def _speaker_role(manifest: SessionManifest, speaker_id: str) -> str | None:
    return next((speaker.role for speaker in manifest.speakers if speaker.speaker_id == speaker_id), None)
