from __future__ import annotations

from src.config.manifest import SessionManifest


def build_chunks(turns: list[dict], manifest: SessionManifest, chunk_minutes: float = 30.0) -> list[dict]:
    chunk_seconds = chunk_minutes * 60
    chunks = []
    current = []
    current_start = None

    for turn in turns:
        if current_start is None:
            current_start = turn["start_seconds"]
        if current and turn["start_seconds"] - current_start > chunk_seconds:
            chunks.append(_chunk_payload(manifest, len(chunks) + 1, current))
            current = []
            current_start = turn["start_seconds"]
        current.append(turn)

    if current:
        chunks.append(_chunk_payload(manifest, len(chunks) + 1, current))
    return chunks


def _chunk_payload(manifest: SessionManifest, index: int, turns: list[dict]) -> dict:
    return {
        "session_id": manifest.session.id,
        "chunk_id": f"chunk_{index:03d}",
        "chunk_index": index,
        "chunk_type": "time",
        "start_seconds": turns[0]["start_seconds"],
        "end_seconds": turns[-1]["end_seconds"],
        "turn_count": len(turns),
        "speaker_ids": sorted({turn["speaker_id"] for turn in turns}),
        "turn_ids": [turn["turn_id"] for turn in turns],
    }
