from __future__ import annotations

from src.config.manifest import SessionManifest


def build_chunks(turns: list[dict], manifest: SessionManifest, chunk_minutes: float = 30.0) -> list[dict]:
    chunk_seconds = chunk_minutes * 60.0
    if not turns:
        return []
    
    # Group turns by chunk index
    turns_by_chunk_idx = {}
    for turn in turns:
        idx = int(turn["start_seconds"] // chunk_seconds)
        if idx not in turns_by_chunk_idx:
            turns_by_chunk_idx[idx] = []
        turns_by_chunk_idx[idx].append(turn)
        
    # Create chunks for all indices that have turns
    sorted_idxs = sorted(turns_by_chunk_idx.keys())
    chunks = []
    for index, idx in enumerate(sorted_idxs, start=1):
        chunk_turns = turns_by_chunk_idx[idx]
        chunk_id = f"chunk_{index:03d}"
        
        # Calculate word count for token estimation
        word_count = sum(len(turn.get("text_raw", "").split()) for turn in chunk_turns)
        
        chunks.append({
            "session_id": manifest.session.id,
            "chunk_id": chunk_id,
            "chunk_index": index,
            "chunk_type": "time",
            "start_seconds": idx * chunk_seconds,
            "end_seconds": (idx + 1) * chunk_seconds,
            "turn_count": len(chunk_turns),
            "speaker_ids": sorted({turn["speaker_id"] for turn in chunk_turns}),
            "turn_ids": [turn["turn_id"] for turn in chunk_turns],
            "first_turn_id": chunk_turns[0]["turn_id"],
            "last_turn_id": chunk_turns[-1]["turn_id"],
            "chunk_file": f"agents/chunks/{chunk_id}.jsonl",
            "token_estimate": int(word_count * 1.3),
        })
    return chunks
