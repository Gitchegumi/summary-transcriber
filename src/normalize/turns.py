from __future__ import annotations

import re

from src.config.manifest import SessionManifest, SpeakerConfig


DEDUPLICATION_DECISIONS = []


def normalize_turns(
    manifest: SessionManifest,
    speaker: SpeakerConfig,
    raw_output: dict,
    backend: str,
    model: str,
) -> list[dict]:
    if "chunks" not in raw_output:
        # Standard non-chunked path
        segments = _extract_segments(raw_output)
        turns: list[dict] = []
        for index, segment in enumerate(segments, start=1):
            text = str(segment.get("text") or "").strip()
            start = float(segment.get("start") or segment.get("start_seconds") or 0)
            end = float(segment.get("end") or segment.get("end_seconds") or start)
            words = _extract_words(segment)
            confidence = _avg_confidence(words, segment.get("confidence"))
            turn_id = f"{manifest.session.id}-{speaker.speaker_id}-turn-{index:05d}"
            turns.append(
                {
                    "session_id": manifest.session.id,
                    "turn_id": turn_id,
                    "speaker_id": speaker.speaker_id,
                    "speaker_name": speaker.display_name,
                    "character_name": speaker.character_name,
                    "start_seconds": start,
                    "end_seconds": end,
                    "duration_seconds": max(0.0, end - start),
                    "text_raw": text,
                    "text_cleaned": text,
                    "confidence_avg": confidence,
                    "word_count": len(text.split()),
                    "source_file": speaker.file,
                    "backend": backend,
                    "model": model,
                }
            )
        return turns

    # Chunked path
    all_turns = []
    
    for chunk in raw_output["chunks"]:
        chunk_id = chunk["chunk_id"]
        chunk_start = chunk["chunk_start_seconds"]
        chunk_end = chunk["chunk_end_seconds"]
        raw_result = chunk["raw_result"]
        
        segments = _extract_segments(raw_result)
        for index, segment in enumerate(segments, start=1):
            text = str(segment.get("text") or "").strip()
            start = float(segment.get("start") or segment.get("start_seconds") or 0) + chunk_start
            end = float(segment.get("end") or segment.get("end_seconds") or start) + chunk_start
            words = _extract_words(segment)
            confidence = _avg_confidence(words, segment.get("confidence"))
            
            temp_turn_id = f"{manifest.session.id}-{speaker.speaker_id}-{chunk_id}-turn-{index:05d}"
            
            all_turns.append(
                {
                    "session_id": manifest.session.id,
                    "turn_id": temp_turn_id,
                    "_temp_turn_id": temp_turn_id,
                    "speaker_id": speaker.speaker_id,
                    "speaker_name": speaker.display_name,
                    "character_name": speaker.character_name,
                    "start_seconds": start,
                    "end_seconds": end,
                    "duration_seconds": max(0.0, end - start),
                    "text_raw": text,
                    "text_cleaned": text,
                    "confidence_avg": confidence,
                    "word_count": len(text.split()),
                    "source_file": speaker.file,
                    "backend": backend,
                    "model": model,
                    "_chunk_id": chunk_id,
                    "_chunk_start": chunk_start,
                    "_chunk_end": chunk_end,
                }
            )
            
    # Overlap deduplication
    kept_turns = []
    discarded_ids = set()
    
    # We group turns by chunk in chronological order
    chunk_ids = [c["chunk_id"] for c in raw_output["chunks"]]
    turns_by_chunk = {cid: [] for cid in chunk_ids}
    for turn in all_turns:
        turns_by_chunk[turn["_chunk_id"]].append(turn)
        
    for i in range(len(chunk_ids) - 1):
        cid_curr = chunk_ids[i]
        cid_next = chunk_ids[i + 1]
        turns_curr = turns_by_chunk[cid_curr]
        turns_next = turns_by_chunk[cid_next]
        
        for t_curr in turns_curr:
            if t_curr["turn_id"] in discarded_ids:
                continue
            for t_next in turns_next:
                if t_next["turn_id"] in discarded_ids:
                    continue
                    
                # Calculate time overlap
                overlap_start = max(t_curr["start_seconds"], t_next["start_seconds"])
                overlap_end = min(t_curr["end_seconds"], t_next["end_seconds"])
                overlap_duration = overlap_end - overlap_start
                
                # Text Jaccard similarity
                words_curr = set(re.findall(r"\w+", t_curr["text_raw"].lower()))
                words_next = set(re.findall(r"\w+", t_next["text_raw"].lower()))
                union_len = len(words_curr.union(words_next))
                similarity = len(words_curr.intersection(words_next)) / union_len if union_len > 0 else 0.0
                
                # Deduplicate if overlapping in time and text is similar
                is_duplicate = (
                    (overlap_duration > 0 and similarity > 0.3)
                    or (overlap_duration > 0.5 and similarity > 0.1)
                )
                
                if is_duplicate:
                    conf_curr = t_curr["confidence_avg"]
                    conf_next = t_next["confidence_avg"]
                    
                    prefer_curr = True
                    reason_type = ""
                    
                    if conf_curr is not None and conf_next is not None:
                        if abs(conf_curr - conf_next) > 0.001:
                            prefer_curr = conf_curr > conf_next
                            reason_type = "confidence"
                        else:
                            dist_curr = t_curr["_chunk_end"] - t_curr["end_seconds"]
                            dist_next = t_next["start_seconds"] - t_next["_chunk_start"]
                            prefer_curr = dist_curr > dist_next
                            reason_type = "boundary_distance"
                    else:
                        dist_curr = t_curr["_chunk_end"] - t_curr["end_seconds"]
                        dist_next = t_next["start_seconds"] - t_next["_chunk_start"]
                        prefer_curr = dist_curr > dist_next
                        reason_type = "boundary_distance"
                        
                    if prefer_curr:
                        discarded = t_next
                        kept = t_curr
                    else:
                        discarded = t_curr
                        kept = t_next
                        
                    discarded_ids.add(discarded["turn_id"])
                    
                    # Record the decision
                    decision = {
                        "speaker_id": speaker.speaker_id,
                        "discarded_turn_id": discarded["turn_id"],
                        "kept_turn_id": kept["turn_id"],
                        "discarded_text": discarded["text_raw"],
                        "kept_text": kept["text_raw"],
                        "reason": f"Preferred {reason_type} ({'kept' if prefer_curr else 'discarded'} first)",
                    }
                    DEDUPLICATION_DECISIONS.append(decision)
                    
                    if not prefer_curr:
                        break
                        
    # Filter and re-sequence final turns
    final_turns = []
    final_index = 1
    for turn in all_turns:
        if turn["turn_id"] not in discarded_ids:
            old_id = turn["turn_id"]
            new_id = f"{manifest.session.id}-{speaker.speaker_id}-turn-{final_index:05d}"
            turn["turn_id"] = new_id
            turn["_temp_turn_id"] = old_id
            final_turns.append(turn)
            final_index += 1
            
    return final_turns


def _extract_segments(raw_output: dict) -> list[dict]:
    result = raw_output.get("result", raw_output)
    if isinstance(result, dict):
        for key in ("segments", "segment", "timestamp", "timestamps"):
            value = result.get(key)
            if isinstance(value, list):
                return [_coerce_segment(item) for item in value]
        if "text" in result:
            return [_coerce_segment(result)]
    if isinstance(result, list):
        return [_coerce_segment(item) for item in result]
    if isinstance(result, str):
        return [{"start": 0.0, "end": 0.0, "text": result, "words": []}]
    return []


def _coerce_segment(value) -> dict:
    if isinstance(value, dict):
        return value
    if hasattr(value, "_asdict"):
        return value._asdict()
    if hasattr(value, "__dict__"):
        return value.__dict__
    return {"text": str(value), "start": 0.0, "end": 0.0}


def _extract_words(segment: dict) -> list[dict]:
    words = segment.get("words") or segment.get("word_timestamps") or []
    return words if isinstance(words, list) else []


def _avg_confidence(words: list[dict], segment_confidence=None) -> float | None:
    values = []
    for word in words:
        if isinstance(word, dict):
            value = word.get("confidence") or word.get("probability") or word.get("score")
            if value is not None:
                values.append(float(value))
    if values:
        return sum(values) / len(values)
    return float(segment_confidence) if segment_confidence is not None else None
