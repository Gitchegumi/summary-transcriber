from __future__ import annotations

import re

from src.config.manifest import SessionManifest, SpeakerConfig


DEDUPLICATION_DECISIONS = []
DISCARDED_TURNS = []


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
        # If segments is empty, but we have text, fallback to chunk boundaries
        if not segments and raw_result.get("text", "").strip():
            segments = [{
                "text": raw_result["text"],
                "start": 0.0,
                "end": chunk_end - chunk_start,
                "words": []
            }]
            
        for index, segment in enumerate(segments, start=1):
            text = str(segment.get("text") or "").strip()
            relative_start = float(
                segment.get("start") or segment.get("start_seconds") or 0
            )
            relative_end = float(
                segment.get("end") or segment.get("end_seconds") or 0
            )
            start = chunk_start + relative_start
            # NeMo sometimes returns chunk text without segment timestamps. In
            # that case the most accurate available interval is the chunk itself.
            end = (
                chunk_start + relative_end
                if relative_end > relative_start
                else chunk_end
            )
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
    
    if manifest.deduplication.enabled:
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
            
            if not turns_curr or not turns_next:
                continue
                
            # Interval overlap pre-filtering:
            # Current chunk turns only overlap with next chunk turns in the boundary region.
            # We can find the overlap time interval: [next_chunk_start, curr_chunk_end]
            overlap_min_start = turns_next[0]["_chunk_start"]
            overlap_max_end = turns_curr[0]["_chunk_end"]
            
            candidate_curr = [t for t in turns_curr if t["end_seconds"] > overlap_min_start]
            candidate_next = [t for t in turns_next if t["start_seconds"] < overlap_max_end]
            
            for t_curr in candidate_curr:
                if t_curr["turn_id"] in discarded_ids:
                    continue
                for t_next in candidate_next:
                    if t_next["turn_id"] in discarded_ids:
                        continue
                        
                    # Pre-filter non-overlapping intervals
                    if t_curr["start_seconds"] >= t_next["end_seconds"] or t_curr["end_seconds"] <= t_next["start_seconds"]:
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
                            
                        # If audit_only is enabled, we record the decision but do not actually discard the turn.
                        if not manifest.deduplication.audit_only:
                            discarded_ids.add(discarded["turn_id"])
                        DISCARDED_TURNS.append(discarded)
                        
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
                        
                        if not prefer_curr and not manifest.deduplication.audit_only:
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
    if not isinstance(result, dict):
        if isinstance(result, list):
            return [_coerce_segment(item) for item in result]
        if isinstance(result, str):
            return [{"start": 0.0, "end": 0.0, "text": result, "words": []}]
        return []

    # 1. Check for standard Whisper-style segments
    for key in ("segments", "segment", "timestamp", "timestamps"):
        value = result.get(key)
        if isinstance(value, list):
            return [_coerce_segment(item) for item in value]

    # 2. Check for NeMo style hypothesis with timestep
    # NeMo 2.7 uses `timestamp`; older hypotheses used `timestep`.
    timestep = result.get("timestamp") or result.get("timestep")
    if isinstance(timestep, dict):
        segments = timestep.get("segment")
        words = timestep.get("word") or []
        
        # If segment timesteps are available:
        if isinstance(segments, list) and len(segments) > 0:
            extracted = []
            for seg in segments:
                seg_dict = _coerce_segment(seg)
                if not seg_dict.get("text"):
                    seg_dict["text"] = seg_dict.get("segment") or ""
                start = _get_timestamp_key(seg_dict, ["start", "start_time", "start_seconds", "start_offset"])
                end = _get_timestamp_key(seg_dict, ["end", "end_time", "end_seconds", "end_offset"])
                seg_dict["start"] = start if start is not None else 0.0
                seg_dict["end"] = end if end is not None else 0.0
                
                # Filter words for this segment if not already present
                if "words" not in seg_dict or not seg_dict["words"]:
                    seg_words = []
                    for w in words:
                        w_dict = _coerce_segment(w)
                        w_start = _get_timestamp_key(w_dict, ["start", "start_time", "start_seconds", "start_offset"])
                        w_end = _get_timestamp_key(w_dict, ["end", "end_time", "end_seconds", "end_offset"])
                        if w_start is not None and w_end is not None:
                            if w_start >= seg_dict["start"] and w_end <= seg_dict["end"]:
                                seg_words.append(w_dict)
                    seg_dict["words"] = seg_words
                extracted.append(seg_dict)
            return extracted
        
        # If no segment timesteps but word timesteps are available:
        if isinstance(words, list) and len(words) > 0:
            word_dicts = []
            min_start = None
            max_end = None
            for w in words:
                w_dict = _coerce_segment(w)
                w_start = _get_timestamp_key(w_dict, ["start", "start_time", "start_seconds", "start_offset"])
                w_end = _get_timestamp_key(w_dict, ["end", "end_time", "end_seconds", "end_offset"])
                w_word = w_dict.get("word") or w_dict.get("text") or w_dict.get("char") or ""
                
                w_dict["word"] = w_word
                w_dict["start"] = w_start if w_start is not None else 0.0
                w_dict["end"] = w_end if w_end is not None else 0.0
                word_dicts.append(w_dict)
                
                if w_start is not None:
                    if min_start is None or w_start < min_start:
                        min_start = w_start
                if w_end is not None:
                    if max_end is None or w_end > max_end:
                        max_end = w_end
            
            return [{
                "text": result.get("text") or "",
                "start": min_start if min_start is not None else 0.0,
                "end": max_end if max_end is not None else 0.0,
                "words": word_dicts
            }]

    # 3. If there is a top-level text but no segments
    if "text" in result:
        words = result.get("words") or result.get("word_timestamps") or []
        start = _get_timestamp_key(result, ["start", "start_time", "start_seconds", "start_offset"]) or 0.0
        end = _get_timestamp_key(result, ["end", "end_time", "end_seconds", "end_offset"]) or 0.0
        return [{
            "text": result["text"],
            "start": start,
            "end": end,
            "words": words
        }]

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


def _get_timestamp_key(d: dict, keys: list[str]) -> float | None:
    for k in keys:
        if k in d and d[k] is not None and d[k] != "":
            try:
                return float(d[k])
            except ValueError:
                pass
    return None
