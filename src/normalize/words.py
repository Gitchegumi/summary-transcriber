from __future__ import annotations

import re

from src.config.manifest import SessionManifest, SpeakerConfig
from src.normalize.turns import _extract_segments


def normalize_words(
    manifest: SessionManifest,
    speaker: SpeakerConfig,
    raw_output: dict,
    turns: list[dict],
    backend: str,
    model: str,
) -> list[dict]:
    if "chunks" not in raw_output:
        # Standard non-chunked path
        segments = _extract_segments(raw_output)
        words: list[dict] = []

        for turn_index, turn in enumerate(turns):
            segment = segments[turn_index] if isinstance(segments, list) and turn_index < len(segments) else {}
            segment_words = _segment_words(segment)
            if not segment_words:
                segment_words = _fallback_words(turn)
            for word_index, word in enumerate(segment_words, start=1):
                text = str(word.get("word") or word.get("text") or "").strip()
                confidence = word.get("confidence") or word.get("probability") or word.get("score")
                confidence = float(confidence) if confidence is not None else None
                words.append(
                    {
                        "session_id": manifest.session.id,
                        "turn_id": turn["turn_id"],
                        "word_id": f"{turn['turn_id']}-word-{word_index:04d}",
                        "speaker_id": speaker.speaker_id,
                        "word": text,
                        "start_seconds": float(word.get("start") or turn["start_seconds"]),
                        "end_seconds": float(word.get("end") or turn["end_seconds"]),
                        "confidence": confidence,
                        "is_low_confidence": confidence is not None and confidence < 0.6,
                        "backend": backend,
                        "model": model,
                    }
                )
        return words

    # Chunked path
    temp_to_final = {}
    for turn in turns:
        if "_temp_turn_id" in turn:
            temp_to_final[turn["_temp_turn_id"]] = turn["turn_id"]

    words = []
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

        for segment_index, segment in enumerate(segments, start=1):
            temp_turn_id = f"{manifest.session.id}-{speaker.speaker_id}-{chunk_id}-turn-{segment_index:05d}"
            if temp_turn_id not in temp_to_final:
                # Discarded during deduplication
                continue

            final_turn_id = temp_to_final[temp_turn_id]
            segment_words = _segment_words(segment)
            if not segment_words:
                # Find turn and use fallback words
                matched_turn = next((t for t in turns if t["turn_id"] == final_turn_id), None)
                if matched_turn:
                    segment_words = _fallback_words(matched_turn)
                    for word_index, w in enumerate(segment_words, start=1):
                        words.append(
                            {
                                "session_id": manifest.session.id,
                                "turn_id": final_turn_id,
                                "word_id": f"{final_turn_id}-word-{word_index:04d}",
                                "speaker_id": speaker.speaker_id,
                                "word": w["word"],
                                "start_seconds": w["start"],
                                "end_seconds": w["end"],
                                "confidence": None,
                                "is_low_confidence": False,
                                "backend": backend,
                                "model": model,
                            }
                        )
                continue

            for word_index, w in enumerate(segment_words, start=1):
                text = str(w.get("word") or w.get("text") or "").strip()
                confidence = w.get("confidence") or w.get("probability") or w.get("score")
                confidence = float(confidence) if confidence is not None else None
                words.append(
                    {
                        "session_id": manifest.session.id,
                        "turn_id": final_turn_id,
                        "word_id": f"{final_turn_id}-word-{word_index:04d}",
                        "speaker_id": speaker.speaker_id,
                        "word": text,
                        "start_seconds": float(w.get("start") or 0) + chunk_start,
                        "end_seconds": float(w.get("end") or 0) + chunk_start,
                        "confidence": confidence,
                        "is_low_confidence": confidence is not None and confidence < 0.6,
                        "backend": backend,
                        "model": model,
                    }
                )
    return words


def _segment_words(segment) -> list[dict]:
    if hasattr(segment, "_asdict"):
        segment = segment._asdict()
    if not isinstance(segment, dict):
        return []
    words = segment.get("words") or segment.get("word_timestamps") or []
    # If the segment itself came from a NeMo Hypothesis and has a timestep dict:
    if not words and "timestep" in segment and isinstance(segment["timestep"], dict):
        words = segment["timestep"].get("word") or []
    return words if isinstance(words, list) else []


def _fallback_words(turn: dict) -> list[dict]:
    tokens = re.findall(r"\S+", turn["text_raw"])
    if not tokens:
        return []
    duration = max(0.0, turn["end_seconds"] - turn["start_seconds"])
    step = duration / len(tokens) if duration else 0
    return [
        {
            "word": token,
            "start": turn["start_seconds"] + (index * step),
            "end": turn["start_seconds"] + ((index + 1) * step),
        }
        for index, token in enumerate(tokens)
    ]
