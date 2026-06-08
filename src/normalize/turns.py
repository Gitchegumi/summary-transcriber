from __future__ import annotations

from src.config.manifest import SessionManifest, SpeakerConfig


def normalize_turns(
    manifest: SessionManifest,
    speaker: SpeakerConfig,
    raw_output: dict,
    backend: str,
    model: str,
) -> list[dict]:
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
