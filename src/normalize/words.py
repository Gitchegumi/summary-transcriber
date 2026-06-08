from __future__ import annotations

import re

from src.config.manifest import SessionManifest, SpeakerConfig


def normalize_words(
    manifest: SessionManifest,
    speaker: SpeakerConfig,
    raw_output: dict,
    turns: list[dict],
    backend: str,
    model: str,
) -> list[dict]:
    segments = raw_output.get("result", raw_output)
    if isinstance(segments, dict):
        segments = segments.get("segments", [])
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


def _segment_words(segment) -> list[dict]:
    if hasattr(segment, "_asdict"):
        segment = segment._asdict()
    if not isinstance(segment, dict):
        return []
    words = segment.get("words") or segment.get("word_timestamps") or []
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
