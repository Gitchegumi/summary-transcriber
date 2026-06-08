from __future__ import annotations

from collections import Counter, defaultdict

from src.config.manifest import SessionManifest


def build_quality_report(
    manifest: SessionManifest,
    turns: list[dict],
    words: list[dict],
    audio_reports: dict,
    corrections: list[dict],
    generated_at: str,
) -> dict:
    by_speaker = defaultdict(list)
    for turn in turns:
        by_speaker[turn["speaker_id"]].append(turn)

    flags = []
    for speaker in manifest.speakers:
        speaker_turns = by_speaker.get(speaker.speaker_id, [])
        word_count = sum(turn["word_count"] for turn in speaker_turns)
        if word_count < 5:
            flags.append(_flag("near_empty_speaker_track", speaker.speaker_id, "Speaker has fewer than 5 transcribed words."))

        audio_duration = audio_reports.get(speaker.speaker_id, {}).get("duration_seconds", 0)
        transcript_end = max((turn["end_seconds"] for turn in speaker_turns), default=0)
        if audio_duration and transcript_end and abs(audio_duration - transcript_end) > 120:
            flags.append(_flag("source_audio_duration_mismatch", speaker.speaker_id, "Transcript end differs from source duration by more than 120 seconds."))

        previous_end = None
        for turn in speaker_turns:
            if turn["duration_seconds"] > 90:
                flags.append(_flag("very_long_segment", speaker.speaker_id, f"Turn {turn['turn_id']} is longer than 90 seconds."))
            if previous_end is not None:
                gap = turn["start_seconds"] - previous_end
                if gap > 300:
                    flags.append(_flag("unusual_silence_gap", speaker.speaker_id, f"Gap before {turn['turn_id']} is {gap:.1f} seconds."))
                if gap < -1:
                    flags.append(_flag("impossible_or_overlapping_timestamps", speaker.speaker_id, f"Turn {turn['turn_id']} overlaps previous turn."))
            if turn["end_seconds"] < turn["start_seconds"]:
                flags.append(_flag("impossible_or_overlapping_timestamps", speaker.speaker_id, f"Turn {turn['turn_id']} ends before it starts."))
            previous_end = turn["end_seconds"]

    repeated = Counter(turn["text_raw"].strip().lower() for turn in turns if turn["text_raw"].strip())
    for text, count in repeated.items():
        if count >= 5:
            flags.append(_flag("repeated_phrase_loop", None, f"Phrase appears {count} times: {text[:80]}"))

    low_confidence = [word for word in words if word.get("is_low_confidence")]
    if low_confidence:
        flags.append(_flag("low_confidence_words", None, f"{len(low_confidence)} words are below confidence threshold."))

    if corrections:
        flags.append(_flag("likely_glossary_mismatches", None, f"{len(corrections)} rule-based corrections were applied."))

    fantasy_candidates = [
        word for word in words
        if word.get("confidence") is not None
        and word["confidence"] < 0.7
        and any(char.isupper() for char in word.get("word", "")[1:])
    ]
    if fantasy_candidates:
        flags.append(_flag("possible_fantasy_name_errors", None, f"{len(fantasy_candidates)} low-confidence mixed-case terms found."))

    return {
        "session_id": manifest.session.id,
        "generated_at": generated_at,
        "speaker_count": len(manifest.speakers),
        "turn_count": len(turns),
        "word_count": len(words),
        "flags": flags,
        "audio": audio_reports,
    }


def _flag(flag_type: str, speaker_id: str | None, message: str) -> dict:
    return {"type": flag_type, "speaker_id": speaker_id, "message": message}
