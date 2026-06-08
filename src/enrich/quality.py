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
    speaker_raw_outputs: dict | None = None,
    oom_events: list | None = None,
    retried_chunks: list | None = None,
    failed_chunks: list | None = None,
    deduplication_decisions: list | None = None,
    nemo_chunking_used: bool = False,
    chunk_duration: float = 60.0,
    wall_clock_time: float = 0.0,
) -> dict:
    by_speaker = defaultdict(list)
    for turn in turns:
        by_speaker[turn["speaker_id"]].append(turn)

    flags = []
    
    # 1. Speaker level checks
    for speaker in manifest.speakers:
        speaker_turns = by_speaker.get(speaker.speaker_id, [])
        word_count = sum(turn["word_count"] for turn in speaker_turns)
        if word_count < 5:
            flags.append(_flag("near_empty_speaker_track", speaker.speaker_id, "Speaker has fewer than 5 transcribed words."))

        audio_duration = audio_reports.get(speaker.speaker_id, {}).get("duration_seconds", 0)
        transcript_end = max((turn["end_seconds"] for turn in speaker_turns), default=0)
        if audio_duration and transcript_end and abs(audio_duration - transcript_end) > 120:
            flags.append(_flag("source_audio_duration_mismatch", speaker.speaker_id, f"Transcript end ({transcript_end:.1f}s) differs from source duration ({audio_duration:.1f}s) by more than 120 seconds."))

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

    # 2. Repeated phrase loop checks
    repeated = Counter(turn["text_raw"].strip().lower() for turn in turns if turn["text_raw"].strip())
    for text, count in repeated.items():
        if count >= 5:
            flags.append(_flag("repeated_phrase_loop", None, f"Phrase appears {count} times: {text[:80]}"))

    # 3. Low confidence words
    low_confidence = [word for word in words if word.get("is_low_confidence")]
    if low_confidence:
        flags.append(_flag("low_confidence_words", None, f"{len(low_confidence)} words are below confidence threshold."))

    # 4. Corrections
    if corrections:
        flags.append(_flag("likely_glossary_mismatches", None, f"{len(corrections)} rule-based corrections were applied."))

    # 5. Possible fantasy name errors
    fantasy_candidates = [
        word for word in words
        if word.get("confidence") is not None
        and word["confidence"] < 0.7
        and any(char.isupper() for char in word.get("word", "")[1:])
    ]
    if fantasy_candidates:
        flags.append(_flag("possible_fantasy_name_errors", None, f"{len(fantasy_candidates)} low-confidence mixed-case terms found."))

    # 6. Overlap deduplication warnings
    if deduplication_decisions:
        flags.append(_flag("overlap_deduplication", None, f"Deduplicated {len(deduplication_decisions)} overlapping segments near boundaries."))

    # 7. Chunk level checks (Parakeet/Canary)
    chunk_counts = {}
    if speaker_raw_outputs and nemo_chunking_used:
        for speaker_id, raw_output in speaker_raw_outputs.items():
            if "chunks" in raw_output:
                chunk_counts[speaker_id] = len(raw_output["chunks"])
                for chunk in raw_output["chunks"]:
                    raw_res = chunk.get("raw_result", {})
                    text = raw_res.get("text", "").strip()
                    segments = raw_res.get("segments") or raw_res.get("segment") or raw_res.get("timestamps") or []
                    
                    # Chunks with no transcript output / unusually silent chunks
                    if not text and not segments:
                        flags.append(_flag("empty_chunk", speaker_id, f"Chunk {chunk['chunk_id']} returned no transcript output."))
                        flags.append(_flag("unusually_silent_chunk", speaker_id, f"Chunk {chunk['chunk_id']} was silent (no speech detected)."))
                    
                    # Chunks with suspicious repeated text
                    # Simple heuristic: if a single word or 2-word phrase makes up more than 40% of a reasonably long chunk text
                    words_in_chunk = text.lower().split()
                    if len(words_in_chunk) > 10:
                        word_counts = Counter(words_in_chunk)
                        most_common_word, count = word_counts.most_common(1)[0]
                        if count / len(words_in_chunk) > 0.4:
                            flags.append(_flag("suspicious_repeated_text", speaker_id, f"Chunk {chunk['chunk_id']} has suspicious repeated text (word '{most_common_word}' is {count/len(words_in_chunk)*100:.1f}% of text)."))

    # Compute duration metrics
    total_audio_duration = sum(info.get("duration_seconds", 0) for info in audio_reports.values())
    speed = (total_audio_duration / 60.0) / (wall_clock_time / 60.0) if wall_clock_time > 0 else 0.0

    return {
        "session_id": manifest.session.id,
        "generated_at": generated_at,
        "speaker_count": len(manifest.speakers),
        "turn_count": len(turns),
        "word_count": len(words),
        "total_audio_duration_seconds": total_audio_duration,
        "wall_clock_transcription_seconds": wall_clock_time,
        "average_processing_speed_ratio": speed,
        "nemo_chunking_used": nemo_chunking_used,
        "chunk_duration_seconds": chunk_duration if nemo_chunking_used else None,
        "chunk_counts": chunk_counts,
        "failed_chunks": failed_chunks or [],
        "retried_chunks": retried_chunks or [],
        "oom_events": oom_events or [],
        "deduplication_decisions": deduplication_decisions or [],
        "flags": flags,
        "audio": audio_reports,
    }


def _flag(flag_type: str, speaker_id: str | None, message: str) -> dict:
    return {"type": flag_type, "speaker_id": speaker_id, "message": message}
