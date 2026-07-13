from __future__ import annotations

from datetime import datetime, timezone

from src.config.manifest import SessionManifest


def _merge_ranges(intervals: list[tuple[float, float]]) -> list[list[float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def _missing_ranges(covered: list[list[float]], duration: float) -> list[list[float]]:
    missing = []
    cursor = 0.0
    for start, end in covered:
        if start - cursor > 0.01:
            missing.append([cursor, start])
        cursor = max(cursor, end)
    if duration - cursor > 0.01:
        missing.append([cursor, duration])
    return missing


def _subtract_ranges(
    covered: list[list[float]], excluded: list[tuple[float, float]]
) -> list[list[float]]:
    remaining: list[list[float]] = []
    excluded_ranges = _merge_ranges(excluded)

    for covered_start, covered_end in _merge_ranges(
        [(start, end) for start, end in covered]
    ):
        cursor = covered_start
        for excluded_start, excluded_end in excluded_ranges:
            if excluded_end <= cursor:
                continue
            if excluded_start >= covered_end:
                break
            if excluded_start > cursor:
                remaining.append([cursor, min(excluded_start, covered_end)])
            cursor = max(cursor, excluded_end)
            if cursor >= covered_end:
                break
        if cursor < covered_end:
            remaining.append([cursor, covered_end])

    return remaining


def compute_completeness_report(
    manifest: SessionManifest,
    turns: list[dict],
    audio_reports: dict,
    speaker_raw_outputs: dict,
) -> dict:
    speaker_reports = {}
    total_missing_seconds = 0.0
    total_silence_seconds = 0.0

    for speaker in manifest.speakers:
        speaker_id = speaker.speaker_id
        duration = float(
            audio_reports.get(speaker_id, {}).get("duration_seconds") or 0.0
        )
        speaker_turns = [turn for turn in turns if turn["speaker_id"] == speaker_id]
        speech_ranges = _merge_ranges([
            (
                float(turn.get("start_seconds") or 0.0),
                float(turn.get("end_seconds") or turn.get("start_seconds") or 0.0),
            )
            for turn in speaker_turns
        ])
        speech_seconds = sum(end - start for start, end in speech_ranges)
        silence_seconds = max(0.0, duration - speech_seconds)
        total_silence_seconds += silence_seconds

        raw_output = (speaker_raw_outputs or {}).get(speaker_id, {})
        chunks_expected = 1
        chunks_success = 0
        chunks_empty = 0
        chunks_failed = 0
        processed_ranges: list[list[float]] = []

        if raw_output and "chunks" in raw_output:
            chunks = raw_output.get("chunks") or []
            failures = raw_output.get("failed_chunks") or []
            failed_ids = {
                failure.get("chunk_id") for failure in failures if failure.get("chunk_id")
            }
            chunk_ids = {chunk.get("chunk_id") for chunk in chunks}
            chunks_failed = len(failed_ids)
            chunks_expected = len(chunk_ids | failed_ids)

            for chunk in chunks:
                chunk_id = chunk.get("chunk_id")
                if chunk_id in failed_ids:
                    continue
                start = float(chunk.get("chunk_start_seconds") or 0.0)
                end = float(chunk.get("chunk_end_seconds") or start)
                if end > start:
                    retry_failed_ranges = [
                        (
                            float(failure.get("start_seconds") or 0.0),
                            float(failure.get("end_seconds") or 0.0),
                        )
                        for failure in failures
                        if str(failure.get("chunk_id") or "").startswith(
                            f"{chunk_id}_retry_"
                        )
                    ]
                    processed_ranges.extend(
                        _subtract_ranges([[start, end]], retry_failed_ranges)
                    )
                chunks_success += 1

                raw_result = chunk.get("raw_result") or {}
                timestamp = raw_result.get("timestamp") or {}
                segments = (
                    raw_result.get("segments")
                    or raw_result.get("segment")
                    or raw_result.get("timestamps")
                    or (timestamp.get("segment") if isinstance(timestamp, dict) else [])
                    or []
                )
                if not str(raw_result.get("text") or "").strip() and not segments:
                    chunks_empty += 1

            processed_ranges = _merge_ranges(processed_ranges)
        elif raw_output and "result" in raw_output:
            # A non-chunked provider successfully processed the entire file;
            # gaps between returned turns are silence, not missing coverage.
            chunks_success = 1
            chunks_empty = 0 if speaker_turns else 1
            processed_ranges = [[0.0, duration]] if duration > 0 else []
        else:
            chunks_failed = 1

        missing_ranges = _missing_ranges(processed_ranges, duration)
        missing_seconds = sum(end - start for start, end in missing_ranges)
        total_missing_seconds += missing_seconds

        speaker_reports[speaker_id] = {
            "speaker_id": speaker_id,
            "display_name": speaker.display_name,
            "source_file": speaker.file,
            "source_duration_seconds": duration,
            "chunks_expected": chunks_expected,
            "chunks_successfully_transcribed": chunks_success,
            "chunks_empty": chunks_empty,
            "chunks_failed": chunks_failed,
            "covered_ranges": processed_ranges,
            "missing_ranges": missing_ranges,
            "total_missing_seconds": round(missing_seconds, 3),
            "speech_ranges": speech_ranges,
            "speech_seconds": round(speech_seconds, 3),
            "estimated_silence_seconds": round(silence_seconds, 3),
        }

    max_missing_warn = getattr(
        manifest.completeness, "max_missing_seconds_warn", 500.0
    )
    warnings = []
    if total_missing_seconds > max_missing_warn:
        status = "incomplete"
        warnings.append(
            f"Transcription is incomplete. Missing {total_missing_seconds:.1f} seconds "
            f"of processing coverage, which exceeds the warning threshold of "
            f"{max_missing_warn} seconds."
        )
    elif total_missing_seconds > 0:
        status = "complete_with_warnings"
        warnings.append(
            f"Transcription complete but with minor missing processing coverage: "
            f"{total_missing_seconds:.1f} seconds."
        )
    else:
        status = "complete"

    return {
        "session_id": manifest.session.id,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "completeness_summary": {
            "status": status,
            "total_missing_seconds": round(total_missing_seconds, 3),
            "estimated_silence_seconds": round(total_silence_seconds, 3),
            "warnings": warnings,
        },
        "speakers": speaker_reports,
    }