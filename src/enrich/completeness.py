from __future__ import annotations

import json
from datetime import datetime, timezone
import re
from pathlib import Path
from src.config.manifest import SessionManifest


def compute_completeness_report(
    manifest: SessionManifest,
    turns: list[dict],
    audio_reports: dict,
    speaker_raw_outputs: dict | None = None
) -> dict:
    speaker_reports = {}
    total_missing_seconds = 0.0
    
    # We will compute coverage per speaker
    for speaker in manifest.speakers:
        spk_id = speaker.speaker_id
        audio_info = audio_reports.get(spk_id, {})
        duration = float(audio_info.get("duration_seconds") or 0.0)
        
        # Get all turns for this speaker
        spk_turns = [t for t in turns if t["speaker_id"] == spk_id]
        
        # Merge overlapping turn intervals to find covered ranges
        intervals = []
        for t in spk_turns:
            start = float(t.get("start_seconds") or 0.0)
            end = float(t.get("end_seconds") or start)
            if end > start:
                intervals.append((start, end))
                
        intervals.sort()
        merged_ranges = []
        for current in intervals:
            if not merged_ranges:
                merged_ranges.append(list(current))
            else:
                prev = merged_ranges[-1]
                if current[0] <= prev[1]:
                    prev[1] = max(prev[1], current[1])
                else:
                    merged_ranges.append(list(current))
                    
        # Find missing coverage ranges
        missing_ranges = []
        curr_time = 0.0
        for start, end in merged_ranges:
            if start > curr_time:
                if start - curr_time > 0.01:
                    missing_ranges.append([curr_time, start])
            curr_time = max(curr_time, end)
        if curr_time < duration:
            if duration - curr_time > 0.01:
                missing_ranges.append([curr_time, duration])
                
        spk_missing_seconds = sum(end - start for start, end in missing_ranges)
        total_missing_seconds += spk_missing_seconds
        
        # Count chunks
        chunks_expected = 0
        chunks_success = 0
        chunks_empty = 0
        chunks_failed = 0
        
        raw_output = (speaker_raw_outputs or {}).get(spk_id, {})
        if raw_output and "chunks" in raw_output:
            # NeMo backend chunking stats
            chunks_list = raw_output.get("chunks") or []
            failed_list = raw_output.get("failed_chunks") or []
            
            chunks_success = len(chunks_list)
            chunks_failed = len(failed_list)
            chunks_expected = chunks_success + chunks_failed
            
            for chunk in chunks_list:
                raw_res = chunk.get("raw_result") or {}
                text = raw_res.get("text") or ""
                # Also check segment list if text is empty
                segments = raw_res.get("segments") or raw_res.get("segment") or raw_res.get("timestamps") or []
                if not text.strip() and not segments:
                    chunks_empty += 1
        else:
            # Non-chunked Whisper-style backend
            chunks_expected = 1
            if spk_turns:
                chunks_success = 1
                chunks_empty = 0
            else:
                # If there are no turns, check if there's any raw transcript output
                if raw_output and raw_output.get("result"):
                    chunks_success = 1
                    chunks_empty = 1
                else:
                    chunks_success = 1
                    chunks_empty = 1
                
        speaker_reports[spk_id] = {
            "speaker_id": spk_id,
            "display_name": speaker.display_name,
            "source_file": speaker.file,
            "source_duration_seconds": duration,
            "chunks_expected": chunks_expected,
            "chunks_successfully_transcribed": chunks_success,
            "chunks_empty": chunks_empty,
            "chunks_failed": chunks_failed,
            "covered_ranges": merged_ranges,
            "missing_ranges": missing_ranges,
            "total_missing_seconds": round(spk_missing_seconds, 3)
        }
        
    # Determine completeness status
    # Max warning threshold from manifest
    max_missing_warn = getattr(manifest.completeness, "max_missing_seconds_warn", 30.0)
    
    warnings = []
    if total_missing_seconds > max_missing_warn:
        status = "incomplete"
        warnings.append(
            f"Transcription is incomplete. Missing {total_missing_seconds:.1f} seconds of audio coverage, "
            f"which exceeds the warning threshold of {max_missing_warn} seconds."
        )
    elif total_missing_seconds > 0:
        status = "complete_with_warnings"
        warnings.append(
            f"Transcription complete but with minor missing coverage: {total_missing_seconds:.1f} seconds."
        )
    else:
        status = "complete"
        
    return {
        "session_id": manifest.session.id,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "completeness_summary": {
            "status": status,
            "total_missing_seconds": round(total_missing_seconds, 3),
            "warnings": warnings
        },
        "speakers": speaker_reports
    }
