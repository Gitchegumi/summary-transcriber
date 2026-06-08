from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, asdict, is_dataclass
from pathlib import Path
from typing import Any, Protocol

from src.config.manifest import SpeakerConfig


@dataclass
class ProviderContext:
    backend: str
    model: str
    device: str = "cuda"
    language: str = "en"
    verbose: bool = False


class TranscriptionProvider(Protocol):
    context: ProviderContext

    def transcribe(
        self,
        audio_path: Path,
        speaker: SpeakerConfig,
        chunks: list[dict] | None = None,
        resume: bool = False,
        output_dir: Path | None = None,
    ) -> dict:
        """Return raw local model output as a JSON-serializable dictionary."""


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if hasattr(value, "_asdict"):
        return _json_safe(value._asdict())
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return {
            key: _json_safe(item)
            for key, item in value.__dict__.items()
            if not key.startswith("_")
        }
    return str(value)


def transcribe_nemo_chunks(
    provider_name: str,
    model: Any,
    audio_path: Path,
    speaker: SpeakerConfig,
    chunks: list[dict],
    resume: bool,
    output_dir: Path,
    continue_on_error: bool = True,
    sample_rate: int = 16000,
    chunk_format: str = "wav",
    overlap_seconds: float = 2.0,
    progress_callback: Any = None,
    verbose: bool = False,
) -> dict:
    """Sequentially transcribes audio chunks, with CUDA OOM retry logic and resume support."""
    import torch

    raw_speaker_dir = output_dir / "raw" / provider_name / speaker.speaker_id
    raw_speaker_dir.mkdir(parents=True, exist_ok=True)

    oom_events = []
    retried_chunks = []
    failed_chunks = []
    results = []

    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        c_start = chunk["chunk_start_seconds"]
        c_end = chunk["chunk_end_seconds"]
        
        raw_chunk_path = raw_speaker_dir / f"{chunk_id}.raw.json"

        # Resume support: check if already successfully transcribed
        if resume and raw_chunk_path.exists():
            try:
                with raw_chunk_path.open("r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                if "raw_result" in cached_data:
                    results.append(cached_data)
                    if progress_callback:
                        progress_callback(c_end - c_start)
                    continue
            except Exception:
                pass

        # Transcribe with OOM retry logic
        # Temporary directory inside output_dir for retried clips
        temp_dir = output_dir / "temp_clips" / speaker.speaker_id
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            chunk_results = _transcribe_chunk_with_retry(
                model=model,
                source_audio_path=audio_path,
                speaker_id=speaker.speaker_id,
                chunk_id=chunk_id,
                start_time=c_start,
                end_time=c_end,
                step_idx=0,  # Starts at 60s (or current chunk length)
                temp_dir=temp_dir,
                sample_rate=sample_rate,
                chunk_format=chunk_format,
                overlap_seconds=overlap_seconds,
                continue_on_error=continue_on_error,
                oom_events=oom_events,
                retried_chunks=retried_chunks,
                failed_chunks=failed_chunks,
                verbose=verbose,
            )

            # Save raw outputs
            # If there was a single result, save to raw_chunk_path
            # If split into sub-chunks, we merge or save individually.
            # To match structure, we can write the chunk result and save it.
            # We'll save a consolidated file for this chunk ID.
            if len(chunk_results) == 1:
                chunk_payload = chunk_results[0]
            else:
                # Merge sub-chunk results into a single segment list
                merged_segments = []
                for cr in chunk_results:
                    raw_res = cr["raw_result"]
                    # If raw_res has segments:
                    segments = raw_res.get("segments") or raw_res.get("segment") or raw_res.get("timestamps") or []
                    if not isinstance(segments, list):
                        segments = [segments]
                    # We need to offset start/end by the sub-chunk offset relative to the main chunk
                    sub_offset = cr["chunk_start_seconds"] - c_start
                    for seg in segments:
                        if isinstance(seg, dict):
                            seg_copy = dict(seg)
                            if "start" in seg_copy:
                                seg_copy["start"] = float(seg_copy["start"]) + sub_offset
                            if "end" in seg_copy:
                                seg_copy["end"] = float(seg_copy["end"]) + sub_offset
                            merged_segments.append(seg_copy)
                
                chunk_payload = {
                    "chunk_id": chunk_id,
                    "chunk_start_seconds": c_start,
                    "chunk_end_seconds": c_end,
                    "raw_result": {"segments": merged_segments, "text": " ".join(cr["raw_result"].get("text", "") for cr in chunk_results)}
                }

            with raw_chunk_path.open("w", encoding="utf-8", newline="\n") as f:
                json.dump(chunk_payload, f, ensure_ascii=False, indent=2)

            results.append(chunk_payload)

        except Exception as exc:
            # If continue_on_error is true, we should log a failed chunk but not raise.
            # Otherwise raise.
            if continue_on_error:
                failed_chunks.append({
                    "chunk_id": chunk_id,
                    "start_seconds": c_start,
                    "end_seconds": c_end,
                    "reason": f"Unhandled exception: {exc}",
                })
                # Write empty chunk output
                chunk_payload = {
                    "chunk_id": chunk_id,
                    "chunk_start_seconds": c_start,
                    "chunk_end_seconds": c_end,
                    "raw_result": {"segments": [], "text": ""},
                }
                with raw_chunk_path.open("w", encoding="utf-8", newline="\n") as f:
                    json.dump(chunk_payload, f, ensure_ascii=False, indent=2)
                results.append(chunk_payload)
            else:
                raise exc

        if progress_callback:
            progress_callback(c_end - c_start)

    return {
        "provider": provider_name,
        "speaker_id": speaker.speaker_id,
        "chunks": results,
        "oom_events": oom_events,
        "retried_chunks": retried_chunks,
        "failed_chunks": failed_chunks,
    }


def _transcribe_chunk_with_retry(
    model: Any,
    source_audio_path: Path,
    speaker_id: str,
    chunk_id: str,
    start_time: float,
    end_time: float,
    step_idx: int,
    temp_dir: Path,
    sample_rate: int,
    chunk_format: str,
    overlap_seconds: float,
    continue_on_error: bool,
    oom_events: list,
    retried_chunks: list,
    failed_chunks: list,
    verbose: bool = False,
) -> list[dict]:
    """Helper to transcribe chunk with OOM retry sequence (60s -> 30s -> 15s)."""
    import torch

    steps = [60.0, 30.0, 15.0]
    current_step_sec = steps[step_idx]

    # Generate a temporary audio clip for this interval
    duration = end_time - start_time
    clip_file = temp_dir / f"{chunk_id}_s{start_time:.3f}_e{end_time:.3f}.{chunk_format}"

    command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_time:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(source_audio_path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
    ]
    if chunk_format == "flac":
        command.extend(["-c:a", "flac"])
    command.append(str(clip_file))

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg clipping failed: {exc.stderr}")

    try:
        from src.runtime.output import suppress_backend_output, configure_quiet_backend_logging, configure_nemo_logging
        configure_quiet_backend_logging(verbose=verbose)
        configure_nemo_logging(verbose=verbose)
        
        # Load and run model transcription on the chunk
        with suppress_backend_output(enabled=not verbose) as suppressor:
            result = model.transcribe([str(clip_file)], timestamps=True, verbose=verbose)
            
        raw_res = result[0] if isinstance(result, list) else result

        # Clean up temporary clip file
        if clip_file.exists():
            try:
                clip_file.unlink()
            except Exception:
                pass

        return [
            {
                "chunk_id": chunk_id,
                "chunk_start_seconds": start_time,
                "chunk_end_seconds": end_time,
                "raw_result": _json_safe(raw_res),
            }
        ]
    except Exception as exc:
        msg = str(exc).lower()
        is_oom = "out of memory" in msg or "cuda out of memory" in msg or "oom" in msg
        
        # Clean up temporary clip file
        if clip_file.exists():
            try:
                clip_file.unlink()
            except Exception:
                pass

        captured = ""
        if 'suppressor' in locals() and suppressor.enabled:
            captured = suppressor.get_captured_output()

        log_file_path = None
        if captured:
            try:
                logs_dir = temp_dir.parent.parent / "logs"
                logs_dir.mkdir(parents=True, exist_ok=True)
                log_file = logs_dir / f"nemo_error_{chunk_id}.log"
                with log_file.open("w", encoding="utf-8") as f:
                    f.write(f"Exception Type: {type(exc).__name__}\n")
                    f.write(f"Exception Message: {str(exc)}\n")
                    f.write(f"Retry Attempt: {step_idx}\n")
                    f.write(f"Backend Output Suppressed: {not verbose}\n\n")
                    f.write(captured)
                log_file_path = str(log_file)
            except Exception:
                pass

        diag_info = {
            "chunk_id": chunk_id,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "retry_attempt": step_idx,
            "backend_output_suppressed": not verbose,
            "debug_log_path": log_file_path,
        }

        if is_oom:
            torch.cuda.empty_cache()
            oom_events.append(
                {
                    "chunk_id": chunk_id,
                    "start_seconds": start_time,
                    "end_seconds": end_time,
                    "step_seconds": current_step_sec,
                    "error": str(exc),
                    **diag_info,
                }
            )

            # If we can go smaller
            if step_idx + 1 < len(steps):
                next_step_sec = steps[step_idx + 1]
                retried_chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "start_seconds": start_time,
                        "end_seconds": end_time,
                        "retry_seconds": next_step_sec,
                    }
                )

                sub_chunks_results = []
                sub_start = start_time
                sub_idx = 0
                while sub_start < end_time:
                    sub_end = min(sub_start + next_step_sec, end_time)
                    sub_chunk_id = f"{chunk_id}_retry_{int(next_step_sec)}s_{sub_idx:02d}"

                    res = _transcribe_chunk_with_retry(
                        model=model,
                        source_audio_path=source_audio_path,
                        speaker_id=speaker_id,
                        chunk_id=sub_chunk_id,
                        start_time=sub_start,
                        end_time=sub_end,
                        step_idx=step_idx + 1,
                        temp_dir=temp_dir,
                        sample_rate=sample_rate,
                        chunk_format=chunk_format,
                        overlap_seconds=overlap_seconds,
                        continue_on_error=continue_on_error,
                        oom_events=oom_events,
                        retried_chunks=retried_chunks,
                        failed_chunks=failed_chunks,
                        verbose=verbose,
                    )
                    sub_chunks_results.extend(res)

                    sub_start += next_step_sec - overlap_seconds
                    sub_idx += 1
                return sub_chunks_results
            else:
                # OOM at minimum size (15s)
                failed_chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "start_seconds": start_time,
                        "end_seconds": end_time,
                        "reason": "CUDA OOM at minimum chunk size (15s)",
                        **diag_info,
                    }
                )
                if continue_on_error:
                    return [
                        {
                            "chunk_id": chunk_id,
                            "chunk_start_seconds": start_time,
                            "chunk_end_seconds": end_time,
                            "raw_result": {"segments": [], "text": ""},
                        }
                    ]
                else:
                    raise exc
        else:
            # Non-OOM exception
            if continue_on_error:
                failed_chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "start_seconds": start_time,
                        "end_seconds": end_time,
                        "reason": f"Exception: {exc}",
                        **diag_info,
                    }
                )
                return [
                    {
                        "chunk_id": chunk_id,
                        "chunk_start_seconds": start_time,
                        "chunk_end_seconds": end_time,
                        "raw_result": {"segments": [], "text": ""},
                    }
                ]
            else:
                raise exc
