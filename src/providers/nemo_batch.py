from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.providers.base import _json_safe, transcribe_nemo_chunks
from src.runtime.output import configure_nemo_logging, configure_quiet_backend_logging, suppress_backend_output


def transcribe_nemo_batch(
    provider_name: str,
    model: Any,
    jobs: list[dict],
    output_dir: Path,
    batch_size: int = 0,
    resume: bool = False,
    sample_rate: int = 16000,
    chunk_format: str = "wav",
    overlap_seconds: float = 2.0,
    progress_callback: Any = None,
    verbose: bool = False,
) -> dict[str, dict]:
    """Transcribe round-robin chunks from multiple tracks in shared batches."""
    if not jobs:
        return {}

    configure_quiet_backend_logging(verbose=verbose)
    configure_nemo_logging(verbose=verbose)
    width = min(len(jobs), batch_size if batch_size > 0 else len(jobs))
    outputs = {
        job["speaker"].speaker_id: {
            "provider": provider_name,
            "speaker_id": job["speaker"].speaker_id,
            "chunks": [],
            "oom_events": [],
            "retried_chunks": [],
            "failed_chunks": [],
            "batch_fallback_events": [],
        }
        for job in jobs
    }
    positions = {job["speaker"].speaker_id: 0 for job in jobs}
    cursor = 0

    while any(positions[job["speaker"].speaker_id] < len(job["chunks"]) for job in jobs):
        pending = []
        scanned = 0
        while len(pending) < width and scanned < len(jobs):
            job = jobs[(cursor + scanned) % len(jobs)]
            speaker_id = job["speaker"].speaker_id
            if positions[speaker_id] < len(job["chunks"]):
                pending.append((job, job["chunks"][positions[speaker_id]]))
            scanned += 1
        cursor = (cursor + max(1, scanned)) % len(jobs)

        runnable = []
        for job, chunk in pending:
            speaker_id = job["speaker"].speaker_id
            raw_dir = output_dir / "raw" / provider_name / speaker_id
            raw_dir.mkdir(parents=True, exist_ok=True)
            raw_path = raw_dir / f'{chunk["chunk_id"]}.raw.json'
            if resume and raw_path.exists():
                try:
                    cached = json.loads(raw_path.read_text(encoding="utf-8"))
                    if "raw_result" in cached:
                        outputs[speaker_id]["chunks"].append(cached)
                        positions[speaker_id] += 1
                        if progress_callback:
                            progress_callback(speaker_id, _duration(chunk))
                        continue
                except Exception:
                    pass
            runnable.append((job, chunk, raw_path))

        if not runnable:
            continue

        paths = [_chunk_path(output_dir, chunk) for _, chunk, _ in runnable]
        try:
            with suppress_backend_output(enabled=not verbose):
                results = model.transcribe(
                    [str(path) for path in paths],
                    batch_size=len(paths),
                    timestamps=True,
                    verbose=verbose,
                )
            if not isinstance(results, list):
                results = [results]
            if len(results) != len(runnable):
                raise RuntimeError(f"NeMo returned {len(results)} results for {len(runnable)} inputs")
        except Exception as exc:
            results = [None] * len(runnable)
            event = {
                "batch_size": len(runnable),
                "speaker_ids": [job["speaker"].speaker_id for job, _, _ in runnable],
                "error": str(exc),
            }
            for job, _, _ in runnable:
                outputs[job["speaker"].speaker_id]["batch_fallback_events"].append(event)

        for index, (job, chunk, raw_path) in enumerate(runnable):
            speaker = job["speaker"]
            speaker_id = speaker.speaker_id
            if results[index] is None:
                fallback = transcribe_nemo_chunks(
                    provider_name=provider_name,
                    model=model,
                    audio_path=job["audio_path"],
                    speaker=speaker,
                    chunks=[chunk],
                    resume=False,
                    output_dir=output_dir,
                    sample_rate=sample_rate,
                    chunk_format=chunk_format,
                    overlap_seconds=overlap_seconds,
                    verbose=verbose,
                )
                payload = fallback["chunks"][0]
                for key in ("oom_events", "retried_chunks", "failed_chunks"):
                    outputs[speaker_id][key].extend(fallback[key])
            else:
                payload = {
                    "chunk_id": chunk["chunk_id"],
                    "chunk_start_seconds": chunk["chunk_start_seconds"],
                    "chunk_end_seconds": chunk["chunk_end_seconds"],
                    "raw_result": _json_safe(results[index]),
                }
                raw_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            outputs[speaker_id]["chunks"].append(payload)
            positions[speaker_id] += 1
            if progress_callback:
                progress_callback(speaker_id, _duration(chunk))

    return outputs


def _chunk_path(output_dir: Path, chunk: dict) -> Path:
    path = Path(chunk["chunk_file"])
    if not path.is_absolute():
        path = output_dir.parent / path
    return path.resolve()


def _duration(chunk: dict) -> float:
    duration = chunk["chunk_end_seconds"] - chunk["chunk_start_seconds"]
    return max(0.0, duration - float(chunk.get("overlap_seconds", 0.0)))