from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from src.audio.inspect import inspect_audio
from src.config.manifest import SessionManifest, SpeakerConfig


def load_chunk_manifest(chunks_dir: Path) -> dict[str, Any]:
    manifest_path = chunks_dir / "chunk_manifest.json"
    if manifest_path.exists():
        try:
            with manifest_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"source_files": {}, "chunks": []}


def save_chunk_manifest(chunks_dir: Path, manifest_data: dict[str, Any]) -> None:
    chunks_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = chunks_dir / "chunk_manifest.json"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest_data, f, ensure_ascii=False, indent=2)


def get_file_metadata(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "mtime": stat.st_mtime,
        "size": stat.st_size,
    }


def chunk_speaker_audio(
    manifest: SessionManifest,
    speaker: SpeakerConfig,
    output_dir: Path,
    force_chunks: bool = False,
    progress_reporter: Any = None,
) -> list[dict[str, Any]]:
    """Chunks a speaker track and returns a list of chunk metadata dicts."""
    chunks_dir = output_dir / "chunks"
    speaker_chunks_dir = chunks_dir / speaker.speaker_id
    speaker_chunks_dir.mkdir(parents=True, exist_ok=True)

    source_path = manifest.resolve_path(speaker.file)
    meta = get_file_metadata(source_path)

    # Load existing manifest
    manifest_data = load_chunk_manifest(chunks_dir)
    existing_source = manifest_data.get("source_files", {}).get(speaker.file, {})
    
    # Check if we can reuse chunks
    reuse_existing = (
        not force_chunks
        and existing_source
        and existing_source.get("mtime") == meta["mtime"]
        and existing_source.get("size") == meta["size"]
    )

    if reuse_existing:
        # Check if all chunk files actually exist on disk
        speaker_chunks = [
            c for c in manifest_data.get("chunks", [])
            if c.get("speaker_id") == speaker.speaker_id
        ]
        if speaker_chunks and all((chunks_dir.parent / c["chunk_file"]).exists() for c in speaker_chunks):
            # Sort by index
            speaker_chunks.sort(key=lambda x: x["chunk_index"])
            
            # Clean up orphaned chunks
            active_filenames = {Path(c["chunk_file"]).name for c in speaker_chunks}
            deleted_count = 0
            if speaker_chunks_dir.exists():
                for f in speaker_chunks_dir.iterdir():
                    if f.is_file() and f.name not in active_filenames:
                        try:
                            f.unlink()
                            deleted_count += 1
                        except Exception:
                            pass
            
            if deleted_count > 0 and progress_reporter:
                progress_reporter.report_cleanup(deleted_count)

            if progress_reporter:
                progress_reporter.report_reuse(speaker.speaker_id, len(speaker_chunks))
                
            return speaker_chunks

    # Otherwise, we need to chunk!
    audio_info = inspect_audio(source_path)
    duration = audio_info["duration_seconds"]

    chunk_seconds = getattr(manifest.audio, "chunk_seconds", 60.0)
    overlap_seconds = getattr(manifest.audio, "overlap_seconds", 2.0)
    min_chunk_seconds = getattr(manifest.audio, "min_chunk_seconds", 15.0)
    chunk_format = getattr(manifest.audio, "chunk_format", "wav")

    # Generate chunk intervals
    intervals: list[tuple[float, float]] = []
    start = 0.0
    chunk_idx = 0
    while start < duration:
        end = min(start + chunk_seconds, duration)
        # Check min chunk constraint for subsequent chunks
        if chunk_idx > 0 and (duration - start) < min_chunk_seconds:
            # Extend the previous chunk to the end of the file
            if intervals:
                prev_start, _ = intervals[-1]
                intervals[-1] = (prev_start, duration)
            else:
                intervals.append((start, duration))
            break
        intervals.append((start, end))
        start += chunk_seconds - overlap_seconds
        chunk_idx += 1

    # Report starting of chunking
    phase = "regenerating chunks" if force_chunks else "generating chunks"
    if progress_reporter:
        progress_reporter.start_chunking(
            speaker_id=speaker.speaker_id,
            phase=phase,
            num_chunks=len(intervals)
        )

    # Write chunks using ffmpeg
    new_chunks = []
    for idx, (c_start, c_end) in enumerate(intervals):
        chunk_id = f"{speaker.speaker_id}_chunk_{idx:05d}"
        chunk_filename = f"{chunk_id}.{chunk_format}"
        chunk_file_path = speaker_chunks_dir / chunk_filename
        c_duration = c_end - c_start

        # Normalize audio during chunk extraction (mono, 16kHz)
        sample_rate = getattr(manifest.audio, "sample_rate", 16000)
        command = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{c_start:.3f}",
            "-t",
            f"{c_duration:.3f}",
            "-i",
            str(source_path),
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
        ]
        if chunk_format == "flac":
            command.extend(["-c:a", "flac"])
        command.append(str(chunk_file_path))

        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"ffmpeg chunking failed for {source_path}: {exc.stderr}")

        if progress_reporter:
            progress_reporter.complete_chunk_prep(
                speaker_id=speaker.speaker_id,
                chunk_duration=c_duration,
                is_regenerated=force_chunks
            )

        # Path of the chunk relative to the output_dir
        rel_chunk_file = os.path.relpath(chunk_file_path, output_dir)

        new_chunks.append({
            "session_id": manifest.session.id,
            "speaker_id": speaker.speaker_id,
            "source_file": speaker.file,
            "chunk_id": chunk_id,
            "chunk_index": idx,
            "chunk_file": f"output/{rel_chunk_file.replace('\\', '/')}",
            "chunk_start_seconds": c_start,
            "chunk_end_seconds": c_end,
            "overlap_seconds": overlap_seconds if idx > 0 else 0.0,
        })

    # Clean up orphaned chunks
    active_filenames = {Path(c["chunk_file"]).name for c in new_chunks}
    deleted_count = 0
    if speaker_chunks_dir.exists():
        for f in speaker_chunks_dir.iterdir():
            if f.is_file() and f.name not in active_filenames:
                try:
                    f.unlink()
                    deleted_count += 1
                except Exception:
                    pass
    if deleted_count > 0 and progress_reporter:
        progress_reporter.report_cleanup(deleted_count)

    # Update manifest
    manifest_data["source_files"][speaker.file] = meta
    # Remove existing chunks for this speaker from manifest_data
    manifest_data["chunks"] = [
        c for c in manifest_data.get("chunks", [])
        if c.get("speaker_id") != speaker.speaker_id
    ]
    manifest_data["chunks"].extend(new_chunks)
    save_chunk_manifest(chunks_dir, manifest_data)

    return new_chunks
