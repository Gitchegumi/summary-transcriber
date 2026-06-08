from __future__ import annotations

import subprocess
from pathlib import Path


DIRECT_USE_EXTENSIONS = {".flac", ".wav"}


def prepare_audio(
    source: Path,
    audio_info: dict,
    prepared_dir: Path,
    sample_rate: int = 16000,
) -> Path:
    if _can_use_directly(source, audio_info, sample_rate):
        return source

    prepared_dir.mkdir(parents=True, exist_ok=True)
    target = prepared_dir / f"{source.stem}.mono{sample_rate}.wav"
    if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
        return target

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-vn",
        str(target),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required for local audio normalization.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed for {source}: {exc.stderr}") from exc
    return target


def _can_use_directly(source: Path, audio_info: dict, sample_rate: int) -> bool:
    return (
        source.suffix.lower() in DIRECT_USE_EXTENSIONS
        and int(audio_info.get("channels") or 0) == 1
        and int(audio_info.get("sample_rate") or 0) == sample_rate
    )
