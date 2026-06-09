from __future__ import annotations

import subprocess
from pathlib import Path


def prepare_audio(source: Path, prepared_dir: Path, sample_rate: int = 16000) -> Path:
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


def prepare_mono_flac(source: Path, prepared_dir: Path) -> Path:
    prepared_dir.mkdir(parents=True, exist_ok=True)
    target = prepared_dir / f"{source.stem}.mono.flac"
    if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
        return target

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-ac",
        "1",
        "-vn",
        "-c:a",
        "flac",
        str(target),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required for local audio channel downmixing.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed for {source}: {exc.stderr}") from exc
    return target


def needs_mono_downmix(audio_info: dict) -> bool:
    return int(audio_info.get("channels") or 0) > 1
