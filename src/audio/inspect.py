from __future__ import annotations

import json
import subprocess
from pathlib import Path


def inspect_audio(path: Path) -> dict:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe is required for local audio inspection.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffprobe failed for {path}: {exc.stderr}") from exc

    payload = json.loads(result.stdout)
    audio_stream = next(
        (stream for stream in payload.get("streams", []) if stream.get("codec_type") == "audio"),
        {},
    )
    fmt = payload.get("format", {})
    return {
        "path": str(path),
        "duration_seconds": float(audio_stream.get("duration") or fmt.get("duration") or 0),
        "sample_rate": int(audio_stream.get("sample_rate") or 0),
        "channels": int(audio_stream.get("channels") or 0),
        "format": fmt.get("format_name", ""),
        "codec": audio_stream.get("codec_name", ""),
    }
