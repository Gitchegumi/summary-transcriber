from __future__ import annotations

import subprocess
from pathlib import Path


def clip_audio(source: Path, target: Path, start_seconds: float, duration_seconds: float) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_seconds:.3f}",
        "-t",
        f"{duration_seconds:.3f}",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(target),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return target
