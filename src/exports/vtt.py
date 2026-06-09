from __future__ import annotations

from pathlib import Path


def export_vtt(output_dir: Path, turns: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "merged.vtt").open("w", encoding="utf-8", newline="\n") as f:
        f.write("WEBVTT\n\n")
        for index, turn in enumerate(turns, start=1):
            f.write(f"{index}\n")
            f.write(f"{_vtt_time(turn['start_seconds'])} --> {_vtt_time(turn['end_seconds'])}\n")
            f.write(f"<v {turn['speaker_name']}>{turn['text_cleaned']}\n\n")


def _vtt_time(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, ms = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"
