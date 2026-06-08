from __future__ import annotations

from pathlib import Path

from src.config.manifest import SessionManifest


def export_markdown(output_dir: Path, manifest: SessionManifest, turns: list[dict], chunks: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_transcript(output_dir / "transcript.md", manifest, turns)
    chunk_dir = output_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    for chunk in chunks:
        chunk_turns = [turn for turn in turns if turn["turn_id"] in chunk["turn_ids"]]
        _write_transcript(chunk_dir / f"{chunk['chunk_id']}.md", manifest, chunk_turns, title=chunk["chunk_id"])


def _write_transcript(path: Path, manifest: SessionManifest, turns: list[dict], title: str = "Transcript") -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"# {title}\n\n")
        f.write(f"- Session: {manifest.session.id}\n")
        f.write(f"- Campaign: {manifest.session.campaign}\n\n")
        for turn in turns:
            f.write(
                f"[{_format_time(turn['start_seconds'])} - {_format_time(turn['end_seconds'])}] "
                f"{turn['speaker_name']}: {turn['text_cleaned']}\n"
            )


def _format_time(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, ms = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"
