from __future__ import annotations

from pathlib import Path

from src.config.manifest import SessionManifest
from src.enrich.cleanup import clean_filler_words


def export_markdown(output_dir: Path, manifest: SessionManifest, turns: list[dict], chunks: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_transcript(output_dir / "transcript.md", manifest, turns)
    chunk_dir = output_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    for chunk in chunks:
        chunk_turns = [turn for turn in turns if turn["turn_id"] in chunk["turn_ids"]]
        _write_transcript(chunk_dir / f"{chunk['chunk_id']}.md", manifest, chunk_turns, title=chunk["chunk_id"])


def export_top_level_markdown(output_file: Path, manifest: SessionManifest, turns: list[dict]) -> None:
    backend = manifest.transcription.backend
    model = manifest.transcription.model
    if turns:
        backend = turns[0].get("backend", backend)
        model = turns[0].get("model", model)
        
    with output_file.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"# {manifest.session.campaign} - Session {manifest.session.session_number} Transcript\n\n")
        f.write(f"Session ID: {manifest.session.id}\n")
        f.write(f"Date: {manifest.session.session_date}\n")
        f.write(f"Backend: {backend}\n")
        f.write(f"Model: {model}\n\n")
        f.write("## Transcript\n\n")
        
        for turn in turns:
            speaker_label = turn["speaker_name"]
            role = next((s.role for s in manifest.speakers if s.speaker_id == turn["speaker_id"]), None)
            if role:
                speaker_label += f" / {role}"
                
            start_fmt = _format_hhmmss(turn["start_seconds"])
            end_fmt = _format_hhmmss(turn["end_seconds"])
            
            cleaned_text = clean_filler_words(turn["text_cleaned"])
            
            f.write(f"[{start_fmt} - {end_fmt}] {speaker_label}:\n{cleaned_text}\n\n")


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


def _format_hhmmss(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
