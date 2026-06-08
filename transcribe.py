#!/usr/bin/env python
"""Local-first D&D session transcription pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.audio.inspect import inspect_audio
from src.audio.prepare import prepare_audio
from src.config.manifest import load_manifest
from src.enrich.chunks import build_chunks
from src.enrich.corrections import apply_corrections
from src.enrich.entities import extract_entities
from src.enrich.glossary_candidates import build_glossary_candidates
from src.enrich.quality import build_quality_report
from src.exports.agents import export_agents
from src.exports.glossary import export_glossary_candidates
from src.exports.markdown import export_markdown
from src.exports.nocodb import export_nocodb
from src.exports.raw import export_raw
from src.exports.vtt import export_vtt
from src.normalize.merge import merge_turns
from src.normalize.turns import normalize_turns
from src.normalize.words import normalize_words
from src.providers.base import ProviderContext
from src.providers.canary import CanaryProvider
from src.providers.parakeet import ParakeetProvider
from src.providers.whisperx import WhisperXProvider


PROVIDERS = {
    "parakeet": ParakeetProvider,
    "canary": CanaryProvider,
    "whisperx": WhisperXProvider,
    "faster-whisper": WhisperXProvider,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe separate local speaker tracks into structured v2 outputs."
    )
    parser.add_argument(
        "--manifest",
        default="session_manifest.yaml",
        help="Path to session_manifest.yaml.",
    )
    parser.add_argument(
        "--output",
        help=(
            "Output directory for canonical CSV/JSON, agent, Markdown, VTT, and raw "
            "files. Defaults to an output/ folder next to the manifest."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=sorted(PROVIDERS),
        help="Override transcription.backend from the manifest.",
    )
    parser.add_argument("--model", help="Override transcription.model from the manifest.")
    parser.add_argument("--device", help="Override transcription.device from the manifest.")
    parser.add_argument(
        "--skip-transcription",
        action="store_true",
        help="Load provider raw JSON from output/raw instead of running local ASR.",
    )
    parser.add_argument(
        "--chunk-minutes",
        type=float,
        default=30.0,
        help="Time-based transcript chunk size.",
    )
    parser.add_argument(
        "--no-audio-normalize",
        action="store_true",
        help="Send manifest audio files directly to the provider instead of preparing mono 16 kHz WAV files.",
    )
    return parser.parse_args()


def get_provider(context: ProviderContext):
    provider_type = PROVIDERS.get(context.backend)
    if provider_type is None:
        supported = ", ".join(sorted(PROVIDERS))
        raise ValueError(f"Unsupported backend '{context.backend}'. Supported: {supported}")
    return provider_type(context)


def load_raw_output(raw_dir: Path, backend: str, speaker_id: str) -> dict:
    path = raw_dir / backend / f"{speaker_id}.raw.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing raw output for --skip-transcription: {path}"
        )
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    output_dir = resolve_output_dir(args.output, manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(manifest_path)
    manifest.apply_overrides(
        backend=args.backend,
        model=args.model,
        device=args.device,
    )
    manifest.validate_audio_files()

    raw_dir = output_dir / "raw"
    provider_context = ProviderContext(
        backend=manifest.transcription.backend,
        model=manifest.transcription.model,
        device=manifest.transcription.device,
        language=manifest.transcription.language,
    )
    provider = get_provider(provider_context)

    all_turns = []
    all_words = []
    audio_reports = {}

    print(f"Session: {manifest.session.id}")
    print(f"Backend: {provider_context.backend} ({provider_context.model})")
    print(f"Output: {output_dir}")

    for speaker in manifest.speakers:
        source_path = manifest.resolve_path(speaker.file)
        print(f"\nInspecting {speaker.speaker_id}: {source_path}")
        audio_info = inspect_audio(source_path)
        audio_reports[speaker.speaker_id] = audio_info
        prepared_path = (
            source_path
            if args.no_audio_normalize
            else prepare_audio(source_path, audio_info, output_dir / "prepared_audio")
        )

        if args.skip_transcription:
            raw_output = load_raw_output(raw_dir, provider_context.backend, speaker.speaker_id)
        else:
            print(f"Transcribing {speaker.speaker_id} locally...")
            raw_output = provider.transcribe(prepared_path, speaker)

        export_raw(raw_dir, provider_context.backend, speaker.speaker_id, raw_output)

        turns = normalize_turns(
            manifest=manifest,
            speaker=speaker,
            raw_output=raw_output,
            backend=provider_context.backend,
            model=provider_context.model,
        )
        words = normalize_words(
            manifest=manifest,
            speaker=speaker,
            raw_output=raw_output,
            turns=turns,
            backend=provider_context.backend,
            model=provider_context.model,
        )
        all_turns.extend(turns)
        all_words.extend(words)

    merged_turns = merge_turns(all_turns)
    cleaned_turns, corrections = apply_corrections(merged_turns, manifest.glossary)
    entities = extract_entities(cleaned_turns, manifest.glossary)
    glossary_candidates = build_glossary_candidates(
        manifest=manifest,
        turns=cleaned_turns,
        words=all_words,
        corrections=corrections,
    )
    chunks = build_chunks(cleaned_turns, manifest, chunk_minutes=args.chunk_minutes)
    quality_report = build_quality_report(
        manifest=manifest,
        turns=cleaned_turns,
        words=all_words,
        audio_reports=audio_reports,
        corrections=corrections,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )

    export_nocodb(
        output_dir=output_dir,
        manifest=manifest,
        turns=cleaned_turns,
        words=all_words,
        entities=entities,
        corrections=corrections,
        chunks=chunks,
        quality_report=quality_report,
        audio_reports=audio_reports,
    )
    export_glossary_candidates(output_dir, glossary_candidates)
    export_agents(output_dir / "agents", manifest, cleaned_turns, chunks, corrections)
    export_markdown(output_dir / "markdown", manifest, cleaned_turns, chunks)
    export_vtt(output_dir / "vtt", cleaned_turns)

    print("\nDone.")
    print(
        f"Wrote {len(cleaned_turns)} turns, {len(all_words)} words, "
        f"{len(chunks)} chunks, {len(glossary_candidates)} glossary candidates."
    )
    return 0


def resolve_output_dir(output_arg: str | None, manifest_path: Path) -> Path:
    if output_arg is None:
        return manifest_path.parent / "output"
    return Path(output_arg).resolve()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        raise SystemExit(130)
