#!/usr/bin/env python
"""Local-first D&D session transcription pipeline."""

from __future__ import annotations

import argparse
import json
import sys

# Check for verbose flag early to configure quiet backend logging before anything else
_early_verbose = "--verbose" in sys.argv
from src.runtime.output import configure_quiet_backend_logging
configure_quiet_backend_logging(verbose=_early_verbose)

import time
from datetime import datetime, timezone
from pathlib import Path

from src.audio.inspect import inspect_audio
from src.audio.prepare import needs_mono_downmix, prepare_audio, prepare_mono_flac
from src.audio.chunk import chunk_speaker_audio
from src.config.manifest import load_manifest
from src.enrich.chunks import build_chunks
from src.enrich.corrections import apply_corrections
from src.enrich.entities import extract_entities
from src.enrich.glossary_candidates import build_glossary_candidates
from src.enrich.quality import build_quality_report
from src.progress.reporter import ProgressReporter, ChunkPrepProgressReporter, format_duration
from src.exports.agents import export_agents
from src.exports.glossary import export_glossary_candidates, export_draft_outputs, export_noise_report
from src.exports.nocodb import export_nocodb
from src.exports.raw import export_raw
from src.exports.vtt import export_vtt
from src.normalize.merge import merge_turns
from src.normalize.turns import normalize_turns
from src.normalize.words import normalize_words
from src.providers.base import ProviderContext
from src.runtime.output import configure_quiet_backend_logging
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
        "--mode",
        choices=["draft", "finalize"],
        default="draft",
        help="Pipeline workflow mode: draft (transcribe and check glossary candidates) or finalize (apply corrections and export).",
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
        "--normalize-audio",
        action="store_true",
        help="Prepare mono 16 kHz WAV files before transcription instead of using direct or mono-FLAC input.",
    )
    parser.add_argument(
        "--force-chunks",
        action="store_true",
        help="Force re-chunking of audio files even if they are unchanged.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume transcription, skipping chunks that have already been successfully transcribed.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress reporting readout.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose mode, showing full log outputs from ASR backends.",
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Enable boundary overlap deduplication (default: False).",
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


def load_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing draft file: {path}")
    import csv
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
        # Coerce types for fields
        for row in rows:
            if "start_seconds" in row:
                row["start_seconds"] = float(row["start_seconds"]) if row["start_seconds"] != "" else 0.0
            if "end_seconds" in row:
                row["end_seconds"] = float(row["end_seconds"]) if row["end_seconds"] != "" else 0.0
            if "duration_seconds" in row:
                row["duration_seconds"] = float(row["duration_seconds"]) if row["duration_seconds"] != "" else 0.0
            if "confidence_avg" in row:
                row["confidence_avg"] = float(row["confidence_avg"]) if row["confidence_avg"] != "" else None
            if "word_count" in row:
                row["word_count"] = int(row["word_count"]) if row["word_count"] != "" else 0
            if "confidence" in row:
                row["confidence"] = float(row["confidence"]) if row["confidence"] != "" else None
            if "is_low_confidence" in row:
                row["is_low_confidence"] = str(row["is_low_confidence"]).lower() == "true"
        return rows


def resolve_output_dir(output_arg: str | None, manifest_path: Path) -> Path:
    if output_arg is None:
        return manifest_path.parent / "output"
    return Path(output_arg).resolve()


def requires_mono_provider_input(backend: str) -> bool:
    return backend in {"parakeet", "canary"}


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
    configure_quiet_backend_logging(verbose=args.verbose)
    provider_context = ProviderContext(
        backend=manifest.transcription.backend,
        model=manifest.transcription.model,
        device=manifest.transcription.device,
        language=manifest.transcription.language,
        verbose=args.verbose,
    )

    if args.deduplicate:
        manifest.deduplication.enabled = True

    from src.normalize.turns import DEDUPLICATION_DECISIONS, DISCARDED_TURNS
    DEDUPLICATION_DECISIONS.clear()
    DISCARDED_TURNS.clear()

    # --- FINALIZE MODE ---
    if args.mode == "finalize":
        draft_turns_path = output_dir / "draft" / "transcript_turns.draft.csv"
        draft_words_path = output_dir / "draft" / "transcript_words.draft.csv"
        
        print("Finalize mode: loading draft outputs...")
        try:
            draft_turns = load_csv_rows(draft_turns_path)
            draft_words = load_csv_rows(draft_words_path)
        except FileNotFoundError as exc:
            print(f"Error: {exc}. Please run draft mode first to generate draft files.", file=sys.stderr)
            return 1
            
        print(f"Loaded {len(draft_turns)} draft turns, {len(draft_words)} draft words.")
        
        audio_reports = {}
        for speaker in manifest.speakers:
            source_path = manifest.resolve_path(speaker.file)
            audio_reports[speaker.speaker_id] = inspect_audio(source_path)
            
        print("Applying glossary corrections...")
        cleaned_turns, corrections = apply_corrections(draft_turns, manifest.glossary)
        entities = extract_entities(cleaned_turns, manifest.glossary)
        chunks = build_chunks(cleaned_turns, manifest, chunk_minutes=args.chunk_minutes)
        
        # Try to load existing quality report to preserve chunk preparation metrics
        existing_report_path = output_dir / "transcript_quality_report.json"
        existing_metrics = {}
        if existing_report_path.exists():
            try:
                with existing_report_path.open("r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                    for k in [
                        "chunk_preparation_wall_clock_time",
                        "total_chunks_expected",
                        "chunks_generated",
                        "chunks_reused",
                        "chunks_skipped",
                        "chunks_regenerated",
                        "chunks_deleted",
                        "chunking_skipped_whisper",
                    ]:
                        if k in existing_data:
                            existing_metrics[k] = existing_data[k]
            except Exception:
                pass

        quality_report = build_quality_report(
            manifest=manifest,
            turns=cleaned_turns,
            words=draft_words,
            audio_reports=audio_reports,
            corrections=corrections,
            generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            nemo_chunking_used=(manifest.transcription.backend in {"parakeet", "canary"}),
            chunk_duration=manifest.audio.chunk_seconds,
            wall_clock_time=0.0,
            **existing_metrics,
        )
        
        print("Exporting canonical and export formats...")
        
        # Load raw outputs in finalize mode if they exist to populate completeness reports
        speaker_raw_outputs = {}
        for speaker in manifest.speakers:
            try:
                speaker_raw_outputs[speaker.speaker_id] = load_raw_output(raw_dir, manifest.transcription.backend, speaker.speaker_id)
            except Exception:
                pass

        from src.enrich.completeness import compute_completeness_report
        completeness_report = compute_completeness_report(manifest, cleaned_turns, audio_reports, speaker_raw_outputs)
        
        # Save output/transcript_completeness_report.json
        comp_report_path = output_dir / "transcript_completeness_report.json"
        with comp_report_path.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(completeness_report, f, ensure_ascii=False, indent=2)

        from src.exports.markdown import export_top_level_markdown
        export_top_level_markdown(output_dir / "transcript.md", manifest, cleaned_turns)

        export_nocodb(
            output_dir=output_dir,
            manifest=manifest,
            turns=cleaned_turns,
            words=draft_words,
            entities=entities,
            corrections=corrections,
            chunks=chunks,
            quality_report=quality_report,
            audio_reports=audio_reports,
        )
        export_agents(output_dir / "agents", manifest, cleaned_turns, chunks, corrections, completeness_report, audio_reports)
        export_vtt(output_dir / "vtt", cleaned_turns)
        
        print("\nFinalize mode complete.")
        print(f"Wrote {len(cleaned_turns)} turns, {len(draft_words)} words, {len(chunks)} chunks.")
        
        comp_summary = completeness_report["completeness_summary"]
        for w in comp_summary["warnings"]:
            print(f"WARNING: {w}", file=sys.stderr)
            
        if manifest.completeness.fail_on_missing_coverage and comp_summary["status"] == "incomplete":
            print("ERROR: Transcription completeness check failed. Exiting.", file=sys.stderr)
            return 1
            
        return 0

    # --- DRAFT MODE ---
    provider = get_provider(provider_context)

    is_nemo = provider_context.backend in {"parakeet", "canary"}
    
    # Initialize prep metrics
    chunk_prep_time = 0.0
    chunks_generated = 0
    chunks_reused = 0
    chunks_skipped = 0
    chunks_regenerated = 0
    chunks_deleted = 0

    progress_enabled = manifest.progress.enabled and not args.no_progress
    update_interval = manifest.progress.update_interval_seconds

    # Initialize prep progress reporter if NeMo is selected
    prep_reporter = ChunkPrepProgressReporter(
        session_id=manifest.session.id,
        backend=provider_context.backend,
        enabled=progress_enabled,
        update_interval_seconds=update_interval,
        total_speakers=len(manifest.speakers),
    )
    prep_reporter.set_speaker_ids([s.speaker_id for s in manifest.speakers])

    audio_reports = {}
    for idx, speaker in enumerate(manifest.speakers):
        source_path = manifest.resolve_path(speaker.file)
        if is_nemo:
            prep_reporter.start_speaker(
                speaker_id=speaker.speaker_id,
                display_name=speaker.display_name or speaker.character_name or "",
                source_file=speaker.file,
                speaker_index=idx,
                phase="inspecting",
            )
        audio_reports[speaker.speaker_id] = inspect_audio(source_path)

    total_audio_seconds = sum(info["duration_seconds"] for info in audio_reports.values())
    prep_reporter.set_total_audio_seconds(total_audio_seconds)
    for speaker in manifest.speakers:
        prep_reporter.set_speaker_duration(
            speaker.speaker_id, audio_reports[speaker.speaker_id]["duration_seconds"]
        )

    all_chunks = {}
    total_chunks = 0
    prepared_paths = {}

    if is_nemo:
        chunk_prep_start = time.time()
        for idx, speaker in enumerate(manifest.speakers):
            source_path = manifest.resolve_path(speaker.file)
            prepared_path = source_path
            
            # Normalization / downmixing step
            if args.normalize_audio:
                prep_reporter.start_speaker(
                    speaker_id=speaker.speaker_id,
                    display_name=speaker.display_name or speaker.character_name or "",
                    source_file=speaker.file,
                    speaker_index=idx,
                    phase="normalizing",
                )
                prepared_path = prepare_audio(source_path, output_dir / "prepared_audio")
            elif requires_mono_provider_input(provider_context.backend) and needs_mono_downmix(audio_reports[speaker.speaker_id]):
                prep_reporter.start_speaker(
                    speaker_id=speaker.speaker_id,
                    display_name=speaker.display_name or speaker.character_name or "",
                    source_file=speaker.file,
                    speaker_index=idx,
                    phase="normalizing",
                )
                prepared_path = prepare_mono_flac(source_path, output_dir / "prepared_audio")
                
            prepared_paths[speaker.speaker_id] = prepared_path
            
            # Chunking step
            chunks_meta = chunk_speaker_audio(
                manifest=manifest,
                speaker=speaker,
                output_dir=output_dir,
                force_chunks=args.force_chunks,
                progress_reporter=prep_reporter,
            )
            all_chunks[speaker.speaker_id] = chunks_meta
            total_chunks += len(chunks_meta)

        prep_reporter.set_phase("completed")
        prep_reporter.finish()
        
        chunk_prep_time = time.time() - chunk_prep_start
        chunks_generated = prep_reporter.chunks_generated
        chunks_reused = prep_reporter.chunks_reused
        chunks_skipped = prep_reporter.chunks_skipped
        chunks_regenerated = prep_reporter.chunks_regenerated
        chunks_deleted = prep_reporter.chunks_deleted

    reporter = ProgressReporter(
        session_id=manifest.session.id,
        backend=provider_context.backend,
        enabled=progress_enabled,
        update_interval_seconds=update_interval,
    )
    if is_nemo:
        reporter.init_totals(total_chunks, total_audio_seconds)

    all_turns = []
    all_words = []
    speaker_raw_outputs = {}
    
    oom_events = []
    retried_chunks = []
    failed_chunks = []
    chunk_counts = {}

    start_wall_time = time.time()

    for speaker in manifest.speakers:
        source_path = manifest.resolve_path(speaker.file)
        if is_nemo:
            prepared_path = prepared_paths[speaker.speaker_id]
        else:
            prepared_path = source_path
            if args.normalize_audio:
                prepared_path = prepare_audio(source_path, output_dir / "prepared_audio")
            elif requires_mono_provider_input(provider_context.backend) and needs_mono_downmix(audio_reports[speaker.speaker_id]):
                prepared_path = prepare_mono_flac(source_path, output_dir / "prepared_audio")

        if args.skip_transcription:
            raw_output = load_raw_output(raw_dir, provider_context.backend, speaker.speaker_id)
        else:
            if is_nemo:
                speaker_chunks = all_chunks[speaker.speaker_id]
                speaker_audio_dur = audio_reports[speaker.speaker_id]["duration_seconds"]
                
                reporter.start_speaker(
                    speaker_id=speaker.speaker_id,
                    display_name=speaker.display_name,
                    speaker_chunks_total=len(speaker_chunks),
                    speaker_audio_total=speaker_audio_dur,
                )
                
                def progress_cb(chunk_duration):
                    reporter.complete_chunk(chunk_duration)
                    
                raw_output = provider.transcribe(
                    audio_path=prepared_path,
                    speaker=speaker,
                    chunks=speaker_chunks,
                    resume=args.resume,
                    output_dir=output_dir,
                    progress_callback=progress_cb,
                    sample_rate=manifest.audio.sample_rate,
                    chunk_format=manifest.audio.chunk_format,
                    overlap_seconds=manifest.audio.overlap_seconds,
                )
            else:
                if progress_enabled:
                    print(f"[{format_duration(time.time() - start_wall_time)}] Transcribing speaker {speaker.speaker_id}...")
                
                raw_output = provider.transcribe(prepared_path, speaker)
                
                if progress_enabled:
                    print(f"[{format_duration(time.time() - start_wall_time)}] Completed transcribing {speaker.speaker_id}.")

        export_raw(raw_dir, provider_context.backend, speaker.speaker_id, raw_output)
        speaker_raw_outputs[speaker.speaker_id] = raw_output

        if "oom_events" in raw_output:
            oom_events.extend(raw_output["oom_events"])
        if "retried_chunks" in raw_output:
            retried_chunks.extend(raw_output["retried_chunks"])
        if "failed_chunks" in raw_output:
            failed_chunks.extend(raw_output["failed_chunks"])
        if "chunks" in raw_output:
            chunk_counts[speaker.speaker_id] = len(raw_output["chunks"])

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

    reporter.finish()
    
    wall_clock_time = time.time() - start_wall_time

    merged_turns = merge_turns(all_turns)
    cleaned_turns, corrections = apply_corrections(merged_turns, manifest.glossary)
    entities = extract_entities(cleaned_turns, manifest.glossary)
    
    glossary_candidates, noise_candidates = build_glossary_candidates(
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
        speaker_raw_outputs=speaker_raw_outputs,
        oom_events=oom_events,
        retried_chunks=retried_chunks,
        failed_chunks=failed_chunks,
        deduplication_decisions=list(DEDUPLICATION_DECISIONS),
        nemo_chunking_used=is_nemo,
        chunk_duration=manifest.audio.chunk_seconds,
        wall_clock_time=wall_clock_time,
        chunk_preparation_wall_clock_time=chunk_prep_time,
        total_chunks_expected=total_chunks,
        chunks_generated=chunks_generated,
        chunks_reused=chunks_reused,
        chunks_skipped=chunks_skipped,
        chunks_regenerated=chunks_regenerated,
        chunks_deleted=chunks_deleted,
        chunking_skipped_whisper=not is_nemo,
    )

    from src.enrich.completeness import compute_completeness_report
    completeness_report = compute_completeness_report(manifest, cleaned_turns, audio_reports, speaker_raw_outputs)
    
    # Save output/transcript_completeness_report.json
    comp_report_path = output_dir / "transcript_completeness_report.json"
    with comp_report_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(completeness_report, f, ensure_ascii=False, indent=2)

    from src.exports.markdown import export_top_level_markdown
    export_top_level_markdown(output_dir / "transcript.md", manifest, cleaned_turns)

    # Write deduplication audit file if enabled
    if manifest.deduplication.enabled:
        audit_path = output_dir / "deduplication_audit.jsonl"
        with audit_path.open("w", encoding="utf-8", newline="\n") as f:
            for turn in DISCARDED_TURNS:
                f.write(json.dumps(turn, ensure_ascii=False) + "\n")

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
    export_draft_outputs(output_dir, glossary_candidates, cleaned_turns, all_words)
    export_noise_report(output_dir, noise_candidates, manifest.glossary_candidate_filters.include_noise_report)
    export_vtt(output_dir / "vtt", cleaned_turns)

    # Print summary to console
    total_audio_seconds = sum(info.get("duration_seconds", 0) for info in audio_reports.values())
    avg_speed = (total_audio_seconds / 60.0) / (wall_clock_time / 60.0) if wall_clock_time > 0 else 0.0
    
    print("\n--- Transcription Summary ---")
    print(f"Backend: {provider_context.backend}")
    print(f"Model: {provider_context.model}")
    print(f"Total Audio Duration: {format_duration(total_audio_seconds)} ({total_audio_seconds/60.0:.1f} minutes)")
    print(f"Total Wall-clock Time: {format_duration(wall_clock_time)}")
    print(f"Average Processing Speed: {avg_speed:.1f} audio-min / wall-min")
    if is_nemo:
        print(f"Total Chunks: {total_chunks}")
        print(f"Failed Chunks: {len(failed_chunks)}")
        print(f"Retried Chunks: {len(retried_chunks)}")
        print(f"CUDA OOM Events: {len(oom_events)}")
        print(f"Boundary Deduplications: {len(DEDUPLICATION_DECISIONS)}")
    print("-----------------------------\n")

    print("Done.")
    print(
        f"Wrote {len(cleaned_turns)} turns, {len(all_words)} words, "
        f"{len(chunks)} chunks, {len(glossary_candidates)} glossary candidates "
        f"(and {len(noise_candidates)} noise candidates filtered)."
    )
    
    comp_summary = completeness_report["completeness_summary"]
    for w in comp_summary["warnings"]:
        print(f"WARNING: {w}", file=sys.stderr)
        
    if manifest.completeness.fail_on_missing_coverage and comp_summary["status"] == "incomplete":
        print("ERROR: Transcription completeness check failed. Exiting.", file=sys.stderr)
        return 1
        
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        raise SystemExit(130)
