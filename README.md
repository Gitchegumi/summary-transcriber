# Summary Transcriber v2

Local-first transcription for long-form D&D sessions, designed around structured transcript data instead of subtitle files.

Version 2 is a breaking rewrite of the old Whisper/VTT workflow. The canonical outputs are JSON and CSV records that are ready for NocoDB imports and downstream recap agents. Markdown and VTT are secondary exports.

## What This Builds

- One manifest-driven CLI: `python transcribe.py --manifest session_manifest.yaml`
- Local-only transcription providers
- NVIDIA Parakeet TDT 0.6B v3 as the primary backend
- NVIDIA Canary 1B v2 as an optional local comparison backend
- WhisperX or faster-whisper as an optional local fallback backend
- Speaker identity from the manifest, not diarization, when Craig separate speaker tracks are available
- Canonical turn, word, correction, entity, chunk, speaker, session, and quality-report exports

No API keys are required. Audio is never uploaded. Hosted transcription services are intentionally not part of this project.

## Repository Layout

```text
summary-transcriber/
  transcribe.py
  src/
    config/manifest.py
    audio/inspect.py
    audio/prepare.py
    audio/clip.py
    providers/base.py
    providers/parakeet.py
    providers/canary.py
    providers/whisperx.py
    normalize/turns.py
    normalize/words.py
    normalize/merge.py
    enrich/glossary.py
    enrich/corrections.py
    enrich/entities.py
    enrich/quality.py
    enrich/chunks.py
    exports/nocodb.py
    exports/agents.py
    exports/markdown.py
    exports/vtt.py
    exports/raw.py
```

## Install

Create a virtual environment and install the lightweight orchestrator dependency:

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

Install FFmpeg and make sure `ffmpeg` and `ffprobe` are on your `PATH`.

```bash
# Windows with Chocolatey
choco install ffmpeg

# macOS with Homebrew
brew install ffmpeg

# Debian/Ubuntu
sudo apt update && sudo apt install ffmpeg
```

## GPU and CUDA

Parakeet and Canary are large local ASR models. For practical long-session use, run on an NVIDIA GPU with a CUDA-enabled PyTorch installation. Install PyTorch for your CUDA version from the official PyTorch selector, then install the local ASR package you want to use.

The CLI defaults to `device: cuda`. Use `device: cpu` only for short tests or when GPU acceleration is unavailable.

## Optional Local Backends

Install only the local backend packages you need.

### Parakeet

Use Parakeet as the default quality backend:

```yaml
transcription:
  backend: parakeet
  model: nvidia/parakeet-tdt-0.6b-v3
  device: cuda
```

This backend expects NVIDIA NeMo ASR to be installed locally.

### Canary Comparison

Use Canary for comparison or benchmarking:

```bash
python transcribe.py --manifest session_manifest.yaml --backend canary --model nvidia/canary-1b-v2
```

### WhisperX / faster-whisper Fallback

Use WhisperX or faster-whisper when the NVIDIA stack is unavailable:

```bash
python transcribe.py --manifest session_manifest.yaml --backend whisperx --model large-v3
```

The fallback is still local. Do not enable diarization for Craig speaker-track runs; the manifest already supplies authoritative speaker identity.

## Craig Speaker-Track Audio

Craig recordings work best when exported as one audio file per participant. Put those files near the manifest, for example:

```text
audio/
  mat.flac
  player_01.flac
  player_02.flac
session_manifest.yaml
```

Use stable `speaker_id` values because they become foreign keys in the exported CSV/JSON records. Use `display_name` for the human-readable speaker name and `character_name` for the D&D character when applicable.

## Session Manifest

```yaml
session:
  id: "campaign-session-id"
  campaign: "Campaign Name"
  session_number: 1
  session_date: "YYYY-MM-DD"
  source: "craig"

transcription:
  backend: parakeet
  model: nvidia/parakeet-tdt-0.6b-v3
  device: cuda

speakers:
  - speaker_id: "dm"
    display_name: "Mat"
    role: "DM"
    character_name: null
    file: "audio/mat.flac"

  - speaker_id: "player_01"
    display_name: "Player Name"
    role: "Player"
    character_name: "Character Name"
    file: "audio/player_01.flac"

glossary:
  pcs: []
  npcs: []
  locations: []
  factions: []
  items: []
  spells: []
  rules_terms: []
  custom_terms: []
```

## Run

```bash
python transcribe.py --manifest session_manifest.yaml
```

Useful overrides:

```bash
python transcribe.py --manifest session_manifest.yaml --output output
python transcribe.py --manifest session_manifest.yaml --backend canary --model nvidia/canary-1b-v2
python transcribe.py --manifest session_manifest.yaml --backend whisperx --model large-v3
python transcribe.py --manifest session_manifest.yaml --skip-transcription
```

`--skip-transcription` reuses raw provider JSON from `output/raw/<backend>/<speaker_id>.raw.json`, which is helpful while iterating on normalization and exports.

## Pipeline

1. Load `session_manifest.yaml`.
2. Validate all listed audio files exist.
3. Inspect audio duration, sample rate, channel count, format, and codec with `ffprobe`.
4. Normalize audio locally to mono 16 kHz WAV with `ffmpeg`.
5. Transcribe each known speaker track independently.
6. Preserve raw local model output for debugging.
7. Normalize provider output into canonical word and turn records.
8. Merge speaker tracks chronologically.
9. Run conservative glossary and rules-based cleanup without overwriting raw text.
10. Generate a transcript quality report.
11. Generate time-based chunks.
12. Export NocoDB-ready CSV files.
13. Export agent-ready JSONL and chunk JSON.
14. Export human-readable Markdown and optional VTT.

## Output

```text
output/
  session.json
  speakers.csv
  transcript_turns.csv
  transcript_words.csv
  transcript_entities.csv
  transcript_corrections.csv
  transcript_chunks.csv
  transcript_quality_report.json

  agents/
    hermes_turns.jsonl
    chunks/
      chunk_001.json
      chunk_002.json
    session_summary_input.md

  markdown/
    transcript.md
    chunks/
      chunk_001.md
      chunk_002.md

  vtt/
    merged.vtt

  raw/
    parakeet/
      speaker_id.raw.json
```

## Canonical CSV Columns

`transcript_turns.csv`:

```text
session_id,turn_id,speaker_id,speaker_name,character_name,start_seconds,end_seconds,duration_seconds,text_raw,text_cleaned,confidence_avg,word_count,source_file,backend,model
```

`transcript_words.csv`:

```text
session_id,turn_id,word_id,speaker_id,word,start_seconds,end_seconds,confidence,is_low_confidence,backend,model
```

`transcript_corrections.csv`:

```text
session_id,turn_id,correction_id,original_text,corrected_text,correction_type,reason,confidence
```

## Conservative Cleanup

Raw text is always preserved in `text_raw`. `text_cleaned` only applies conservative rule and glossary corrections, such as:

```text
water deep -> Waterdeep
counter spell -> Counterspell
deck save -> Dex save
inside check -> Insight check
```

The pipeline does not freely rewrite transcript text with an LLM.

## Quality Report

`transcript_quality_report.json` flags:

- empty or near-empty speaker tracks
- unusual silence gaps
- very long segments
- repeated phrase loops
- impossible or overlapping timestamps
- low-confidence words
- likely glossary mismatches
- possible fantasy-name errors
- source audio duration mismatches

## NocoDB Import

Import the CSV files as separate NocoDB tables. Suggested table names:

- `sessions` from `session.json`
- `speakers` from `speakers.csv`
- `transcript_turns` from `transcript_turns.csv`
- `transcript_words` from `transcript_words.csv`
- `transcript_entities` from `transcript_entities.csv`
- `transcript_corrections` from `transcript_corrections.csv`
- `transcript_chunks` from `transcript_chunks.csv`

Use `session_id`, `speaker_id`, `turn_id`, `word_id`, and `chunk_id` as stable identifiers when creating relationships.

## Agent Consumption

Use `output/agents/hermes_turns.jsonl` as the canonical chronological input for Hermes or other recap agents. Each JSONL record includes session metadata, stable turn ID, speaker metadata, character metadata, timestamps, raw text, cleaned text, confidence metadata, correction candidates, and tags.

Use `output/agents/chunks/chunk_###.json` for bounded summarization passes, then combine chunk notes into a final recap. `output/agents/session_summary_input.md` provides a compact index for agent orchestration.

## Notes

- This project is local-first by design.
- Speaker identity comes from the manifest.
- VTT and Markdown are exports, not the data model.
- Hosted transcription services and API-key workflows are out of scope for v2.

## License

MIT
