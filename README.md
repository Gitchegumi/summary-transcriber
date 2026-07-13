# Summary Transcriber v2

Local-first, completeness-priority transcription pipeline for long-form separate speaker audio tracks (e.g., D&D session audio recorded via Craig).

The pipeline prioritizes **complete transcript retention** over aggressive cleanup, ensuring no character monologues, rules calls, or DM narrations are lost.

---

## Key Features

1. **Completeness-First Philosophy**: Overlap deduplication is disabled by default. If enabled, any discarded duplicate text is saved to an audit file (`deduplication_audit.jsonl`).
2. **Local-Only Transcription**: Supports `faster-whisper`, `whisperx`, `parakeet`, and `canary` without external APIs.
3. **Structured Agent Outputs**: Generates a set of canonical JSON/JSONL outputs designed for consumption by downstream Recap or Summary AI agents.
4. **Human-Readable Transcript**: Generates a cleaned, filler-reduced Markdown transcript at `output/transcript.md`.
5. **Coverage Validation**: Automatically computes session coverage per speaker and flags missing ranges or failed chunks in `output/transcript_completeness_report.json`.

---

## Repository Layout

```text
summary-transcriber/
|-- transcribe.py             # CLI entry point
|-- requirements.txt         # Default faster-whisper dependencies
|-- src/
|   |-- audio/               # Inspection, preparation, and chunking
|   |-- config/              # Manifest loading and validation
|   |-- enrich/              # Corrections, entities, quality, completeness
|   |-- exports/             # CSV, JSON/JSONL, Markdown, VTT, and raw exports
|   |-- normalize/           # Provider output normalization and turn merging
|   |-- progress/            # Console progress reporting
|   `-- providers/           # faster-whisper, WhisperX, Parakeet, and Canary
`-- tests/
```

---

## Installation

Python 3.12 is the version tracked by this repository. Create a virtual environment and install the default `faster-whisper` dependencies:

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

Ensure `ffmpeg` and `ffprobe` are installed and available on your system `PATH`.

The other backends require optional packages that are intentionally not included in `requirements.txt`:

```bash
# WhisperX
pip install whisperx

# Parakeet or Canary: install a CUDA-compatible PyTorch build first,
# then install NVIDIA NeMo ASR.
pip install 'nemo_toolkit[asr]'
```

For NeMo, use the PyTorch installation command appropriate for the CUDA version installed on the machine.

---

## Supported Local Backends

Models are fetched and cached locally by the backend the first time they are loaded.

### 1. Default: `faster-whisper`

Recommended default backend. Stable and highly reliable.

### 2. Optional: `whisperx`

Provides optional Whisper-family transcription.

### 3. Optional: `parakeet` (`nvidia/parakeet-tdt-0.6b-v3`)

Preferred NVIDIA comparison backend. Requires GPU/CUDA and NVIDIA NeMo libraries.

### 4. Optional: `canary` (`nvidia/canary-1b-v2`)

Optional NVIDIA comparison backend.

---

## Manifest Configuration

The pipeline is driven by a `session_manifest.yaml` file that specifies metadata, backend configurations, speaker tracks, and glossary files:

```yaml
session:
  id: "james-session-9"
  campaign: "James Campaign"
  session_number: 9
  session_date: "2026-06-08"
  source: "craig"

audio:
  sample_rate: 16000
  channels: 1
  chunk_seconds: 60
  overlap_seconds: 2
  min_chunk_seconds: 15
  chunk_format: "wav"

transcription:
  backend: "faster-whisper"
  model: "large-v3"
  device: "cuda"
  batch_size: 0 # 0 = batch one current chunk from every speaker track

progress:
  enabled: true
  update_interval_seconds: 5

speakers:
  - speaker_id: "dm"
    display_name: "James"
    role: "DM"
    character_name: null
    file: "audio/dm.flac"
  - speaker_id: "kibiw"
    display_name: "Player 1"
    role: "Player"
    character_name: "Kibiw"
    file: "audio/player_1.flac"

glossary_sources:
  - "glossary.yaml"

# Optional pipeline behaviors:
deduplication:
  enabled: false # Disable destructive overlap deduplication (default)

completeness:
  max_missing_seconds_warn: 500.0 # Warn if missing coverage exceeds 500s
  fail_on_missing_coverage: false # Exit with code 1 if missing coverage exceeds threshold
```

---

## Glossary Setup

Create a `glossary.yaml` to specify correct spellings, aliases, and metadata for D&D proper nouns:

```yaml
glossary:
  pcs:
    - canonical: "Kibiw"
      type: "pc"
      aliases:
        - "Kibew"
        - "Kibu"
  npcs:
    - canonical: "Lady Saris"
      type: "npc"
      description: "Local ruling noble."
```

---

## Chunking Behaviors

### 1. NeMo Audio Chunking

NVIDIA NeMo models (Parakeet/Canary) cannot ingest hours of raw audio without out-of-memory errors. The pipeline automatically splits long speaker tracks into **short audio chunks** (default: 60 seconds with 2 seconds overlap) before transcribing.
_Whisper-based backends bypass this audio splitting step entirely._

Parakeet and Canary transcribe speaker tracks in round-robin batches through one shared model. With `batch_size: 0` (the default), a six-speaker manifest sends six chunks to each NeMo inference call. Set a positive value to cap simultaneous tracks when GPU memory is limited. If a batch fails, the pipeline records `batch_fallback_events` in raw output and retries each chunk through the existing 60s -> 30s -> 15s OOM recovery path.

### 2. Agent Transcript Chunking

After chronological merging, the pipeline groups final turns into approximately **30-minute transcript chunks**.

- Chunk boundaries are time-based but **never split a turn** (turns are assigned to a chunk using their start timestamp).
- Per-chunk files are derived views over the canonical turns, not independently transcribed or separately cleaned.

---

## Pipeline Modes

### 1. Draft Mode

```bash
python transcribe.py --manifest session_manifest.yaml --mode draft
```

Inspects and prepares audio as needed, chunks tracks for NeMo backends, and runs local model transcription (ASR). It resolves turns chronologically, builds proper noun candidate lists, and writes initial draft records under `output/draft/`.

Draft mode also writes raw provider results, canonical CSV/JSON exports, `transcript.md`, `transcript_completeness_report.json`, and `vtt/merged.vtt`. Candidate review uses these files under `output/draft/`:

- `transcript_turns.draft.csv` and `transcript_words.draft.csv`: inputs consumed by finalize mode.
- `unknown_terms_report.csv`: detailed candidates and scoring evidence for review.
- `candidate_glossary.yaml`: import-ready glossary entries.
- `noise_report.csv`: filtered candidate terms, when noise reporting is enabled.

### 2. Finalize Mode

```bash
python transcribe.py --manifest session_manifest.yaml --mode finalize
```

Review or merge desired candidates into a glossary listed by `glossary_sources`, then run finalize mode. It loads the draft CSV files, applies glossary corrections, extracts entities, and writes final agent outputs plus the refreshed canonical and human-readable exports. Finalize mode does not load the ASR model and therefore does not require a GPU.

By default, outputs are written to an `output/` directory beside the manifest. Use `--output PATH` to choose another location. Other useful draft-mode options include:

- `--skip-transcription` to rebuild draft outputs from existing provider JSON in `output/raw/`.
- `--resume` to reuse successfully transcribed NeMo chunks.
- `--deduplicate` to enable overlap deduplication and write `deduplication_audit.jsonl`.
- `--normalize-audio` to prepare mono 16 kHz WAV input before transcription.
- `--chunk-minutes N` to change the agent transcript window size from 30 minutes.

---

## Required Outputs

After running `--mode finalize`, the output directory includes:

### Canonical and Interchange Exports

- `session.json` and `speakers.csv`: session, model, speaker, and source-audio metadata.
- `transcript_turns.csv` and `transcript_words.csv`: canonical turn- and word-level records.
- `transcript_entities.csv` and `transcript_corrections.csv`: extracted entities and applied correction audit records.
- `transcript_chunks.csv`: the canonical transcript chunk index.
- `transcript_quality_report.json`: transcription and processing diagnostics.
- `vtt/merged.vtt`: merged WebVTT transcript.

### Top-Level Exports

#### `output/transcript.md`

Human-readable, chronological session transcript.

- Uses HH:MM:SS timestamps and speaker labels (e.g. `[00:01:16 - 00:01:34] Eldrin / Player:`).
- Displays cleaned text with conservative filler-words removed.
- _Removes pure fillers (um, uh, ah, er, hmm, mm-hmm, you know) and accidental double starts (I I, I'm I'm) without summarizing, paraphrasing, or altering dialogue meaning._

#### `output/transcript_completeness_report.json`

Details of session-wide transcription coverage.

- Measures processed audio intervals from successful chunks; silent/empty chunks count as successfully processed.
- Reports timestamp-derived speech and estimated silence separately from genuinely missing processing coverage.
- Lists covered ranges, missing ranges, expected/successful/failed/empty chunk counts per speaker, and total missing seconds.

---

### Agent Structured Exports (`output/agents/`)

#### `output/agents/turns.jsonl`

The compact chronological transcript intended for narrative-summary agents. Empty turns are omitted. Each JSON object contains only:

- `start_time`: the beginning of the turn as `HH:MM:SS.mmm`.
- `end_time`: the end of the turn as `HH:MM:SS.mmm`; overlapping ranges represent simultaneous speech.
- `speaker`: the player character name, `DM`, or display-name fallback.
- `text`: glossary-corrected transcription text.
- Consecutive segments from the same speaker are merged when separated by no more than 10 seconds.

Detailed source, model, quality, and raw-text fields remain available in the canonical NocoDB and raw exports.

#### `output/agents/session.json`

Overall session and transcription metadata, including campaign details, model details, outputs manifest, and completeness reports (failed chunks and missing coverage ranges).

#### `output/agents/transcript_entities.jsonl`

One JSON object per glossary-backed entity found in the transcript, listing its description, type, first seen timestamp/turn, and a list of evidence turn IDs.

#### `output/agents/chunks.jsonl`

Index of transcript chunk files (30-minute windows by default).

#### `output/agents/chunks/chunk_001.jsonl`, `chunk_002.jsonl`, ...

Convenience JSONL files containing turns for each transcript window. These are deterministic views regenerated from the canonical turns.

---

## License

MIT
