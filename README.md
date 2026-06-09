# Summary Transcriber v2

Local-first, completeness-priority transcription pipeline for long-form separate speaker audio tracks (e.g., D&D session audio recorded via Craig). 

The pipeline prioritizes **complete transcript retention** over aggressive cleanup, ensuring no character monologues, rules calls, or DM narrations are lost.

---

## Key Features

1. **Completeness-First Philosophy**: Overlap deduplication is disabled by default. If enabled, any discarded duplicate text is saved to an audit file (`deduplication_audit.jsonl`).
2. **Local-Only Transcription**: Supporting `faster-whisper`, `whisperx`, `parakeet`, and `canary` without external APIs.
3. **Structured Agent Outputs**: Generates a set of canonical JSON/JSONL outputs designed for consumption by downstream Recap or Summary AI agents.
4. **Human-Readable Transcript**: Generates a cleaned, filler-reduced Markdown transcript at `output/transcript.md`.
5. **Coverage Validation**: Automatically computes session coverage per speaker and flags missing ranges or failed chunks in `output/transcript_completeness_report.json`.

---

## Repository Layout

```text
summary-transcriber/
├── transcribe.py
└── src/
    ├── config/
    │   └── manifest.py
    ├── audio/
    │   ├── inspect.py
    │   ├── prepare.py
    │   └── chunk.py
    ├── providers/
    │   ├── base.py
    │   ├── parakeet.py
    │   ├── canary.py
    │   └── whisperx.py
    ├── normalize/
    │   ├── turns.py
    │   ├── words.py
    │   └── merge.py
    ├── enrich/
    │   ├── cleanup.py       # Conservative filler word remover
    │   ├── completeness.py  # Transcript coverage validator
    │   ├── glossary_candidates.py
    │   ├── corrections.py
    │   ├── entities.py
    │   ├── quality.py
    │   └── chunks.py
    └── exports/
        ├── agents.py        # Required agent outputs
        ├── markdown.py      # Human-friendly markdown exports
        ├── vtt.py
        └── raw.py
```

---

## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

Ensure `ffmpeg` and `ffprobe` are installed and available on your system `PATH`.

---

## Supported Local Backends

Models are fetched and cached locally by the backend the first time they are loaded.

### 1. default: `faster-whisper`
Recommended default backend. Stable and highly reliable.

### 2. optional: `whisperx`
Provides optional Whisper-family transcription.

### 3. optional: `parakeet` (`nvidia/parakeet-tdt-0.6b-v3`)
Preferred NVIDIA comparison backend. Requires GPU/CUDA and NVIDIA NeMo libraries.

### 4. optional: `canary` (`nvidia/canary-1b-v2`)
Optional NVIDIA comparison backend.

For PyTorch CUDA & NeMo installations (required for Parakeet/Canary), refer to the GPU Setup guidelines.

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
  enabled: false             # Disable destructive overlap deduplication (default)

completeness:
  max_missing_seconds_warn: 30.0   # Warn if missing coverage exceeds 30s
  fail_on_missing_coverage: false  # Exit with code 1 if missing coverage exceeds threshold
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
*Whisper-based backends bypass this audio splitting step entirely.*

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
Runs the audio downmixing, chunking, and local model transcription (ASR). Resolves turns chronologically, builds proper noun candidate lists, and writes initial draft records under `output/draft/`.

### 2. Finalize Mode
```bash
python transcribe.py --manifest session_manifest.yaml --mode finalize
```
Loads draft outputs, applies glossary corrections, extracts entities, and writes final agent outputs and human-readable Markdown transcripts. **Does not require a GPU** and runs in seconds.

---

## Required Outputs

After running `--mode finalize`, the output directory includes:

### Top-Level Exports

#### `output/transcript.md`
Human-readable, chronological session transcript.
- Uses HH:MM:SS timestamps and speaker labels (e.g. `[00:01:16 - 00:01:34] Eldrin / Player:`).
- Displays cleaned text with conservative filler-words removed.
- *Removes pure fillers (um, uh, ah, er, hmm, mm-hmm, you know) and accidental double starts (I I, I'm I'm) without summarizing, paraphrasing, or altering dialogue meaning.*

#### `output/transcript_completeness_report.json`
Details of session-wide transcription coverage.
- Lists covered ranges, missing ranges, expected/successful/failed/empty chunk counts per speaker, and total missing seconds.

---

### Agent Structured Exports (`output/agents/`)

#### `output/agents/turns.jsonl`
The canonical chronological transcript. One JSON object per turn containing:
- Authoritative session, speaker, and character metadata.
- Timestamps and source audio metrics.
- `text.raw`: the untouched original output.
- `text.cleaned`: glossary corrected text.
- `text.markdown_cleaned`: filler-reduced text.

#### `output/agents/session.json`
Overall session and transcription metadata, including campaign details, model details, outputs manifest, and completeness reports (failed chunks and missing coverage ranges).

#### `output/agents/transcript_entities.jsonl`
One JSON object per glossary-backed entity found in the transcript, listing its description, type, first seen timestamp/turn, and a list of evidence turn IDs.

#### `output/agents/chunks.jsonl`
Index of 30-minute transcript chunk files.

#### `output/agents/chunks/chunk_001.jsonl`, `chunk_002.jsonl`, ...
Convenience JSONL files containing turns for each specific 30-minute transcript window.Deterministic views regenerated from `turns.jsonl`.

---

## License

MIT
