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
├── transcribe.py
└── src/
    ├── config/
    │   └── manifest.py
    ├── audio/
    │   ├── inspect.py
    │   ├── prepare.py
    │   └── clip.py
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
    │   ├── glossary.py
    │   ├── glossary_candidates.py
    │   ├── corrections.py
    │   ├── entities.py
    │   ├── quality.py
    │   └── chunks.py
    └── exports/
        ├── nocodb.py
        ├── glossary.py
        ├── agents.py
        ├── markdown.py
        ├── vtt.py
        └── raw.py
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

Keep this virtual environment active for every backend install command below. If you open a new terminal later, activate `.venv` again before running any `pip install ...` command.

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

The CLI defaults to `device: cuda`. Use `device: cpu` only for short tests or when GPU acceleration is unavailable. NVIDIA's current NeMo model documentation is Linux-centered; on Windows, WSL2 with NVIDIA CUDA support is usually the least surprising route for the NVIDIA backends.

### Install CUDA-enabled PyTorch

Use a Python version that has matching CUDA PyTorch wheels and is supported by the NVIDIA stack. Python 3.12 is the safest Windows choice for this project right now. Avoid Python 3.13 for the NVIDIA backend unless the PyTorch selector and NeMo install both explicitly support it for your chosen CUDA build.

If you need a fresh NVIDIA environment, create it with Python 3.12, then activate it:

```bash
python3.12 -m venv .nvidia-venv
.\.nvidia-venv\Scripts\Activate.ps1  # Windows
# source .nvidia-venv/bin/activate   # macOS/Linux
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

If you already installed CPU-only PyTorch, or mixed CPU/CUDA package variants, uninstall all three PyTorch packages first:

```bash
.\.nvidia-venv\Scripts\Activate.ps1  # Windows, if not already active
# source .nvidia-venv/bin/activate   # macOS/Linux, if not already active
pip uninstall -y torch torchvision torchaudio
```

Then use the official PyTorch selector:

1. Open <https://pytorch.org/get-started/locally/>.
2. Choose `Stable`.
3. Choose your OS.
4. Choose `Pip`.
5. Choose `Python`.
6. Choose a CUDA compute platform. The current selector lists CUDA 11.8, 12.6, and 12.8; choose the newest CUDA option supported by your NVIDIA driver.
7. Run the generated `pip install ... --index-url https://download.pytorch.org/whl/cu...` command inside the active venv.

If `nvidia-smi` shows a newer CUDA UMD version than the PyTorch selector offers, that is okay. For example, a driver reporting CUDA 13.x can run PyTorch CUDA 12.6 or 12.8 wheels; choose one of the CUDA builds that PyTorch actually publishes for your Python version.

Example for CUDA 12.6:

> NOTE: You can check the CUDA version supported by your NVIDIA driver with `nvidia-smi` in a terminal. If your driver only supports an older CUDA version, install the corresponding PyTorch build instead of the newest one. Use the [PyTorch Stable selector](https://pytorch.org/get-started/locally/) to generate the correct command.

```bash
.\.nvidia-venv\Scripts\Activate.ps1  # Windows, if not already active
# source .nvidia-venv/bin/activate   # macOS/Linux, if not already active
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Verify that PyTorch can see CUDA before installing NeMo:

```bash
python -c "import torch; print(torch.__version__); print('cuda available:', torch.cuda.is_available()); print('cuda version:', torch.version.cuda); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

If `cuda available` prints `False`, Parakeet will fail when the manifest uses `device: cuda`. Fix PyTorch/CUDA first, or temporarily set `device: cpu` in the manifest.

If you see `RuntimeError: operator torchvision::nms does not exist`, the venv probably has mismatched packages, such as CPU-only `torch` with CUDA `torchvision`. Remove all three packages and reinstall them from the same PyTorch selector command:

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; import torchvision; import torchaudio; print(torch.__version__); print(torchvision.__version__); print(torchaudio.__version__); print('cuda available:', torch.cuda.is_available())"
```

## Backend Installation

The script does not install transcription backends automatically. Install the backend package first, then run `transcribe.py`. Model checkpoints are fetched and cached locally by the backend the first time that backend loads a model.

### Primary: Parakeet TDT 0.6B v3

Parakeet is the intended primary model for this project. Use it for normal D&D session transcription unless you have a specific reason to compare or fall back.

```yaml
transcription:
  backend: parakeet
  model: nvidia/parakeet-tdt-0.6b-v3
  device: cuda
```

After CUDA-enabled PyTorch is installed and verified, install NVIDIA NeMo ASR:

```bash
.\.venv\Scripts\Activate.ps1  # Windows, if not already active
# source .venv/bin/activate   # macOS/Linux, if not already active
pip install -U "nemo_toolkit[asr]"
```

Quick local check:

```bash
python -c "import nemo.collections.asr as nemo_asr; nemo_asr.models.ASRModel.from_pretrained('nvidia/parakeet-tdt-0.6b-v3'); print('parakeet ready')"
```

Then run:

```bash
python transcribe.py --manifest session_manifest.yaml
```

### Optional Comparison: Canary 1B v2

Canary is optional. Use it when you want a local comparison run or benchmark against Parakeet. It uses the same NVIDIA NeMo ASR install as Parakeet.

```yaml
transcription:
  backend: canary
  model: nvidia/canary-1b-v2
  device: cuda
```

If you did not already install NeMo for Parakeet, install it after CUDA-enabled PyTorch is installed and verified:

```bash
.\.venv\Scripts\Activate.ps1  # Windows, if not already active
# source .venv/bin/activate   # macOS/Linux, if not already active
pip install -U "nemo_toolkit[asr]"
```

Quick local check:

```bash
python -c "import nemo.collections.asr as nemo_asr; nemo_asr.models.ASRModel.from_pretrained('nvidia/canary-1b-v2'); print('canary ready')"
```

Run Canary as an override:

```bash
python transcribe.py --manifest session_manifest.yaml --backend canary --model nvidia/canary-1b-v2
```

### Optional Fallback: WhisperX

WhisperX is optional. Use it when the NVIDIA NeMo stack is not available or when you want a local Whisper-family fallback.

```bash
.\.venv\Scripts\Activate.ps1  # Windows, if not already active
# source .venv/bin/activate   # macOS/Linux, if not already active
pip install whisperx
```

Run:

```bash
python transcribe.py --manifest session_manifest.yaml --backend whisperx --model large-v3
```

Do not enable WhisperX diarization for Craig speaker-track runs; the manifest already supplies authoritative speaker identity. Diarization can require Hugging Face access and model agreements, which this project intentionally avoids.

### Optional Fallback: faster-whisper

The `whisperx` provider also falls back to faster-whisper if `whisperx` is not installed but `faster-whisper` is available.

```bash
.\.venv\Scripts\Activate.ps1  # Windows, if not already active
# source .venv/bin/activate   # macOS/Linux, if not already active
pip install faster-whisper
```

Run:

```bash
python transcribe.py --manifest session_manifest.yaml --backend faster-whisper --model large-v3
```

For GPU acceleration, faster-whisper also needs compatible NVIDIA cuBLAS/cuDNN libraries available to CTranslate2. CPU runs can use `device: cpu`, but long sessions will be slower.

## Backend Priority

Use the backends in this order:

1. Parakeet TDT 0.6B v3: primary, intended default.
2. Canary 1B v2: optional comparison or benchmark backend.
3. WhisperX: optional local fallback.
4. faster-whisper: optional local fallback when WhisperX is not installed.

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

The pipeline sends manifest audio files directly to the transcription provider by default. Craig `.flac` files do not need to be remuxed to WAV first.

Use `--normalize-audio` only if a backend has trouble reading the source file or you explicitly want local mono 16 kHz WAV files in `output/prepared_audio/`:

```bash
python transcribe.py --manifest session_manifest.yaml --normalize-audio
```

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
  pcs:
    - "Thava"
    - "Brother Alden"
  npcs:
    - "Volo"
    - "Laeral Silverhand"
  locations:
    - "Waterdeep"
    - "Yawning Portal"
  factions:
    - "Harpers"
    - "Zhentarim"
  items:
    - "Stone of Golorr"
    - "Bag of Holding"
  spells:
    - "Counterspell"
    - "Misty Step"
  rules_terms:
    - "Dex save"
    - "Insight check"
  custom_terms:
    - "Session zero"
    - "The Black Door"
```

The glossary values above are examples only. Replace them with the actual player characters, NPCs, places, factions, items, spells, rules phrases, and table-specific terms from your campaign before running transcription.

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
python transcribe.py --manifest session_manifest.yaml --normalize-audio
```

`--skip-transcription` reuses raw provider JSON from `output/raw/<backend>/<speaker_id>.raw.json`, which is helpful while iterating on normalization and exports.

If `--output` is omitted, outputs are written to an `output/` folder beside the manifest file. For example, `python transcribe.py --manifest campaigns/session_12/session_manifest.yaml` writes to `campaigns/session_12/output/`.

## Pipeline

1. Load `session_manifest.yaml`.
2. Validate all listed audio files exist.
3. Inspect audio duration, sample rate, channel count, format, and codec with `ffprobe`.
4. Send source audio directly to the provider, or normalize audio locally to mono 16 kHz WAV with `ffmpeg` when `--normalize-audio` is used.
5. Transcribe each known speaker track independently.
6. Preserve raw local model output for debugging.
7. Normalize provider output into canonical word and turn records.
8. Merge speaker tracks chronologically.
9. Run conservative glossary and rules-based cleanup without overwriting raw text.
10. Generate glossary candidates for future manifest updates.
11. Generate a transcript quality report.
12. Generate time-based chunks.
13. Export NocoDB-ready CSV files.
14. Export agent-ready JSONL and chunk JSON.
15. Export human-readable Markdown and optional VTT.

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
  glossary_candidates.csv
  glossary_candidates.json

  agents/
    turns.jsonl
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

`glossary_candidates.csv`:

```text
session_id,candidate_id,term,candidate_type,suggested_glossary_bucket,reason,example_text,speaker_id,turn_id,start_seconds,end_seconds,confidence,occurrence_count
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

## Glossary Candidates

After each run, review `output/glossary_candidates.csv` or `output/glossary_candidates.json` for possible terms to add to future manifests. These files are suggestions only; the pipeline does not automatically edit `session_manifest.yaml`.

Candidate entries are generated from repeated low-confidence words, unfamiliar capitalized terms, and phrases that triggered conservative correction rules. Use the `suggested_glossary_bucket` value as a starting point, then move terms into the bucket that fits your campaign.

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

Use `output/agents/turns.jsonl` as the canonical chronological input for recap agents. Each JSONL record includes session metadata, stable turn ID, speaker metadata, character metadata, timestamps, raw text, cleaned text, confidence metadata, correction candidates, and tags.

Use `output/agents/chunks/chunk_###.json` for bounded summarization passes, then combine chunk notes into a final recap. `output/agents/session_summary_input.md` provides a compact index for agent orchestration.

## Notes

- This project is local-first by design.
- Speaker identity comes from the manifest.
- VTT and Markdown are exports, not the data model.
- Hosted transcription services and API-key workflows are out of scope for v2.

## License

MIT
