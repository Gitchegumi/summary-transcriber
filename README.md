# Summary Transcriber v2

Local-first transcription for long-form D&D sessions, designed around structured transcript data instead of subtitle files.

Version 2 is a breaking rewrite of the old Whisper/VTT workflow. The canonical outputs are JSON and CSV records that are ready for NocoDB imports and downstream recap agents. Markdown and VTT are secondary exports.

## What This Builds

- One manifest-driven CLI: `python transcribe.py --manifest session_manifest.yaml`
- Local-only transcription providers
- faster-whisper as the recommended default backend
- WhisperX as an optional Whisper-family backend
- NVIDIA Parakeet TDT 0.6B v3 as the preferred NVIDIA backend
- NVIDIA Canary 1B v2 as an optional local comparison backend
- Speaker identity from the manifest, not diarization, when Craig separate speaker tracks are available
- Canonical turn, word, correction, entity, chunk, speaker, session, and quality-report exports
- A draft/finalize progressive glossary workflow

No API keys are required. Audio is never uploaded. Hosted transcription services are intentionally not part of this project.

## Repository Layout

```text
summary-transcriber/
├── transcribe.py
└── src/
    ├── config/
    │   ├── manifest.py
    │   └── glossary.py
    ├── audio/
    │   ├── inspect.py
    │   ├── prepare.py
    │   ├── clip.py
    │   └── chunk.py
    ├── providers/
    │   ├── base.py
    │   ├── parakeet.py
    │   ├── canary.py
    │   └── whisperx.py
    ├── progress/
    │   └── reporter.py
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

Create a virtual environment and install the default local transcription stack:

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` installs the orchestrator dependency plus faster-whisper, the default backend. Keep this virtual environment active for every backend install command below. If you open a new terminal later, activate `.venv` again before running any `pip install ...` command.

Install FFmpeg and make sure `ffmpeg` and `ffprobe` are on your `PATH`.

```bash
# Windows with Chocolatey
choco install ffmpeg

# macOS with Homebrew
brew install ffmpeg

# Debian/Ubuntu
sudo apt update && sudo apt install ffmpeg
```

## Backend Installation

The script does not install transcription backends automatically at runtime. Install the backend package first, then run `transcribe.py`. Model checkpoints are fetched and cached locally by the backend the first time that backend loads a model.

### Default: faster-whisper

faster-whisper is the recommended default for this tool because it is local, reliable on long Craig speaker tracks, and less fragile on Windows than the NVIDIA NeMo stack.

```yaml
transcription:
  backend: faster-whisper
  model: large-v3
  device: cuda
```

Install it through the project requirements:

```bash
.\.venv\Scripts\Activate.ps1  # Windows, if not already active
# source .venv/bin/activate   # macOS/Linux, if not already active
pip install -r requirements.txt
```

Run:

```bash
python transcribe.py --manifest session_manifest.yaml
```

For CPU-only fallback, set `device: cpu` in the manifest. Long sessions will be slower on CPU.

### Optional: WhisperX

WhisperX is optional. Use it when you specifically want WhisperX behavior instead of plain faster-whisper.

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

## Optional NVIDIA Backends

Parakeet and Canary remain available as optional local comparison backends. They are not the default path. Use them when your CUDA/NeMo environment is stable and you want to benchmark transcript quality against faster-whisper.

### GPU and CUDA Setup

Parakeet and Canary are large local ASR models. For practical long-session use, run on an NVIDIA GPU with a CUDA-enabled PyTorch installation. Install PyTorch for your CUDA version from the official PyTorch selector, then install the local ASR package you want to use.

The CLI defaults to `device: cuda`. Use `device: cpu` only for short tests or when GPU acceleration is unavailable. NVIDIA's current NeMo model documentation is Linux-centered; on Windows, WSL2 with NVIDIA CUDA support is usually the least surprising route for the NVIDIA backends.

#### Install CUDA-enabled PyTorch

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

### Preferred: Parakeet TDT 0.6B v3

Parakeet is the preferred NVIDIA model. Use it for comparison runs after the default faster-whisper workflow is working.

```yaml
transcription:
  backend: parakeet
  model: nvidia/parakeet-tdt-0.6b-v3
  device: cuda
```

After CUDA-enabled PyTorch is installed and verified, install NVIDIA NeMo ASR:

```bash
.\.nvidia-venv\Scripts\Activate.ps1  # Windows, if not already active
# source .nvidia-venv/bin/activate   # macOS/Linux, if not already active
pip install -U "nemo_toolkit[asr]"
```

Quick local check:

```bash
python -c "import nemo.collections.asr as nemo_asr; nemo_asr.models.ASRModel.from_pretrained('nvidia/parakeet-tdt-0.6b-v3'); print('parakeet ready')"
```

Run Parakeet as an override:

```bash
python transcribe.py --manifest session_manifest.yaml --backend parakeet --model nvidia/parakeet-tdt-0.6b-v3
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
.\.nvidia-venv\Scripts\Activate.ps1  # Windows, if not already active
# source .nvidia-venv/bin/activate   # macOS/Linux, if not already active
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

## Backend Priority

Use the backends in this order:

1. **faster-whisper**: recommended default.
2. **WhisperX**: optional Whisper-family backend.
3. **Parakeet TDT 0.6B v3**: preferred NVIDIA comparison backend.
4. **Canary 1B v2**: optional NVIDIA comparison or benchmark backend.

---

## Long-form Audio Chunking

NeMo models (Parakeet and Canary) are not designed to process massive 3+ hour audio tracks in a single transcription call, which can lead to catastrophic memory usage or timeouts. 

### Conditional Chunking Rule

- **Parakeet / Canary**: The pipeline automatically pre-splits long speaker audio files into shorter chunk files (default 60 seconds) with temporal overlaps (default 2 seconds) and processes them sequentially before reconstructing the speaker track.
- **WhisperX / faster-whisper**: Skip the chunking module entirely. Whisper-based backends use their own built-in long-form sliding window behaviors and should not have audio pre-split.

### Audio Chunker Features

1. **Audio Normalization**: During chunking, tracks are automatically downmixed to mono, resampled (default 16 kHz), and output as WAV or FLAC.
2. **Chunk Caching**: A file modified-timestamp (`mtime`) and size check is kept in `output/chunks/chunk_manifest.json`. Unchanged tracks are not re-split unless `--force-chunks` is provided.
3. **CUDA OOM Fallback Retry Sequence**: If a chunk fails with a CUDA out-of-memory error:
   - Clear CUDA GPU Cache.
   - Retry with smaller chunk size setting (first **30 seconds**, then **15 seconds** if needed).
   - If the chunk fails at 15 seconds, it is marked as failed in the quality report.
   - Continue processing if `continue_on_error` is true.
4. **Boundary Overlap Deduplication**: In the overlap window between adjacent chunks, text segments and word intervals are compared. The pipeline resolves duplicates by:
   - Selecting the segment with higher transcription confidence.
   - Falling back to the segment farther from the chunk boundary if confidence is identical/unavailable.
   - Deduplication decisions are appended to the quality report.

---

## Progress Reporting

Transcription progress is reported using actual processed duration units, rather than generic file count metrics (which are misleading for long sessions where individual files represent hours of audio).

### Progress Readout (NeMo backends)

Displays in the terminal:
- Current speaker track.
- Completed chunks / Total chunks for speaker.
- Processed speaker audio duration / Total speaker audio duration.
- Overall processed session audio duration / Total session audio duration.
- Elapsed time and estimated remaining time (ETA).
- Processing speed in `audio minutes / wall-clock minutes`.

Example:

```text
Transcribing campaign-session-018 with parakeet
Speaker: player_01 / Kibiw
Chunk: 42 / 186
Audio processed: 42.0 min / 186.0 min
Overall audio processed: 252.0 min / 1116.0 min
Elapsed: 00:38:12 | ETA: 02:10:45 | Speed: 6.6 audio-min / wall-min
```

To disable progress printing, run with `--no-progress`.

---

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

---

## Session Manifest Configuration

Manifest YAML files represent a single session run:

```yaml
session:
  id: "campaign-session-018"
  campaign: "Campaign Name"
  session_number: 18
  session_date: "2026-06-07"
  source: "craig"

audio:
  sample_rate: 16000
  channels: 1
  chunk_seconds: 60
  overlap_seconds: 2
  min_chunk_seconds: 15
  chunk_format: "wav"

transcription:
  backend: "parakeet"
  model: "nvidia/parakeet-tdt-0.6b-v3"
  device: "cuda"
  batch_size: 1
  precision: "float16"
  continue_on_error: true
  retry_on_oom: true
  oom_retry_chunk_seconds:
    - 30
    - 15

progress:
  enabled: true
  show_eta: true
  update_interval_seconds: 5

speakers:
  - speaker_id: "dm"
    display_name: "Mat"
    role: "DM"
    character_name: null
    file: "audio/mat.flac"

  - speaker_id: "player_01"
    display_name: "Player Name"
    role: "Player"
    character_name: "Kibiw"
    file: "audio/player_01.flac"

glossary_sources:
  - "../glossaries/campaign.yaml"
  - "../glossaries/characters.yaml"
  - "../glossaries/locations.yaml"
  - "../glossaries/rules.yaml"
  - "../glossaries/session_018.yaml"

workflow:
  mode: "draft"
  generate_candidate_glossary: true
```

---

## Glossary Design

Glossaries are decoupled from manifest session files. The glossary is optional; an empty glossary configuration is valid.

Glossary files support both simple string entries and structured entries:

```yaml
glossary:
  pcs:
    - canonical: "Kibiw"
      type: "pc"
      aliases:
        - "Kibew"
        - "Kibu"

  familiars:
    - canonical: "Batty"
      type: "familiar"
      description: "Kibiw's familiar."
      related_to:
        - entity: "Kibiw"
          relationship: "familiar_of"

  npcs:
    - canonical: "Lady Saris"
      type: "npc"
      aliases:
        - "Saris"
        - "Lady Sarahs"
      description: "NPC who may be referred to as either Saris or Lady Saris."
```

Legacy string lists are also supported:

```yaml
glossary:
  pcs:
    - "Kibiw"
    - "Eldrin"
```

Internally, both formats are normalized into structured records. Case variations and aliases are mapped directly to their canonical spellings.

---

## Progressive Glossary Workflow

You do not need to compile an exhaustive campaign glossary before transcribing.

### 1. Draft Mode

```bash
python transcribe.py --manifest session_018/manifest.yaml --mode draft
```

Runs the audio preparation, chunking, ASR transcription (or loads raw cached model results if `--skip-transcription` is passed), applies existing glossary rules, and exports:
- `output/draft/transcript_turns.draft.csv`
- `output/draft/transcript_words.draft.csv`
- `output/draft/candidate_glossary.yaml`
- `output/draft/unknown_terms_report.csv`

During draft runs, candidate proper nouns and repeated low-confidence terms are extracted. Evidence turn IDs and text contexts are bundled in `candidate_glossary.yaml` so you can easily review them.

### 2. Finalize Mode

```bash
python transcribe.py --manifest session_018/manifest.yaml --mode finalize
```

Loads the raw transcript data directly from the draft CSV files and applies the reviewed glossary. **This does not run model transcription or require a GPU**, completing in seconds. It updates and regenerates:
- NocoDB CSVs and Session JSON
- VTT and Markdown files
- Quality reports and agent files

---

## CLI Options

```bash
# Standard draft transcription
python transcribe.py --manifest session_manifest.yaml --mode draft

# Skip transcribing and reuse raw provider json (useful for testing corrections)
python transcribe.py --manifest session_manifest.yaml --mode draft --skip-transcription

# Force audio re-chunking for Parakeet/Canary runs
python transcribe.py --manifest session_manifest.yaml --mode draft --force-chunks

# Resume failed or missing chunks in Parakeet/Canary ASR runs
python transcribe.py --manifest session_manifest.yaml --mode draft --resume

# Disable status display
python transcribe.py --manifest session_manifest.yaml --mode draft --no-progress

# Run fast finalization after glossary updates
python transcribe.py --manifest session_manifest.yaml --mode finalize
```

---

## Outputs

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

  chunks/
    chunk_manifest.json
    player_01/
      player_01_chunk_00000.wav
      player_01_chunk_00001.wav

  raw/
    parakeet/
      player_01/
        player_01_chunk_00000.raw.json
        player_01_chunk_00001.raw.json

  draft/
    transcript_turns.draft.csv
    transcript_words.draft.csv
    candidate_glossary.yaml
    unknown_terms_report.csv

  agent/
    agent_turns.jsonl
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
```

---

## Database and Agent Integration

### NocoDB Import

Import the CSV files in `output/` directly into NocoDB as tables. Set up relationships using:
- `session_id` as session foreign key.
- `speaker_id` as speaker foreign key.
- `turn_id` as turn foreign key.
- `chunk_id` for time-based bounds.

### Agent Consumption

Recap agents (e.g. Hermes, OpenClaw) should consume:
- `output/agent/agent_turns.jsonl` (contains stable turn IDs, speaker/character metadata, raw and glossary-corrected texts, confidence scores, and low-confidence terms).
- `output/agent/chunks/chunk_###.json` for chunk-by-chunk summarization.
- `output/agent/session_summary_input.md` as an index page.

---

## License

MIT
