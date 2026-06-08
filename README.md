# Summary Transcriber

Transcribe multiple speaker audio tracks with OpenAI Whisper, interactively label speakers, merge every line into a single chronologically ordered transcript, and generate agent-ready files for creating a session summary.

## Background

This tool was developed to help with a Dungeons & Dragons campaign. The group consists of working adults with families, which means there can be long breaks between sessions. To help everyone get back up to speed, this script transcribes game sessions and produces source material for a "read-ahead" recap before the next session.

The audio is captured using the [craig.chat](https://craig.chat/) bot for Discord, but Craig is beyond the scope of this project. You can use any audio recording software that produces separate audio tracks for each speaker.

## Features

- Interactive folder prompt and model selection (Whisper sizes: tiny, base, small, medium, large, turbo)
- Per-speaker track handling: record each participant separately, then label before transcription
- Merges all `.vtt` files into one unified, time-sorted CSV with start and end timestamps
- Deduplicates consecutive identical lines from the same speaker file
- Optional chunking of the merged CSV into fixed-duration segments (default 30-minute slices)
- Agent-ready exports: JSONL transcript, summary input brief, and Markdown transcript chunks
- Friendly error handling for missing Whisper CLI, missing audio files, and naming collisions

## Workflow Overview

1. Supply a directory containing one audio file per speaker (`.flac`, `.mp3`, `.wav`, `.m4a`, `.ogg`).
2. The script creates `transcript/` and `merged/` inside that directory as needed.
3. You assign a human-readable speaker name for each audio track before transcription starts.
4. Each audio file is transcribed with the chosen Whisper model; the resulting `.vtt` is moved into `merged/` using the speaker name you chose.
5. The renamed VTTs are parsed and merged into `merged/session_transcript.csv`.
6. The merged CSV is split into `merged/chunked/chunk_#.csv` files.
7. Agent-oriented files are written to `merged/agent/` for summary generation.

## Requirements

| Component | Purpose |
| --- | --- |
| Python 3.8+ | Run the orchestration script |
| FFmpeg | Required by Whisper for audio decoding |
| Whisper CLI (`openai-whisper`) | Performs transcription |
| Optional CUDA GPU + PyTorch | Performance boost for larger models |

Installable Python dependency tracked in `requirements.txt`:

```text
openai-whisper
```

## Quick Start

```bash
git clone https://github.com/Gitchegumi/summary-transcriber.git
cd summary-transcriber
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Ensure ffmpeg is installed and in PATH
python transcribe.py
```

When prompted:

- Enter the absolute or relative path to your audio folder.
- Choose a model size, or press Enter for default `turbo`.
- Provide friendly speaker names for each audio track before transcription begins.

## Model Selection Notes

| Model | Speed | Accuracy | Typical Use |
| --- | --- | --- | --- |
| tiny | Fastest | Lowest | Quick skim or draft |
| base | Fast | Low-mid | Casual notes |
| small | Medium | Medium | General session recaps |
| medium | Slower | High | More accurate logs |
| large | Slowest | Highest | Best quality, longest sessions |
| turbo | Fast optimized | High | Balanced default |

Larger models are slower and require more VRAM; a GPU is highly recommended above `small`.

## Installing Prerequisites

### FFmpeg

- Windows with Chocolatey: `choco install ffmpeg`
- macOS with Homebrew: `brew install ffmpeg`
- Debian/Ubuntu: `sudo apt update && sudo apt install ffmpeg`

### Whisper CLI

```bash
pip install openai-whisper
```

If you need GPU acceleration, install PyTorch first using the selector at the [official PyTorch site](https://pytorch.org/get-started/locally/), then install Whisper.

## Output Structure

After a run, your audio directory will contain:

```text
transcript/
    # temporary Whisper output folder; VTTs are moved after each track finishes
merged/
    Alice.vtt
    Bob.vtt
    session_transcript.csv
    chunked/
        chunk_1.csv
        chunk_2.csv
        ...
    agent/
        session_summary_input.md
        session_transcript.jsonl
        chunks/
            chunk_001.md
            chunk_002.md
            ...
```

`session_transcript.csv` columns:

```text
Start,End,Speaker,Text
00:00:12.345,00:00:14.000,Alice,Hello everyone...
00:00:14.101,00:00:15.200,Bob,Hi!
```

## Agent Summary Output

The `merged/agent/` folder is designed for use by an AI agent that needs to create a session summary without re-parsing CSV:

- `session_summary_input.md` contains summary instructions, session metadata, speaker activity, and a chunk index.
- `session_transcript.jsonl` stores one chronological transcript turn per line with `index`, `start`, `end`, `speaker`, and `text`.
- `chunks/chunk_###.md` stores time-bounded transcript slices with speaker and timestamp labels.

Recommended summary workflow:

1. Give the agent `session_summary_input.md`.
2. Have it process each `chunks/chunk_###.md` in order and produce brief chunk notes.
3. Ask it to combine chunk notes into a final recap with sections for major events, decisions, NPCs/locations/items, unresolved hooks, and next-session reminders.

## Current Transcription Backend Options

The script currently uses local `openai-whisper`, which is still a good no-API-key baseline when you already have separate speaker tracks from Craig. Recent tools worth considering:

| Option | Why consider it | Tradeoff |
| --- | --- | --- |
| OpenAI `gpt-4o-transcribe` / `gpt-4o-mini-transcribe` | Newer API speech-to-text models with better accuracy than classic Whisper in OpenAI's published benchmarks | Cloud API cost; separate-speaker local workflow would need an API integration |
| OpenAI `gpt-4o-transcribe-diarize` | Built-in diarization for mixed-speaker audio | API-only; model availability and diarization behavior should be tested on campaign audio |
| Deepgram Nova-3 + `diarize_model=latest` | Strong managed batch transcription, custom terminology/keyterm support, and a newer diarization v2 option | Cloud API; best fit if you want automatic diarization or vocabulary prompting |
| ElevenLabs Scribe / Scribe v2 | Structured JSON, speaker diarization, word timestamps, and non-speech event markers | Cloud API; verify long-session limits and diarization quality on your recordings |
| Soniox v4 Async | Long-form multilingual transcription with improved speaker separation and normalization | Cloud API; less familiar ecosystem than Whisper |
| WhisperX | Local/open-source path for faster Whisper, word timestamps, VAD, and diarization via pyannote | More setup complexity, possible Hugging Face token/model access, GPU recommended |
| Groq Whisper Large v3 Turbo | Very fast hosted Whisper inference at low per-hour pricing | No native diarization; best as a speed upgrade rather than an output-structure upgrade |

For this specific tool, the most practical upgrade path is:

1. Keep local Whisper CLI as the default backend.
2. Add optional provider backends later (`openai`, `deepgram`, `elevenlabs`, `soniox`, `groq`, `whisperx`) behind the same merged transcript schema.
3. Add a campaign glossary/custom vocabulary prompt where providers support it, because D&D names and fantasy terms are often the biggest transcription failure point.
4. Preserve the agent output format regardless of backend, so the summarization step stays stable.

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `Error: 'whisper' command not found.` | Whisper CLI not installed or wrong venv | Activate venv, then `pip install -r requirements.txt` |
| `No audio files found` | Wrong folder path or unsupported extensions | Check path; ensure files end in supported extensions |
| Very slow transcription | Using large model on CPU | Switch to a smaller model or install CUDA + correct PyTorch build |
| GPU not used | PyTorch CPU-only build installed | Reinstall PyTorch with CUDA per PyTorch guidance |
| VTT rename collision | Same speaker name chosen twice | Provide unique names when prompted |

## Roadmap / Ideas

- Optional API backends for OpenAI, Deepgram, ElevenLabs, Soniox, Groq, and WhisperX
- Campaign glossary / vocabulary prompting
- Automated chunk-level summary drafts
- Final Markdown session recap export

## License

MIT
