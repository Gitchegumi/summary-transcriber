# Summary Transcriber

Transcribe multiple speaker audio tracks with OpenAI Whisper, interactively label speakers, merge every line into a single chronologically ordered transcript (CSV + VTTs), and optionally split the long transcript into manageable time‑based chunks for easier review or campaign recap.

## Background

This tool was developed to help with a Dungeons & Dragons campaign. Our group consists of working adults with families, which means there can be long breaks between sessions. To help everyone get back up to speed, this script is used to transcribe our game sessions and generate a summary that can be sent out as a "read-ahead" before we meet.

The audio is captured using the [craig.chat](https://craig.chat/) bot for Discord, but the use of Craig is beyond the scope of this project. You are free to use any audio recording software that can produce separate audio tracks for each speaker.

## Features

- Interactive folder prompt and model selection (Whisper sizes: tiny, base, small, medium, large, turbo)
- Per‑speaker track handling – you record each participant separately, then label after transcription
- Merges all `.vtt` files into one unified, time‑sorted CSV (speaker + timestamp + text)
- Deduplicates consecutive identical lines coming from the same speaker file
- Optional chunking of the merged CSV into fixed-duration segments (default 30‑minute slices)
- Friendly error handling (missing Whisper CLI, no audio files found, naming collisions)

## Workflow Overview

1. You supply a directory containing one audio file per speaker (`.flac`, `.mp3`, `.wav`, `.m4a`, `.ogg`).
2. Script creates `transcript/` plus `transcript/merged/` inside that directory as needed.
3. Each audio file is transcribed with the chosen Whisper model -> individual `.vtt` files.
4. You are prompted to assign a human‑readable speaker name for each `.vtt` (files are renamed accordingly).
5. All renamed VTTs are parsed & merged into `session_transcript.csv` (and you still keep the individual VTTs).
6. The merged CSV is split into chunk files in `merged/chunked/` (e.g. `chunk_1.csv`, `chunk_2.csv`, ...), each covering N minutes.
7. You can open the merged or chunk CSVs to craft summaries / session notes.

## Requirements

| Component                      | Purpose                                |
| ------------------------------ | -------------------------------------- |
| Python 3.8+                    | Run the orchestration script           |
| FFmpeg                         | Required by Whisper for audio decoding |
| Whisper CLI (`openai-whisper`) | Performs transcription                 |
| (Optional) CUDA GPU + PyTorch  | Performance boost for larger models    |

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
# Ensure ffmpeg is installed & in PATH
python transcribe.py
```

When prompted:

- Enter the absolute (or relative) path to your audio folder (without quotes).
- Choose a model size (press Enter for default `turbo`).
- Provide friendly speaker names for each detected `.vtt` file after transcription.

## Model Selection Notes

| Model  | Speed            | Accuracy | Typical Use                    |
| ------ | ---------------- | -------- | ------------------------------ |
| tiny   | Fastest          | Lowest   | Quick skim / draft             |
| base   | Fast             | Low‑mid  | Casual notes                   |
| small  | Medium           | Medium   | General session recaps         |
| medium | Slower           | High     | More accurate logs             |
| large  | Slowest          | Highest  | Best quality, longest sessions |
| turbo  | Fast (optimized) | High     | Balanced (default)             |

Larger models are slower and require more VRAM; GPU highly recommended above `small`.

## Installing Prerequisites

### FFmpeg

- Windows (Chocolatey): `choco install ffmpeg`
- macOS (Homebrew): `brew install ffmpeg`
- Debian/Ubuntu: `sudo apt update && sudo apt install ffmpeg`

### Whisper CLI

```bash
pip install openai-whisper
```

If you need GPU acceleration, install PyTorch first using the selector at the [official PyTorch site](https://pytorch.org/get-started/locally/) then install Whisper.

## Output Structure

After a run your audio directory will contain (paths shown relative to your input folder):

```text
transcript/
    speaker1.vtt
    speaker2.vtt
    ...
    merged/
        session_transcript.csv
        chunked/
            chunk_1.csv
            chunk_2.csv
            ...
```

`session_transcript.csv` columns:

```text
start_time,speaker,text
00:00:12.345,Alice,Hello everyone...
00:00:14.101,Bob,Hi!
...
```

## Troubleshooting

| Symptom                               | Cause                                       | Fix                                                             |
| ------------------------------------- | ------------------------------------------- | --------------------------------------------------------------- |
| `Error: 'whisper' command not found.` | Whisper CLI not installed / wrong venv      | Activate venv then `pip install -r requirements.txt`            |
| `No audio files found`                | Wrong folder path or unsupported extensions | Check path; ensure files end in supported extensions            |
| Very slow transcription               | Using large model on CPU                    | Switch to smaller model or install CUDA + correct PyTorch build |
| GPU not used                          | PyTorch CPU only build installed            | Reinstall PyTorch with CUDA (per PyTorch site)                  |
| VTT rename collision                  | Same speaker name chosen twice              | Provide unique names when prompted                              |

## Roadmap / Ideas

- Optional automatic speaker clustering (future)
- Summarization step after chunking
- Export to Markdown session recap

## License

MIT

## Requirement Resources

- Python 3.8+
- [FFmpeg](https://ffmpeg.org/download.html)
- [OpenAI Whisper](https://github.com/openai/whisper)

## Setup and Usage

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/summary-transcriber.git
cd summary-transcriber
```

### 2. Create and Activate a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

**On Windows (Power Shell):**

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On macOS and Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install FFmpeg

Whisper requires FFmpeg to be installed and available in your system's PATH.

**On Windows (using [Chocolatey](https://chocolatey.org/)):**

```bash
choco install ffmpeg
```

**On macOS (using [Homebrew](https://brew.sh/)):**

```bash
brew install ffmpeg
```

**On Debian/Ubuntu:**

```bash
sudo apt update && sudo apt install ffmpeg
```

<!-- Legacy sections above were consolidated into Quick Start / Installing Prerequisites -->
