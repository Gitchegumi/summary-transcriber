# Summary Transcriber

This script transcribes multiple audio files using OpenAI's Whisper, then merges and sorts the transcriptions into a single CSV file. It's designed to process audio from multiple speakers and combine their dialogue in chronological order.

## Background

This tool was developed to help with a Dungeons & Dragons campaign. Our group consists of working adults with families, which means there can be long breaks between sessions. To help everyone get back up to speed, this script is used to transcribe our game sessions and generate a summary that can be sent out as a "read-ahead" before we meet.

The audio is captured using the [craig.chat](https://craig.chat/) bot for Discord, but the use of Craig is beyond the scope of this project. You are free to use any audio recording software that can produce separate audio tracks for each speaker.

## How it Works

The script performs the following steps:

1.  **Gets Audio Files**: It prompts the user for the path to a folder containing audio files.
2.  **Creates Output Directory**: It creates a `transcript` subdirectory within the audio folder to store the output.
3.  **Transcribes Audio**: It uses the `whisper` command-line tool to transcribe each audio file. It looks for files named `1-*` with common audio extensions (`.flac`, `.mp3`, `.wav`, `.m4a`, `.ogg`), corresponding to the number of tracks you specify.
4.  **Renames Speaker Files**: After transcription, it prompts the user to enter a speaker name for each generated `.vtt` file, renaming the files accordingly.
5.  **Merges Transcripts**: It parses the VTT files, combines them, sorts the entries by timestamp, and saves the final merged transcript as `merged.csv` in the `transcript` folder.

## Requirements

*   Python 3.8+
*   [FFmpeg](https://ffmpeg.org/download.html)
*   [OpenAI Whisper](https://github.com/openai/whisper)

## Setup and Usage

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/summary-transcriber.git
cd summary-transcriber
```

### 2. Create and Activate a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

**On Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
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

### 4. Install Whisper


You can install Whisper using pip:

```bash
pip install git+https://github.com/openai/whisper.git
```

For better performance, especially with larger models, a CUDA-capable NVIDIA GPU is recommended. Ensure you have the correct version of PyTorch installed for your system by following the instructions on the [PyTorch website](https://pytorch.org/).

### 5. Run the Script

Once the setup is complete, you can run the script:

```bash
python transcribe.py
```

The script will guide you through the process of selecting your audio folder and specifying the number of tracks.
