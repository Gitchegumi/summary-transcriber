#!/usr/bin/env python
import re
import sys
import csv
import subprocess
from pathlib import Path

# --- Functions from merge_vtt.py ---


def parse_vtt(file_path):
    """
    Parse a VTT file into a list of tuples: (start_time, text, speaker).
    """
    speaker = Path(file_path).stem
    entries = []

    with open(file_path, encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    prev_text = None
    while i < len(lines):
        line = lines[i].strip()
        if re.match(r"^(\d{2}:)?\d{2}:\d{2}\.\d{3}", line):
            time = line.split(" --> ")[0].strip()
            i += 1
            text_lines = []
            while i < len(lines) and lines[i].strip():
                text_lines.append(lines[i].strip())
                i += 1
            text = " ".join(text_lines)
            # De-duplicate consecutive same-text lines in this file
            if text != prev_text:
                entries.append((time, speaker, text))
                prev_text = text
        i += 1

    return entries


def pad_time_string(t):
    """
    Ensure time strings are in hh:mm:ss.mmm format by adding hours if missing.
    """
    if re.match(r"^\d{2}:\d{2}\.\d{3}$", t):
        return f"00:{t}"
    return t


def time_to_seconds(t):
    """
    Convert hh:mm:ss.mmm to seconds (float).
    """
    t = pad_time_string(t)
    h, m, s = t.split(":")
    s, ms = s.split(".")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def merge_transcripts(vtt_folder, output_file="merged.csv"):
    """
    Merges all .vtt files in a folder into a single sorted CSV file.
    """
    folder = Path(vtt_folder)
    if not folder.is_dir():
        print(f"Error: {vtt_folder} is not a directory.")
        return

    all_entries = []

    for vtt_file in folder.glob("*.vtt"):
        entries = parse_vtt(vtt_file)
        all_entries.extend(entries)
        print(f"Parsed {len(entries)} lines from {vtt_file.name}")

    # Sort everything by numeric time
    sorted_entries = sorted(all_entries, key=lambda x: time_to_seconds(x[0]))

    # Write to CSV
    output_path = folder / output_file
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Time", "Speaker", "Text"])
        writer.writerows(sorted_entries)

    print(f"Merged {len(sorted_entries)} total lines into {output_path}")


# --- New function for renaming files ---


def rename_vtt_files(source_dir, dest_dir):
    """
    Asks the user for new speaker names for each VTT file and saves the renamed
    files to the destination directory.
    """
    print("\n--- Rename Speaker Files ---")
    print("Enter a new name for each speaker. Press Enter to keep the original name.")

    vtt_files = sorted(list(source_dir.glob("*.vtt")))
    if not vtt_files:
        print("No .vtt files found to rename.")
        return

    for vtt_file in vtt_files:
        original_name = vtt_file.stem
        while True:
            try:
                new_speaker_name = input(f"  Speaker for '{vtt_file.name}': ")
                if not new_speaker_name.strip():
                    new_speaker_name = original_name
                    print(f"    Keeping original name: {original_name}")

                new_file_path = dest_dir / f"{new_speaker_name}.vtt"

                if new_file_path.exists():
                    print(
                        f"    Error: A file named '{new_file_path.name}' already exists in '{dest_dir}'. Please choose a different name."
                    )
                    continue  # Ask again for a new name

                # Rename (move) the file to the new directory with the new name
                vtt_file.rename(new_file_path)
                print(f"    Saved and renamed to: {new_file_path}")
                break  # Success, move to the next file
            except OSError as e:
                print(f"    Error renaming file: {e}")
                break  # Exit loop for this file on error



# --- Main transcription logic ---


def main():
    """
    Main function to transcribe audio files and merge the VTT outputs.
    """
    try:
        # 1. Get the input directory from the user
        input_dir_str = input("Enter the full path to your audio folder: ")
        input_dir = Path(input_dir_str).resolve()

        if not input_dir.is_dir():
            print(f"Error: The folder '{input_dir}' does not exist.")
            sys.exit(1)

        # 2. Create output directories
        transcript_dir = input_dir / "transcript"
        merged_dir = input_dir / "merged"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        merged_dir.mkdir(parents=True, exist_ok=True)
        print(f"Transcript files will be saved to: {transcript_dir}")
        print(f"Merged output will be saved to: {merged_dir}")

        # 3. Get model size from user
        model_sizes = ["tiny", "base", "small", "medium", "large", "turbo"]
        model_prompt = f"Please choose a model size ({', '.join(model_sizes)}). Press Enter for default (turbo): "
        selected_model = input(model_prompt).strip().lower()
        if not selected_model or selected_model not in model_sizes:
            selected_model = "turbo"
        print(f"Using model size: {selected_model}")

        # 4. Find all audio files in the directory
        audio_extensions = [".flac", ".mp3", ".wav", ".m4a", ".ogg"]
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(input_dir.glob(f"*{ext}"))

        if not audio_files:
            print(f"No audio files found in '{input_dir}'.")
            sys.exit(1)

        print(f"Found {len(audio_files)} audio files to transcribe:")
        for f in audio_files:
            print(f"  - {f.name}")

        # 5. Loop over tracks and transcribe
        for audio_file in audio_files:
            print(f"\nStarting transcription for: {audio_file.name}")

            try:
                process = subprocess.Popen(
                    [
                        "whisper",
                        str(audio_file),
                        "--model",
                        selected_model,
                        "--device",
                        "cuda",
                        "--output_dir",
                        str(transcript_dir),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    bufsize=1,
                )

                for line in iter(process.stdout.readline, ""):
                    print(line, end="")

                process.wait()
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(
                        process.returncode, process.args
                    )
                print(f"Finished transcription for: {audio_file.name}")
            except subprocess.CalledProcessError as e:
                print(f"Error during transcription for {audio_file.name}: {e}")
            except FileNotFoundError:
                print("Error: 'whisper' command not found.")
                print(
                    "Please ensure the Whisper CLI is installed and in your system's PATH."
                )
                sys.exit(1)

        print("\nAll requested tracks processed.")

        # 6. Rename the generated VTT files
        rename_vtt_files(transcript_dir, merged_dir)

        # 7. Merge the generated VTT files
        print("\nNow merging transcript files...")
        merge_transcripts(merged_dir)

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
