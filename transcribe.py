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
    with open(output_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Time", "Speaker", "Text"])
        writer.writerows(sorted_entries)

    print(f"Merged {len(sorted_entries)} total lines into {output_path}")

# --- New function for renaming files ---

def rename_vtt_files(output_dir):
    """
    Asks the user for new speaker names for each VTT file and renames the files.
    """
    print("\n--- Rename Speaker Files ---")
    print("Enter a new name for each speaker. Press Enter to keep the original name.")

    vtt_files = sorted(list(output_dir.glob("*.vtt")))
    if not vtt_files:
        print("No .vtt files found to rename.")
        return

    for vtt_file in vtt_files:
        while True:
            try:
                new_speaker_name = input(f"  Speaker for '{vtt_file.name}': ")
                if not new_speaker_name.strip():
                    print(f"    Keeping original name: {vtt_file.stem}")
                    break  # Keep original name and move to the next file

                new_file_path = vtt_file.with_name(f"{new_speaker_name}.vtt")

                if new_file_path.exists():
                    print(f"    Error: A file named '{new_file_path.name}' already exists. Please choose a different name.")
                    continue  # Ask again for a new name

                vtt_file.rename(new_file_path)
                print(f"    Renamed to: {new_file_path.name}")
                break  # Success, move to the next file
            except OSError as e:
                print(f"    Error renaming file: {e}")
                break # Exit loop for this file on error

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

        # 2. Create the output directory
        output_dir = input_dir / "transcript"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output will be saved to: {output_dir}")

        # 3. Get the number of tracks
        while True:
            try:
                num_tracks_str = input("Enter the number of tracks you recorded: ")
                num_tracks = int(num_tracks_str)
                if num_tracks > 0:
                    break
                else:
                    print("Please enter a positive number.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        # 4. Loop over tracks and transcribe
        for i in range(1, num_tracks + 1):
            # Find the audio file for the current track number
            audio_extensions = [".flac", ".mp3", ".wav", ".m4a", ".ogg"]
            files = []
            for ext in audio_extensions:
                files.extend(input_dir.glob(f"{i}-*{ext}"))

            if not files:
                print(f"No file found for track {i} (pattern: {i}-*.[flac, mp3, wav, m4a, ogg]). Skipping.")
                continue

            print(f"Found {len(files)} files for track {i}: {[f.name for f in files]}")
            
            audio_file = files[0]
            print(f"\nStarting transcription for: {audio_file.name}")

            try:
                subprocess.run([
                    "whisper",
                    str(audio_file),
                    "--model", "large",
                    "--device", "cuda",
                    "--output_dir", str(output_dir)
                ], check=True)
                print(f"Finished transcription for: {audio_file.name}")
            except subprocess.CalledProcessError as e:
                print(f"Error during transcription for {audio_file.name}: {e}")
            except FileNotFoundError:
                print("Error: 'whisper' command not found.")
                print("Please ensure the Whisper CLI is installed and in your system's PATH.")
                sys.exit(1)

        print("\nAll requested tracks processed.")

        # 5. Rename the generated VTT files
        rename_vtt_files(output_dir)

        # 6. Merge the generated VTT files
        print("\nNow merging transcript files...")
        merge_transcripts(output_dir)

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)

if __name__ == "__main__":
    main()