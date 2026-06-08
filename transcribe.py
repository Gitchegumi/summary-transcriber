#!/usr/bin/env python
"""
Audio transcription tool with speaker diarization and summarization capabilities.
"""
import re
import sys
import csv
import json
import subprocess
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# --- Functions from merge_vtt.py ---


def parse_vtt(file_path):
    """
    Parse a VTT file into a list of tuples: (start_time, end_time, speaker, text).
    """
    speaker = Path(file_path).stem
    entries = []

    with open(file_path, encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    prev_text = None
    while i < len(lines):
        line = lines[i].strip()
        if re.match(r"^(\d{2}:)?\d{2}:\d{2}\.\d{3}\s+-->", line):
            start_time, end_time = [
                part.strip().split()[0] for part in line.split(" --> ", 1)
            ]
            i += 1
            text_lines = []
            while i < len(lines) and lines[i].strip():
                text_lines.append(lines[i].strip())
                i += 1
            text = " ".join(text_lines)
            # De-duplicate consecutive same-text lines in this file
            if text != prev_text:
                entries.append(
                    (
                        pad_time_string(start_time),
                        pad_time_string(end_time),
                        speaker,
                        text,
                    )
                )
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

    # Sort everything by numeric start time
    sorted_entries = sorted(all_entries, key=lambda x: time_to_seconds(x[0]))

    # Write to CSV
    output_path = folder / output_file
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Start", "End", "Speaker", "Text"])
        writer.writerows(sorted_entries)

    print(f"Merged {len(sorted_entries)} total lines into {output_path}")
    return output_path


# --- New function for renaming files ---


def rename_vtt_files(audio_files, dest_dir):
    """
    Ask the user for speaker names before transcription starts.
    """
    print("\n--- Name Speaker Tracks ---")
    print("Enter a speaker name for each audio track.")
    print("Press Enter to use the audio filename as the speaker name.")

    speaker_names = {}
    planned_outputs = set()

    for audio_file in audio_files:
        original_name = audio_file.stem
        while True:
            new_speaker_name = input(f"  Speaker for '{audio_file.name}': ").strip()
            if not new_speaker_name:
                new_speaker_name = original_name
                print(f"    Using original name: {original_name}")

            new_file_path = dest_dir / f"{new_speaker_name}.vtt"

            if new_file_path in planned_outputs:
                print(
                    f"    Error: '{new_file_path.name}' is already planned for "
                    "another track. Please choose a different name."
                )
                continue

            if new_file_path.exists():
                print(
                    f"    Error: A file named '{new_file_path.name}' already "
                    f"exists in '{dest_dir}'. Please choose a different name."
                )
                continue

            speaker_names[audio_file] = new_speaker_name
            planned_outputs.add(new_file_path)
            break

    return speaker_names


def move_transcribed_vtt(audio_file, transcript_dir, merged_dir, speaker_name):
    """
    Move a transcribed VTT file into the merged folder with its speaker name.
    """
    source_path = transcript_dir / f"{audio_file.stem}.vtt"
    dest_path = merged_dir / f"{speaker_name}.vtt"

    if not source_path.exists():
        print(f"Warning: Expected VTT was not found: {source_path}")
        return

    try:
        shutil.move(str(source_path), str(dest_path))
        print(f"Saved speaker transcript to: {dest_path}")
        return dest_path
    except OSError as e:
        print(f"Error moving transcript for {audio_file.name}: {e}")
        return None


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
        model_prompt = (
            f"Please choose a model size ({', '.join(model_sizes)}). "
            f"Press Enter for default (turbo): "
        )
        selected_model = input(model_prompt).strip().lower()
        if not selected_model or selected_model not in model_sizes:
            selected_model = "turbo"
        print(f"Using model size: {selected_model}")

        # 4. Find all audio files in the directory
        audio_extensions = [".flac", ".mp3", ".wav", ".m4a", ".ogg"]
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(input_dir.glob(f"*{ext}"))
        audio_files = sorted(audio_files)

        if not audio_files:
            print(f"No audio files found in '{input_dir}'.")
            sys.exit(1)

        print(f"Found {len(audio_files)} audio files to transcribe:")
        for f in audio_files:
            print(f"  - {f.name}")

        # 5. Collect speaker names before the long transcription step.
        speaker_names = rename_vtt_files(audio_files, merged_dir)

        # 6. Loop over tracks and transcribe
        for audio_file in audio_files:
            print(f"\nStarting transcription for: {audio_file.name}")

            try:
                subprocess.run(
                    [
                        "whisper",
                        str(audio_file),
                        "--model",
                        selected_model,
                        "--device",
                        "cuda",
                        "--language",
                        "English",
                        "--output_format",
                        "vtt",
                        "--output_dir",
                        str(transcript_dir),
                    ],
                    check=True,
                )

                print(f"Finished transcription for: {audio_file.name}")
                move_transcribed_vtt(
                    audio_file,
                    transcript_dir,
                    merged_dir,
                    speaker_names[audio_file],
                )
            except subprocess.CalledProcessError as e:
                print(f"Error during transcription for {audio_file.name}: {e}")
            except FileNotFoundError:
                print("Error: 'whisper' command not found.")
                print(
                    "Please ensure the Whisper CLI is installed and in your system's PATH."
                )
                sys.exit(1)

        print("\nAll requested tracks processed.")

        # 7. Merge the generated VTT files
        print("\nNow merging transcript files...")
        merged_csv_path = merged_dir / "session_transcript.csv"
        merge_transcripts(merged_dir, output_file=merged_csv_path.name)

        # 8. Chunk the merged CSV
        chunk_merged_csv(merged_csv_path)

        # 9. Write agent-oriented transcript artifacts
        write_agent_outputs(merged_csv_path)

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)


def seconds_to_time(s):
    """
    Convert seconds (float) to hh:mm:ss.mmm format.
    """
    h = int(s // 3600)
    s %= 3600
    m = int(s // 60)
    s %= 60
    sec = int(s)
    ms = int((s - sec) * 1000)
    return f"{h:02d}:{m:02d}:{sec:02d}.{ms:03d}"


def format_duration(total_seconds):
    """
    Convert seconds to a compact human-readable duration.
    """
    total_seconds = int(total_seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def read_merged_entries(merged_file):
    """
    Read the merged transcript CSV into dictionaries for downstream exports.
    """
    entries = []
    with open(merged_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append(
                {
                    "start": row["Start"],
                    "end": row["End"],
                    "speaker": row["Speaker"],
                    "text": row["Text"],
                    "start_seconds": time_to_seconds(row["Start"]),
                    "end_seconds": time_to_seconds(row["End"]),
                }
            )
    return entries


def group_entries_by_duration(entries, chunk_duration_minutes):
    """
    Group transcript entries into time-based chunks.
    """
    chunk_duration_seconds = chunk_duration_minutes * 60
    chunks = []
    current_chunk = []
    current_chunk_start_time = 0

    for entry in entries:
        current_time = entry["start_seconds"]
        if not current_chunk:
            current_chunk_start_time = current_time
            current_chunk.append(entry)
            continue

        if current_time - current_chunk_start_time <= chunk_duration_seconds:
            current_chunk.append(entry)
        else:
            chunks.append(current_chunk)
            current_chunk = [entry]
            current_chunk_start_time = current_time

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def summarize_speaker_activity(entries):
    """
    Build simple speaker activity stats from transcript entries.
    """
    stats = defaultdict(lambda: {"turns": 0, "words": 0})
    for entry in entries:
        speaker = entry["speaker"]
        stats[speaker]["turns"] += 1
        stats[speaker]["words"] += len(entry["text"].split())

    return dict(sorted(stats.items(), key=lambda item: item[0].lower()))


def write_agent_outputs(merged_file, chunk_duration_minutes=30):
    """
    Write JSONL and Markdown artifacts designed for AI summary agents.
    """
    merged_path = Path(merged_file)
    entries = read_merged_entries(merged_path)
    if not entries:
        print("No merged transcript entries found for agent outputs.")
        return

    chunks = group_entries_by_duration(entries, chunk_duration_minutes)
    agent_dir = merged_path.parent / "agent"
    chunk_dir = agent_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    for old_chunk in chunk_dir.glob("chunk_*.md"):
        old_chunk.unlink()

    jsonl_path = agent_dir / "session_transcript.jsonl"
    with open(jsonl_path, "w", encoding="utf-8", newline="\n") as f:
        for index, entry in enumerate(entries, start=1):
            payload = {
                "index": index,
                "start": entry["start"],
                "end": entry["end"],
                "speaker": entry["speaker"],
                "text": entry["text"],
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    speaker_stats = summarize_speaker_activity(entries)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    session_start = entries[0]["start"]
    session_end = entries[-1]["end"]
    session_duration = format_duration(
        entries[-1]["end_seconds"] - entries[0]["start_seconds"]
    )

    brief_path = agent_dir / "session_summary_input.md"
    with open(brief_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("# Session Summary Input\n\n")
        f.write("## Agent Instructions\n\n")
        f.write("- Use this transcript to produce a chronological session summary.\n")
        f.write("- Preserve speaker-attributed facts when they affect decisions or intent.\n")
        f.write("- Separate confirmed events from inferred motivations or unresolved questions.\n")
        f.write("- Track names, places, items, quests, decisions, and follow-up hooks.\n")
        f.write("- Prefer concise recap prose over exhaustive transcript restatement.\n\n")

        f.write("## Session Metadata\n\n")
        f.write(f"- Generated at: {generated_at}\n")
        f.write(f"- Transcript span: {session_start} to {session_end}\n")
        f.write(f"- Approximate duration: {session_duration}\n")
        f.write(f"- Transcript turns: {len(entries)}\n")
        f.write(f"- Chunk duration: {chunk_duration_minutes} minutes\n\n")

        f.write("## Speaker Activity\n\n")
        for speaker, stats in speaker_stats.items():
            f.write(
                f"- {speaker}: {stats['turns']} turns, "
                f"approximately {stats['words']} words\n"
            )

        f.write("\n## Chunk Index\n\n")
        for index, chunk in enumerate(chunks, start=1):
            start = chunk[0]["start"]
            end = chunk[-1]["end"]
            speakers = sorted({entry["speaker"] for entry in chunk})
            f.write(
                f"- chunk_{index:03d}.md: {start} to {end}, "
                f"{len(chunk)} turns, speakers: {', '.join(speakers)}\n"
            )

    for index, chunk in enumerate(chunks, start=1):
        chunk_path = chunk_dir / f"chunk_{index:03d}.md"
        with open(chunk_path, "w", encoding="utf-8", newline="\n") as f:
            start = chunk[0]["start"]
            end = chunk[-1]["end"]
            f.write(f"# Chunk {index:03d}\n\n")
            f.write(f"- Time span: {start} to {end}\n")
            f.write(f"- Turns: {len(chunk)}\n")
            speakers = sorted({entry["speaker"] for entry in chunk})
            f.write(f"- Speakers: {', '.join(speakers)}\n\n")
            f.write("## Transcript\n\n")
            for entry in chunk:
                f.write(
                    f"[{entry['start']} - {entry['end']}] "
                    f"{entry['speaker']}: {entry['text']}\n"
                )

    print(f"Wrote agent JSONL transcript to: {jsonl_path}")
    print(f"Wrote agent summary input to: {brief_path}")
    print(f"Wrote {len(chunks)} agent chunk files to: {chunk_dir}")


def chunk_merged_csv(merged_file, chunk_duration_minutes=30):
    """
    Splits the merged CSV into smaller chunks of a specified duration.
    """
    merged_path = Path(merged_file)
    chunk_dir = merged_path.parent / "chunked"
    chunk_dir.mkdir(exist_ok=True)
    for old_chunk in chunk_dir.glob("chunk_*.csv"):
        old_chunk.unlink()

    chunk_duration_seconds = chunk_duration_minutes * 60
    chunk_number = 1
    current_chunk_start_time = 0
    rows_in_chunk = []

    print(f"\nChunking transcript into {chunk_duration_minutes}-minute files...")

    with open(merged_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header

        for row in reader:
            time_str = row[0]
            current_time = time_to_seconds(time_str)

            if not rows_in_chunk:
                # This is the first row of a new chunk
                current_chunk_start_time = current_time
                rows_in_chunk.append(row)
            else:
                # Check if adding this row exceeds the chunk duration
                if current_time - current_chunk_start_time <= chunk_duration_seconds:
                    rows_in_chunk.append(row)
                else:
                    # Write the current chunk to a file
                    chunk_filename = chunk_dir / f"chunk_{chunk_number}.csv"
                    with open(
                        chunk_filename, "w", newline="", encoding="utf-8"
                    ) as chunk_f:
                        writer = csv.writer(chunk_f)
                        writer.writerow(header)
                        writer.writerows(rows_in_chunk)
                    print(
                        f"  - Wrote {len(rows_in_chunk)} lines to {chunk_filename.name}"
                    )

                    # Start a new chunk
                    chunk_number += 1
                    rows_in_chunk = [row]
                    current_chunk_start_time = current_time

    # Write the last remaining chunk
    if rows_in_chunk:
        chunk_filename = chunk_dir / f"chunk_{chunk_number}.csv"
        with open(chunk_filename, "w", newline="", encoding="utf-8") as chunk_f:
            writer = csv.writer(chunk_f)
            writer.writerow(header)
            writer.writerows(rows_in_chunk)
        print(f"  - Wrote {len(rows_in_chunk)} lines to {chunk_filename.name}")

    print("Finished chunking.")


if __name__ == "__main__":
    main()
