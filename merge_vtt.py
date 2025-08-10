import re
import sys
from pathlib import Path
import csv

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

def main(vtt_folder, output_file="merged.csv"):
    folder = Path(vtt_folder)
    if not folder.is_dir():
        print(f"Error: {vtt_folder} is not a directory.")
        sys.exit(1)

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

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python merge_vtt.py <folder_with_vtt_files>")
    else:
        main(sys.argv[1])
