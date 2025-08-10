#!/bin/bash

# Base directory containing all sessions
BASE_DIR="/mnt/e/Content Creation/10 - DnD sessions"

# Prompt for the relative path to the folder containing your audio files
echo "Your input path should start after: $BASE_DIR"
read -p "Enter the relative path to your audio folder: " RELATIVE_PATH

# Build the full input/output directory paths
INPUT_DIR="$BASE_DIR/$RELATIVE_PATH"
echo "Input directory: $INPUT_DIR"
OUTPUT_DIR="$INPUT_DIR/transcript"
echo "Output directory: $OUTPUT_DIR"

# Check if the folder exists
if [ ! -d "$INPUT_DIR" ]; then
  echo "Error: The folder '$INPUT_DIR' does not exist."
  exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Prompt for number of tracks
read -p "Enter the number of tracks you recorded: " NUMBER_OF_TRACKS

# Loop over track numbers
for (( NUM=1; NUM<=NUMBER_OF_TRACKS; NUM++ )); do
  # Look for the file matching pattern like 1-*.flac, 2-*.flac, etc.
  FILE=$(ls "$INPUT_DIR/${NUM}-"*.flac 2>/dev/null)

  if [ -z "$FILE" ]; then
    echo "No file found for track ${NUM} (pattern: ${NUM}-*.flac). Skipping."
    continue
  fi

  echo "Starting transcription for: $(basename "$FILE")"
  whisper "$FILE" \
    --model large \
    --device cuda \
    --output_dir "$OUTPUT_DIR"
  echo "Finished transcription for: $(basename "$FILE")"
done

echo "All requested tracks processed."
