from __future__ import annotations

import sys
import time
from typing import Any


def format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class ProgressReporter:
    def __init__(
        self,
        session_id: str,
        backend: str,
        enabled: bool = True,
        update_interval_seconds: float = 5.0,
    ):
        self.session_id = session_id
        self.backend = backend
        self.enabled = enabled
        self.update_interval_seconds = update_interval_seconds
        
        self.start_time = time.time()
        self.last_update_time = 0.0
        
        self.total_chunks = 0
        self.completed_chunks = 0
        self.total_audio_seconds = 0.0
        self.completed_audio_seconds = 0.0
        
        self.current_speaker_id = ""
        self.current_speaker_name = ""
        self.current_speaker_chunks_total = 0
        self.current_speaker_chunks_completed = 0
        self.current_speaker_audio_completed = 0.0
        self.current_speaker_audio_total = 0.0
        
        self._has_printed = False

    def init_totals(self, total_chunks: int, total_audio_seconds: float) -> None:
        self.total_chunks = total_chunks
        self.total_audio_seconds = total_audio_seconds

    def start_speaker(
        self,
        speaker_id: str,
        display_name: str,
        speaker_chunks_total: int,
        speaker_audio_total: float,
    ) -> None:
        self.current_speaker_id = speaker_id
        self.current_speaker_name = display_name
        self.current_speaker_chunks_total = speaker_chunks_total
        self.current_speaker_chunks_completed = 0
        self.current_speaker_audio_completed = 0.0
        self.current_speaker_audio_total = speaker_audio_total
        self.print_progress(force=True)

    def complete_chunk(self, chunk_duration: float) -> None:
        self.completed_chunks += 1
        self.completed_audio_seconds += chunk_duration
        self.current_speaker_chunks_completed += 1
        self.current_speaker_audio_completed += chunk_duration
        self.print_progress()

    def print_progress(self, force: bool = False) -> None:
        if not self.enabled:
            return
            
        now = time.time()
        if not force and (now - self.last_update_time) < self.update_interval_seconds:
            return
            
        self.last_update_time = now
        elapsed = now - self.start_time
        
        # Speed: audio minutes per wall-clock minute
        if elapsed > 0:
            speed = self.completed_audio_seconds / elapsed  # audio-sec / wall-sec
        else:
            speed = 0.0
            
        # ETA
        if speed > 0:
            remaining_audio = max(0.0, self.total_audio_seconds - self.completed_audio_seconds)
            eta = remaining_audio / speed
        else:
            eta = 0.0
            
        elapsed_str = format_duration(elapsed)
        eta_str = format_duration(eta)
        
        # Audio minutes conversions
        completed_min = self.completed_audio_seconds / 60.0
        total_min = self.total_audio_seconds / 60.0
        
        speaker_completed_min = self.current_speaker_audio_completed / 60.0
        speaker_total_min = self.current_speaker_audio_total / 60.0
        
        # Print block
        # Clear lines and print
        sys.stdout.write(
            f"\r\033[KTranscribing {self.session_id} with {self.backend}\n"
            f"\r\033[KSpeaker: {self.current_speaker_id} / {self.current_speaker_name}\n"
            f"\r\033[KChunk: {self.current_speaker_chunks_completed} / {self.current_speaker_chunks_total}\n"
            f"\r\033[KAudio processed: {speaker_completed_min:.1f} min / {speaker_total_min:.1f} min\n"
            f"\r\033[KOverall audio processed: {completed_min:.1f} min / {total_min:.1f} min\n"
            f"\r\033[KElapsed: {elapsed_str} | ETA: {eta_str} | Speed: {speed:.1f} audio-min / wall-min\033[A\033[A\033[A\033[A\033[A"
        )
        sys.stdout.flush()
        self._has_printed = True

    def finish(self) -> None:
        if not self.enabled:
            return
            
        # Move cursor past the status lines
        if self._has_printed:
            sys.stdout.write("\n\n\n\n\n\n")
            sys.stdout.flush()
