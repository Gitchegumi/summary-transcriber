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


class ChunkPrepProgressReporter:
    def __init__(
        self,
        session_id: str,
        backend: str,
        enabled: bool = True,
        update_interval_seconds: float = 5.0,
        total_speakers: int = 0,
    ):
        self.session_id = session_id
        self.backend = backend
        self.enabled = enabled
        self.update_interval_seconds = update_interval_seconds
        
        self.start_time = time.time()
        self.last_update_time = 0.0
        
        self.total_speakers = total_speakers
        self.current_speaker_index = 0
        self.speaker_ids: list[str] = []
        self.speaker_durations: dict[str, float] = {}
        
        self.total_audio_seconds = 0.0
        self.completed_audio_seconds = 0.0
        
        self.current_speaker_id = ""
        self.current_speaker_name = ""
        self.current_source_file = ""
        self.current_phase = "inspecting"
        self.status_msg = ""
        
        self.current_speaker_chunks_completed = 0
        self.current_speaker_chunks_total = 0
        self.current_speaker_audio_completed = 0.0
        self.current_speaker_audio_total = 0.0
        
        self.chunks_generated = 0
        self.chunks_reused = 0
        self.chunks_skipped = 0
        self.chunks_regenerated = 0
        self.chunks_deleted = 0
        
        self.last_lines_count = 0
        self.last_active_lines_count = 0
        self._has_printed = False

    def set_speaker_ids(self, speaker_ids: list[str]) -> None:
        self.speaker_ids = speaker_ids

    def set_total_audio_seconds(self, total_seconds: float) -> None:
        self.total_audio_seconds = total_seconds

    def set_speaker_duration(self, speaker_id: str, duration: float) -> None:
        self.speaker_durations[speaker_id] = duration

    def set_phase(self, phase: str) -> None:
        self.current_phase = phase
        self.print_progress(force=True)

    def start_speaker(
        self,
        speaker_id: str,
        display_name: str,
        source_file: str,
        speaker_index: int,
        phase: str,
    ) -> None:
        self.current_speaker_id = speaker_id
        self.current_speaker_name = display_name
        self.current_source_file = source_file
        self.current_speaker_index = speaker_index
        self.current_phase = phase
        self.current_speaker_chunks_completed = 0
        self.current_speaker_chunks_total = 0
        self.current_speaker_audio_completed = 0.0
        self.current_speaker_audio_total = self.speaker_durations.get(speaker_id, 0.0)
        self.status_msg = ""
        
        # Calculate completed duration of all previous speakers
        completed_speakers_duration = sum(
            self.speaker_durations.get(sp_id, 0.0)
            for sp_id in self.speaker_ids[:self.current_speaker_index]
        )
        self.completed_audio_seconds = completed_speakers_duration
        self.print_progress(force=True)

    def start_chunking(self, speaker_id: str, phase: str, num_chunks: int) -> None:
        self.current_phase = phase
        self.current_speaker_chunks_total = num_chunks
        self.current_speaker_chunks_completed = 0
        self.current_speaker_audio_completed = 0.0
        self.print_progress(force=True)

    def complete_chunk_prep(self, speaker_id: str, chunk_duration: float, is_regenerated: bool) -> None:
        self.chunks_generated += 1
        if is_regenerated:
            self.chunks_regenerated += 1
        self.current_speaker_chunks_completed += 1
        self.current_speaker_audio_completed += chunk_duration
        
        # Recalculate completed_audio_seconds
        completed_speakers_duration = sum(
            self.speaker_durations.get(sp_id, 0.0)
            for sp_id in self.speaker_ids[:self.current_speaker_index]
        )
        self.completed_audio_seconds = completed_speakers_duration + self.current_speaker_audio_completed
        self.status_msg = "regenerated" if is_regenerated else "created"
        self.print_progress()

    def report_reuse(self, speaker_id: str, num_chunks: int) -> None:
        self.current_phase = "reusing cached chunks"
        self.chunks_reused += num_chunks
        self.chunks_skipped += num_chunks
        self.current_speaker_chunks_completed = num_chunks
        self.current_speaker_chunks_total = num_chunks
        
        dur = self.speaker_durations.get(speaker_id, 0.0)
        self.current_speaker_audio_completed = dur
        
        completed_speakers_duration = sum(
            self.speaker_durations.get(sp_id, 0.0)
            for sp_id in self.speaker_ids[:self.current_speaker_index]
        )
        self.completed_audio_seconds = completed_speakers_duration + dur
        self.status_msg = "reused from cache"
        self.print_progress(force=True)

    def report_cleanup(self, deleted_count: int) -> None:
        self.current_phase = "cleaning chunks"
        self.chunks_deleted += deleted_count
        self.status_msg = f"deleted {deleted_count} orphaned chunks"
        self.print_progress(force=True)

    def print_progress(self, force: bool = False) -> None:
        if not self.enabled:
            return
            
        now = time.time()
        if not force and (now - self.last_update_time) < self.update_interval_seconds:
            return
            
        self.last_update_time = now
        elapsed = now - self.start_time
        
        # Speed: audio seconds per wall-clock second
        if elapsed > 0:
            speed = self.completed_audio_seconds / elapsed
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
        
        lines = []
        lines.append("Preparing audio chunks for NeMo backend")
        lines.append(f"Speaker: {self.current_speaker_id} / {self.current_speaker_name}")
        
        if self.current_phase == "reusing cached chunks":
            lines.append(f"Phase: {self.current_phase}")
            lines.append(f"Chunks: {self.current_speaker_chunks_completed} / {self.current_speaker_chunks_total}")
            lines.append(f"Status: {self.status_msg}")
        elif self.current_phase == "regenerating chunks":
            lines.append(f"Phase: {self.current_phase}")
            lines.append(f"Chunks: {self.current_speaker_chunks_completed} / {self.current_speaker_chunks_total}")
        else:
            if self.current_source_file:
                lines.append(f"File: {self.current_source_file}")
            lines.append(f"Phase: {self.current_phase}")
            
            speaker_completed_min = self.current_speaker_audio_completed / 60.0
            speaker_total_min = self.current_speaker_audio_total / 60.0
            completed_min = self.completed_audio_seconds / 60.0
            total_min = self.total_audio_seconds / 60.0
            
            lines.append(f"Speaker progress: {speaker_completed_min:.1f} min / {speaker_total_min:.1f} min")
            lines.append(f"Overall progress: {completed_min:.1f} min / {total_min:.1f} min")
            
            if self.current_speaker_chunks_total > 0:
                lines.append(f"Chunks: {self.current_speaker_chunks_completed} / {self.current_speaker_chunks_total}")
            else:
                lines.append(f"Chunks: {self.chunks_generated}")
                
            lines.append(f"Reused: {self.chunks_reused}")
            lines.append(f"Elapsed: {elapsed_str}")
            if eta > 0:
                lines.append(f"ETA: {eta_str}")
        
        # Clear printed lines and output new block
        if self._has_printed and self.last_lines_count > 0:
            sys.stdout.write("\033[A" * (self.last_lines_count - 1) + "\r")
            
        self.last_active_lines_count = len(lines)
        actual_len = len(lines)
        if self.last_lines_count > actual_len:
            lines.extend([""] * (self.last_lines_count - actual_len))
            
        sys.stdout.write("\r\033[K" + "\n\r\033[K".join(lines))
        sys.stdout.flush()
        
        self.last_lines_count = len(lines)
        self._has_printed = True

    def finish(self) -> None:
        if not self.enabled:
            return
        if self._has_printed:
            excess_lines = self.last_lines_count - getattr(self, "last_active_lines_count", self.last_lines_count)
            if excess_lines > 0:
                sys.stdout.write("\033[A" * excess_lines + "\r")
                sys.stdout.write("\n".join("\r\033[K" for _ in range(excess_lines + 1)))
                sys.stdout.write("\033[A" * excess_lines + "\r")
            sys.stdout.write("\n")
            sys.stdout.flush()
