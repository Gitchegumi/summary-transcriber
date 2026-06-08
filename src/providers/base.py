from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from src.config.manifest import SpeakerConfig


@dataclass
class ProviderContext:
    backend: str
    model: str
    device: str = "cuda"
    language: str = "en"


class TranscriptionProvider(Protocol):
    context: ProviderContext

    def transcribe(self, audio_path: Path, speaker: SpeakerConfig) -> dict:
        """Return raw local model output as a JSON-serializable dictionary."""
