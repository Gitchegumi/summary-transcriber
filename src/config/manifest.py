from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SessionConfig:
    id: str
    campaign: str
    session_number: int | str
    session_date: str
    source: str = "craig"


@dataclass
class TranscriptionConfig:
    backend: str = "faster-whisper"
    model: str = "large-v3"
    device: str = "cuda"
    language: str = "en"


@dataclass
class SpeakerConfig:
    speaker_id: str
    display_name: str
    role: str
    character_name: str | None
    file: str


@dataclass
class GlossaryConfig:
    pcs: list[str] = field(default_factory=list)
    npcs: list[str] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)
    factions: list[str] = field(default_factory=list)
    items: list[str] = field(default_factory=list)
    spells: list[str] = field(default_factory=list)
    rules_terms: list[str] = field(default_factory=list)
    custom_terms: list[str] = field(default_factory=list)

    def all_terms(self) -> list[str]:
        terms: list[str] = []
        for values in self.__dict__.values():
            terms.extend(values or [])
        return sorted({term for term in terms if term})


@dataclass
class SessionManifest:
    path: Path
    session: SessionConfig
    transcription: TranscriptionConfig
    speakers: list[SpeakerConfig]
    glossary: GlossaryConfig

    @property
    def root(self) -> Path:
        return self.path.parent

    def resolve_path(self, value: str) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = self.root / path
        return path.resolve()

    def apply_overrides(
        self,
        backend: str | None = None,
        model: str | None = None,
        device: str | None = None,
    ) -> None:
        if backend:
            self.transcription.backend = backend
        if model:
            self.transcription.model = model
        if device:
            self.transcription.device = device

    def validate_audio_files(self) -> None:
        missing = [
            f"{speaker.speaker_id}: {self.resolve_path(speaker.file)}"
            for speaker in self.speakers
            if not self.resolve_path(speaker.file).exists()
        ]
        if missing:
            joined = "\n  - ".join(missing)
            raise FileNotFoundError(f"Manifest references missing audio files:\n  - {joined}")


def _require_mapping(data: Any, name: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(f"Manifest field '{name}' must be a mapping.")
    return data


def load_manifest(path: Path) -> SessionManifest:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    session_data = _require_mapping(data.get("session"), "session")
    transcription_data = _require_mapping(data.get("transcription", {}), "transcription")
    speakers_data = data.get("speakers")
    if not isinstance(speakers_data, list) or not speakers_data:
        raise ValueError("Manifest field 'speakers' must be a non-empty list.")

    speakers = [SpeakerConfig(**_require_mapping(item, "speakers[]")) for item in speakers_data]
    speaker_ids = [speaker.speaker_id for speaker in speakers]
    if len(speaker_ids) != len(set(speaker_ids)):
        raise ValueError("Manifest speaker_id values must be unique.")

    return SessionManifest(
        path=path,
        session=SessionConfig(**session_data),
        transcription=TranscriptionConfig(**transcription_data),
        speakers=speakers,
        glossary=GlossaryConfig(**(data.get("glossary") or {})),
    )
