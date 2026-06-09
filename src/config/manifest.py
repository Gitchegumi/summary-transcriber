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
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    chunk_seconds: float = 60.0
    overlap_seconds: float = 2.0
    min_chunk_seconds: float = 15.0
    chunk_format: str = "wav"


@dataclass
class TranscriptionConfig:
    backend: str = "faster-whisper"
    model: str = "large-v3"
    device: str = "cuda"
    language: str = "en"
    batch_size: int = 1
    precision: str = "float16"
    continue_on_error: bool = True
    retry_on_oom: bool = True
    oom_retry_chunk_seconds: list[float] = field(default_factory=lambda: [30.0, 15.0])


@dataclass
class ProgressConfig:
    enabled: bool = True
    show_eta: bool = True
    update_interval_seconds: float = 5.0


@dataclass
class WorkflowConfig:
    mode: str = "draft"
    generate_candidate_glossary: bool = True


@dataclass
class SpeakerConfig:
    speaker_id: str
    display_name: str
    role: str
    character_name: str | None
    file: str


@dataclass
class GlossaryEntry:
    canonical: str
    type: str | None = None
    aliases: list[str] = field(default_factory=list)
    description: str | None = None
    related_to: list[dict[str, str]] = field(default_factory=list)


@dataclass
class GlossaryConfig:
    entries: dict[str, list[GlossaryEntry]] = field(default_factory=dict)

    def all_terms(self) -> list[str]:
        terms: set[str] = set()
        for entries_list in self.entries.values():
            for entry in entries_list:
                if entry.canonical:
                    terms.add(entry.canonical)
                for alias in entry.aliases:
                    if alias:
                        terms.add(alias)
        return sorted({term for term in terms if term})


@dataclass
class GlossaryCandidateFiltersConfig:
    min_score: int = 3
    min_occurrences: int = 2
    filter_sentence_starters: bool = True
    filter_common_words: bool = True
    filter_profanity: bool = True
    include_real_people: bool = False
    include_noise_report: bool = True
    custom_noise_terms: list[str] = field(default_factory=list)

@dataclass
class DeduplicationConfig:
    enabled: bool = False


@dataclass
class CompletenessConfig:
    max_missing_seconds_warn: float = 30.0
    fail_on_missing_coverage: bool = False


@dataclass
class SessionManifest:
    path: Path
    session: SessionConfig
    audio: AudioConfig
    transcription: TranscriptionConfig
    progress: ProgressConfig
    speakers: list[SpeakerConfig]
    glossary_sources: list[str] = field(default_factory=list)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    glossary: GlossaryConfig = field(default_factory=GlossaryConfig)
    glossary_candidate_filters: GlossaryCandidateFiltersConfig = field(default_factory=GlossaryCandidateFiltersConfig)
    deduplication: DeduplicationConfig = field(default_factory=DeduplicationConfig)
    completeness: CompletenessConfig = field(default_factory=CompletenessConfig)

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
    audio_data = data.get("audio") or {}
    transcription_data = data.get("transcription") or {}
    progress_data = data.get("progress") or {}
    workflow_data = data.get("workflow") or {}
    speakers_data = data.get("speakers")
    if not isinstance(speakers_data, list) or not speakers_data:
        raise ValueError("Manifest field 'speakers' must be a non-empty list.")

    speakers = [SpeakerConfig(**_require_mapping(item, "speakers[]")) for item in speakers_data]
    speaker_ids = [speaker.speaker_id for speaker in speakers]
    if len(speaker_ids) != len(set(speaker_ids)):
        raise ValueError("Manifest speaker_id values must be unique.")

    session_config = SessionConfig(**session_data)
    audio_config = AudioConfig(**audio_data)
    transcription_config = TranscriptionConfig(**transcription_data)
    progress_config = ProgressConfig(**progress_data)
    workflow_config = WorkflowConfig(**workflow_data)

    glossary_sources = data.get("glossary_sources") or []

    # Resolve paths for glossary sources
    resolved_sources = []
    manifest_root = path.parent
    for src in glossary_sources:
        src_path = Path(src)
        if not src_path.is_absolute():
            src_path = manifest_root / src_path
        resolved_sources.append(src_path.resolve())

    # Build consolidated glossary entries
    entries: dict[str, list[GlossaryEntry]] = {}

    def add_items(category: str, items: list) -> None:
        if not isinstance(items, list):
            return
        if category not in entries:
            entries[category] = []
        for item in items:
            if isinstance(item, str):
                entries[category].append(
                    GlossaryEntry(canonical=item, type=category)
                )
            elif isinstance(item, dict):
                canonical = item.get("canonical")
                if not canonical:
                    continue
                entries[category].append(
                    GlossaryEntry(
                        canonical=canonical,
                        type=item.get("type", category),
                        aliases=item.get("aliases") or [],
                        description=item.get("description"),
                        related_to=item.get("related_to") or [],
                    )
                )

    # Process manifest's own glossary if it exists
    if "glossary" in data:
        own_glossary = data["glossary"] or {}
        if isinstance(own_glossary, dict):
            for cat, items in own_glossary.items():
                add_items(cat, items)

    # Process glossary source files
    for src_path in resolved_sources:
        if src_path.exists():
            try:
                with src_path.open("r", encoding="utf-8") as f:
                    file_data = yaml.safe_load(f) or {}
                    glossary_data = file_data.get("glossary", file_data) or {}
                    if isinstance(glossary_data, dict):
                        for cat, items in glossary_data.items():
                            add_items(cat, items)
            except Exception:
                # If a glossary file is empty or invalid, skip it
                pass

    glossary_config = GlossaryConfig(entries=entries)

    filters_data = data.get("glossary_candidate_filters") or {}
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(GlossaryCandidateFiltersConfig)}
    safe_filters = {k: v for k, v in filters_data.items() if k in field_names}
    if "custom_noise_terms" in safe_filters and not isinstance(safe_filters["custom_noise_terms"], list):
        safe_filters["custom_noise_terms"] = []
    glossary_candidate_filters = GlossaryCandidateFiltersConfig(**safe_filters)

    deduplication_data = data.get("deduplication") or {}
    deduplication_config = DeduplicationConfig(
        enabled=bool(deduplication_data.get("enabled", False))
    )

    completeness_data = data.get("completeness") or {}
    completeness_config = CompletenessConfig(
        max_missing_seconds_warn=float(completeness_data.get("max_missing_seconds_warn", 30.0)),
        fail_on_missing_coverage=bool(completeness_data.get("fail_on_missing_coverage", False)),
    )

    return SessionManifest(
        path=path,
        session=session_config,
        audio=audio_config,
        transcription=transcription_config,
        progress=progress_config,
        speakers=speakers,
        glossary_sources=glossary_sources,
        workflow=workflow_config,
        glossary=glossary_config,
        glossary_candidate_filters=glossary_candidate_filters,
        deduplication=deduplication_config,
        completeness=completeness_config,
    )
