from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path

from src.config.manifest import SpeakerConfig
from src.providers.base import ProviderContext


class WhisperXProvider:
    def __init__(self, context: ProviderContext):
        self.context = context
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        whisperx = None
        if self.context.backend != "faster-whisper":
            try:
                import whisperx
            except ImportError:
                whisperx = None

        if self.context.backend == "whisperx" and whisperx is None:
            raise RuntimeError(
                "WhisperX backend requires local whisperx. Install it with "
                "`pip install whisperx`, or use backend: faster-whisper."
            )

        if whisperx is not None:
            self._model = ("whisperx", whisperx.load_model(self.context.model, self.context.device))
            return self._model

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper backend requires local faster-whisper. "
                "Install it with `pip install faster-whisper` or `pip install -r requirements.txt`."
            ) from exc

        compute_type = _compute_type(self.context.device)
        self._model = (
            "faster-whisper",
            WhisperModel(self.context.model, device=self.context.device, compute_type=compute_type),
        )
        return self._model

    def transcribe(self, audio_path: Path, speaker: SpeakerConfig) -> dict:
        kind, model = self._load_model()
        if kind == "whisperx":
            result = model.transcribe(str(audio_path), language=self.context.language)
            return {
                "provider": "whisperx",
                "model": self.context.model,
                "speaker_id": speaker.speaker_id,
                "audio_path": str(audio_path),
                "result": result,
            }

        segments, info = model.transcribe(
            str(audio_path),
            language=self.context.language,
            word_timestamps=True,
        )
        return {
            "provider": "faster-whisper",
            "model": self.context.model,
            "speaker_id": speaker.speaker_id,
            "audio_path": str(audio_path),
            "language": getattr(info, "language", None),
            "duration": getattr(info, "duration", None),
            "result": {"segments": [_json_safe(segment) for segment in segments]},
        }


def _compute_type(device: str) -> str:
    return "float16" if device == "cuda" else "int8"


def _json_safe(value):
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if hasattr(value, "_asdict"):
        return _json_safe(value._asdict())
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return {
            key: _json_safe(item)
            for key, item in value.__dict__.items()
            if not key.startswith("_")
        }
    return str(value)
