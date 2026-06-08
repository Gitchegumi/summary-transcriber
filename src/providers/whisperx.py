from __future__ import annotations

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
        try:
            import whisperx
        except ImportError:
            whisperx = None

        if whisperx is not None:
            self._model = ("whisperx", whisperx.load_model(self.context.model, self.context.device))
            return self._model

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "WhisperX fallback requires local whisperx or faster-whisper. "
                "Install the optional local dependencies documented in README.md."
            ) from exc

        compute_type = "float16" if self.context.device == "cuda" else "int8"
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

        segments, info = model.transcribe(str(audio_path), language=self.context.language)
        return {
            "provider": "faster-whisper",
            "model": self.context.model,
            "speaker_id": speaker.speaker_id,
            "audio_path": str(audio_path),
            "language": getattr(info, "language", None),
            "duration": getattr(info, "duration", None),
            "result": {"segments": [segment._asdict() for segment in segments]},
        }
