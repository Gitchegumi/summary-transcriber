from __future__ import annotations

from pathlib import Path

from src.config.manifest import SpeakerConfig
from src.providers.base import ProviderContext
from src.providers.parakeet import _json_safe


class CanaryProvider:
    def __init__(self, context: ProviderContext):
        self.context = context
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from nemo.collections.asr.models import ASRModel
        except ImportError as exc:
            raise RuntimeError(
                "Canary requires local NVIDIA NeMo ASR packages. "
                "Install the optional local dependencies documented in README.md."
            ) from exc

        model = ASRModel.from_pretrained(self.context.model)
        if self.context.device:
            model = model.to(self.context.device)
        self._model = model
        return model

    def transcribe(self, audio_path: Path, speaker: SpeakerConfig) -> dict:
        model = self._load_model()
        result = model.transcribe([str(audio_path)], timestamps=True)
        item = result[0] if isinstance(result, list) else result
        return {
            "provider": "canary",
            "model": self.context.model,
            "speaker_id": speaker.speaker_id,
            "audio_path": str(audio_path),
            "result": _json_safe(item),
        }
