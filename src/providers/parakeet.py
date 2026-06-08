from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config.manifest import SpeakerConfig
from src.providers.base import ProviderContext, transcribe_nemo_chunks


class ParakeetProvider:
    def __init__(self, context: ProviderContext):
        self.context = context
        self._model = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from nemo.collections.asr.models import ASRModel
        except ImportError as exc:
            raise RuntimeError(
                "Parakeet requires local NVIDIA NeMo ASR packages. "
                "Install the optional local dependencies documented in README.md."
            ) from exc

        model = ASRModel.from_pretrained(self.context.model)
        if self.context.device:
            model = model.to(self.context.device)
        self._model = model
        return model

    def transcribe(
        self,
        audio_path: Path,
        speaker: SpeakerConfig,
        chunks: list[dict] | None = None,
        resume: bool = False,
        output_dir: Path | None = None,
        progress_callback: Any = None,
        sample_rate: int = 16000,
        chunk_format: str = "wav",
        overlap_seconds: float = 2.0,
    ) -> dict:
        model = self._load_model()
        if chunks is not None and output_dir is not None:
            return transcribe_nemo_chunks(
                provider_name="parakeet",
                model=model,
                audio_path=audio_path,
                speaker=speaker,
                chunks=chunks,
                resume=resume,
                output_dir=output_dir,
                continue_on_error=True,
                sample_rate=sample_rate,
                chunk_format=chunk_format,
                overlap_seconds=overlap_seconds,
                progress_callback=progress_callback,
            )

        # Fallback to full file if no chunks specified
        result = model.transcribe([str(audio_path)], timestamps=True)
        item = result[0] if isinstance(result, list) else result
        from src.providers.base import _json_safe
        return {
            "provider": "parakeet",
            "model": self.context.model,
            "speaker_id": speaker.speaker_id,
            "audio_path": str(audio_path),
            "result": _json_safe(item),
        }
