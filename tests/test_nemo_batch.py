from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from src.config.manifest import SpeakerConfig
from src.providers.nemo_batch import _duration, transcribe_nemo_batch


class RecordingModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, paths, **kwargs):
        self.calls.append((paths, kwargs))
        return [{"text": Path(path).stem} for path in paths]


class NemoBatchTests(TestCase):
    def test_sends_six_speaker_tracks_in_one_inference_batch(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            output_dir = root / "output"
            jobs = []
            for index in range(6):
                speaker_id = f"speaker-{index}"
                chunk_path = output_dir / "chunks" / speaker_id / f"{speaker_id}.wav"
                chunk_path.parent.mkdir(parents=True, exist_ok=True)
                chunk_path.touch()
                jobs.append({
                    "speaker": SpeakerConfig(speaker_id, speaker_id, "player", None, f"audio/{speaker_id}.wav"),
                    "audio_path": root / "audio" / f"{speaker_id}.wav",
                    "chunks": [{
                        "chunk_id": speaker_id,
                        "chunk_file": str(chunk_path),
                        "chunk_start_seconds": 0.0,
                        "chunk_end_seconds": 60.0,
                    }],
                })

            model = RecordingModel()
            with patch("src.providers.nemo_batch.configure_quiet_backend_logging"), patch("src.providers.nemo_batch.configure_nemo_logging"), patch("src.providers.nemo_batch.suppress_backend_output") as suppress:
                suppress.return_value.__enter__.return_value = None
                suppress.return_value.__exit__.return_value = None
                outputs = transcribe_nemo_batch("parakeet", model, jobs, output_dir)

            self.assertEqual(len(model.calls), 1)
            self.assertEqual(len(model.calls[0][0]), 6)
            self.assertEqual(model.calls[0][1]["batch_size"], 6)
            self.assertEqual(set(outputs), {f"speaker-{index}" for index in range(6)})
            self.assertTrue(all(len(output["chunks"]) == 1 for output in outputs.values()))

    def test_progress_duration_excludes_chunk_overlap(self):
        chunk = {
            "chunk_start_seconds": 58.0,
            "chunk_end_seconds": 120.0,
            "overlap_seconds": 2.0,
        }
        self.assertEqual(_duration(chunk), 60.0)
    def test_positive_batch_size_caps_concurrent_tracks(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            output_dir = root / "output"
            jobs = []
            for index in range(3):
                speaker_id = f"speaker-{index}"
                chunk_path = root / f"{speaker_id}.wav"
                chunk_path.touch()
                jobs.append({
                    "speaker": SpeakerConfig(speaker_id, speaker_id, "player", None, str(chunk_path)),
                    "audio_path": chunk_path,
                    "chunks": [{"chunk_id": speaker_id, "chunk_file": str(chunk_path), "chunk_start_seconds": 0.0, "chunk_end_seconds": 10.0}],
                })

            model = RecordingModel()
            with patch("src.providers.nemo_batch.configure_quiet_backend_logging"), patch("src.providers.nemo_batch.configure_nemo_logging"), patch("src.providers.nemo_batch.suppress_backend_output") as suppress:
                suppress.return_value.__enter__.return_value = None
                suppress.return_value.__exit__.return_value = None
                transcribe_nemo_batch("parakeet", model, jobs, output_dir, batch_size=2)

            self.assertEqual([len(call[0]) for call in model.calls], [2, 1])