from __future__ import annotations

import unittest
from pathlib import Path

from src.config.manifest import (
    SessionManifest,
    SessionConfig,
    AudioConfig,
    TranscriptionConfig,
    ProgressConfig,
    SpeakerConfig,
    GlossaryConfig,
    GlossaryEntry,
    load_manifest,
)
from src.normalize.turns import normalize_turns
from src.enrich.corrections import apply_corrections
from src.enrich.completeness import compute_completeness_report
from src.enrich.glossary_candidates import score_candidate, ScoringWeights


class TestV2Core(unittest.TestCase):
    """Unit tests covering core components of the v2 transcription pipeline."""

    def setUp(self) -> None:
        # Create a mock manifest config
        session = SessionConfig(
            id="test-session",
            campaign="Test Campaign",
            session_number=1,
            session_date="2026-06-09",
        )
        audio = AudioConfig()
        transcription = TranscriptionConfig()
        progress = ProgressConfig()

        self.speaker1 = SpeakerConfig(
            speaker_id="dm",
            display_name="James",
            role="DM",
            character_name=None,
            file="audio/dm.flac",
        )
        self.speaker2 = SpeakerConfig(
            speaker_id="kibiw",
            display_name="Player 1",
            role="Player",
            character_name="Kibiw",
            file="audio/player_1.flac",
        )

        self.manifest_path = Path("E:/GitHub/summary-transcriber/test-5/manifest.yaml")

        self.manifest = SessionManifest(
            path=self.manifest_path,
            session=session,
            audio=audio,
            transcription=transcription,
            progress=progress,
            speakers=[self.speaker1, self.speaker2],
            glossary=GlossaryConfig(
                entries={
                    "pcs": [GlossaryEntry(canonical="Kibiw", type="pc", aliases=["Kibew"])]
                }
            ),
        )

    def test_normalize_turns_non_chunked(self) -> None:
        """Test normalize_turns handles Whisper-style non-chunked input correctly."""
        raw_output = {
            "result": {
                "segments": [
                    {
                        "start": 1.5,
                        "end": 4.5,
                        "text": "Hello world from DM.",
                        "confidence": 0.95,
                    }
                ]
            }
        }

        turns = normalize_turns(
            manifest=self.manifest,
            speaker=self.speaker1,
            raw_output=raw_output,
            backend="whisperx",
            model="large-v3",
        )

        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0]["text_raw"], "Hello world from DM.")
        self.assertEqual(turns[0]["start_seconds"], 1.5)
        self.assertEqual(turns[0]["end_seconds"], 4.5)
        self.assertEqual(turns[0]["speaker_id"], "dm")
        self.assertEqual(turns[0]["backend"], "whisperx")

    def test_normalize_turns_chunked(self) -> None:
        """Test normalize_turns handles chunked ASR input and boundaries correctly."""
        raw_output = {
            "chunks": [
                {
                    "chunk_id": "dm_chunk_00000",
                    "chunk_start_seconds": 0.0,
                    "chunk_end_seconds": 60.0,
                    "raw_result": {
                        "timestep": {
                            "segment": [
                                {
                                    "text": "This is in chunk zero.",
                                    "start_offset": 5.0,
                                    "end_offset": 10.0,
                                }
                            ]
                        }
                    },
                },
                {
                    "chunk_id": "dm_chunk_00001",
                    "chunk_start_seconds": 58.0,
                    "chunk_end_seconds": 120.0,
                    "raw_result": {
                        "text": "Overlapping turn text.",
                        "timestep": {
                            "segment": [
                                {
                                    "text": "Overlapping turn text.",
                                    "start_offset": 2.0,
                                    "end_offset": 6.0,
                                }
                            ]
                        }
                    },
                },
            ]
        }

        turns = normalize_turns(
            manifest=self.manifest,
            speaker=self.speaker1,
            raw_output=raw_output,
            backend="parakeet",
            model="nvidia/parakeet-tdt-0.6b-v3",
        )

        # Expect 2 turns since overlap deduplication is disabled in mock setup
        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[0]["text_raw"], "This is in chunk zero.")
        self.assertEqual(turns[0]["start_seconds"], 5.0)
        self.assertEqual(turns[0]["end_seconds"], 10.0)
        self.assertEqual(turns[1]["text_raw"], "Overlapping turn text.")
        self.assertEqual(turns[1]["start_seconds"], 60.0)  # 58.0 + 2.0
        self.assertEqual(turns[1]["end_seconds"], 64.0)  # 58.0 + 6.0

    def test_apply_corrections(self) -> None:
        """Test apply_corrections resolves glossary aliases and common corrections."""
        turns = [
            {
                "session_id": "test-session",
                "turn_id": "turn-001",
                "speaker_id": "kibiw",
                "text_raw": "Hello Kibew, we are in water deep.",
                "text_cleaned": "Hello Kibew, we are in water deep.",
            }
        ]

        cleaned, corrections = apply_corrections(turns, self.manifest.glossary)

        self.assertEqual(len(cleaned), 1)
        # Kibew corrected to Kibiw, water deep corrected to Waterdeep
        self.assertEqual(cleaned[0]["text_cleaned"], "Hello Kibiw, we are in Waterdeep.")
        self.assertTrue(len(corrections) >= 2)

    def test_compute_completeness_report(self) -> None:
        """Test compute_completeness_report tracks coverage gaps and success rates."""
        turns = [
            {"speaker_id": "dm", "start_seconds": 10.0, "end_seconds": 50.0},
            {"speaker_id": "kibiw", "start_seconds": 5.0, "end_seconds": 55.0},
        ]
        audio_reports = {
            "dm": {"duration_seconds": 100.0},
            "kibiw": {"duration_seconds": 100.0},
        }

        speaker_raw_outputs = {
            "dm": {"result": {"text": "dummy text"}},
            "kibiw": {"result": {"text": "dummy text"}},
        }

        report = compute_completeness_report(
            manifest=self.manifest,
            turns=turns,
            audio_reports=audio_reports,
            speaker_raw_outputs=speaker_raw_outputs,
        )

        self.assertEqual(report["session_id"], "test-session")
        self.assertIn("dm", report["speakers"])
        dm_rep = report["speakers"]["dm"]
        self.assertEqual(dm_rep["chunks_expected"], 1)
        self.assertEqual(dm_rep["chunks_successfully_transcribed"], 1)
        self.assertGreater(dm_rep["total_missing_seconds"], 0)

    def test_score_candidate(self) -> None:
        """Test candidate scoring calculation matches expected weights configurations."""
        speaker_names = {"james", "player 1"}

        # Score term resembling known PC name
        score, pos, neg = score_candidate(
            term="Kibew",
            occurrence_count=2,
            speaker_ids={"kibiw"},
            turn_ids=["turn-001"],
            example_texts=["Kibew was running"],
            sentence_start_count=0,
            confidence_values=[0.9, 0.85],
            is_correction=False,
            speaker_names=speaker_names,
            manifest=self.manifest,
            weights=ScoringWeights(),
        )

        # Check: repeated_term (+2) + mid_sentence_cap (+3) + resembles_existing (+2) + short_utterance (-2) = 5
        self.assertEqual(score, 5)
        self.assertIn("repeated_term_occurrences_2 (+2)", pos)
        self.assertIn("resembles_existing_glossary_term (+2)", pos)

    def test_manifest_path_resolution_outside_root(self) -> None:
        """Test that resolve_path successfully resolves paths outside the manifest root."""
        # Using a relative path that goes up from E:/GitHub/summary-transcriber/test-5/manifest.yaml
        resolved = self.manifest.resolve_path("../audio/dm.flac")
        expected = Path("E:/GitHub/summary-transcriber/audio/dm.flac").resolve()
        self.assertEqual(resolved, expected)

    def test_load_manifest_outside_glossary(self) -> None:
        """Test that load_manifest resolves glossary sources that are outside manifest root."""
        import tempfile
        import yaml
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir).resolve()
            
            # Create a glossary file outside the manifest root
            glossary_dir = tmpdir_path / "glossaries"
            glossary_dir.mkdir()
            glossary_file = glossary_dir / "campaign.yaml"
            with glossary_file.open("w", encoding="utf-8") as f:
                yaml.dump({
                    "glossary": {
                        "pcs": [
                            {"canonical": "SharedHero", "type": "pc"}
                        ]
                    }
                }, f)
                
            # Create manifest root directory
            manifest_dir = tmpdir_path / "session-1"
            manifest_dir.mkdir()
            manifest_file = manifest_dir / "manifest.yaml"
            
            manifest_data = {
                "session": {
                    "id": "session-1",
                    "campaign": "Campaign",
                    "session_number": 1,
                    "session_date": "2026-06-09"
                },
                "speakers": [
                    {
                        "speaker_id": "dm",
                        "display_name": "James",
                        "role": "DM",
                        "character_name": None,
                        "file": "audio/dm.flac"
                    }
                ],
                "glossary_sources": [
                    "../glossaries/campaign.yaml"
                ]
            }
            
            with manifest_file.open("w", encoding="utf-8") as f:
                yaml.dump(manifest_data, f)
                
            # Load manifest should succeed and resolve the glossary source
            manifest = load_manifest(manifest_file)
            self.assertEqual(len(manifest.glossary_sources), 1)
            self.assertEqual(manifest.glossary_sources[0], "../glossaries/campaign.yaml")
            
            # Verify glossary entries were loaded
            self.assertIn("pcs", manifest.glossary.entries)
            pcs = manifest.glossary.entries["pcs"]
            self.assertEqual(len(pcs), 1)
            self.assertEqual(pcs[0].canonical, "SharedHero")

    def test_completeness_warning_threshold_default(self) -> None:
        """Test that compute_completeness_report uses the new default threshold of 500.0 seconds."""
        # Total missing is 348.0 seconds
        turns = [
            {"speaker_id": "dm", "start_seconds": 0.0, "end_seconds": 152.0},
        ]
        audio_reports = {
            "dm": {"duration_seconds": 500.0},
        }
        speaker_raw_outputs = {
            "dm": {"result": {"text": "dummy text"}},
        }
        
        # Since missing is 348.0 < 500.0, it should be status = 'complete_with_warnings' instead of 'incomplete'
        report = compute_completeness_report(
            manifest=self.manifest,
            turns=turns,
            audio_reports=audio_reports,
            speaker_raw_outputs=speaker_raw_outputs,
        )
        self.assertEqual(report["completeness_summary"]["status"], "complete_with_warnings")
        
        # If we configure the manifest with a lower threshold, e.g. 300.0 seconds, then 348.0 > 300.0, so 'incomplete'
        self.manifest.completeness.max_missing_seconds_warn = 300.0
        report2 = compute_completeness_report(
            manifest=self.manifest,
            turns=turns,
            audio_reports=audio_reports,
            speaker_raw_outputs=speaker_raw_outputs,
        )
        self.assertEqual(report2["completeness_summary"]["status"], "incomplete")
