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
from src.enrich.glossary_candidates import (
    ScoringWeights,
    build_glossary_candidates,
    score_candidate,
)
from src.exports.agents import _agent_payloads, _turn_payload
from src.exports.glossary import export_draft_outputs


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

    def test_agent_turn_payload_is_minimal_and_character_attributed(self) -> None:
        turn = {
            "turn_id": "turn-1",
            "speaker_id": "kibiw",
            "speaker_name": "Player 1",
            "character_name": "Kibiw",
            "start_seconds": 65.8,
            "end_seconds": 68.125,
            "text_cleaned": "  We enter the dungeon.  ",
        }
        self.assertEqual(
            _turn_payload(self.manifest, turn),
            {
                "start_time": "00:01:05.800",
                "end_time": "00:01:08.125",
                "speaker": "Kibiw",
                "text": "We enter the dungeon.",
            },
        )

    def test_agent_payloads_merge_nearby_consecutive_same_speaker(self) -> None:
        turns = [
            {"turn_id": "1", "speaker_id": "dm", "speaker_name": "James", "character_name": None, "start_seconds": 418.0, "end_seconds": 418.64, "text_cleaned": "Yep."},
            {"turn_id": "2", "speaker_id": "dm", "speaker_name": "James", "character_name": None, "start_seconds": 419.6, "end_seconds": 421.6, "text_cleaned": "No, I don't think she should have had it though."},
            {"turn_id": "3", "speaker_id": "dm", "speaker_name": "James", "character_name": None, "start_seconds": 426.96, "end_seconds": 428.32, "text_cleaned": "Yeah, okay, good call."},
            {"turn_id": "4", "speaker_id": "dm", "speaker_name": "James", "character_name": None, "start_seconds": 428.48, "end_seconds": 436.88, "text_cleaned": "Um nope, stop, god damn it."},
        ]

        payloads = _agent_payloads(self.manifest, turns)

        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0]["start_time"], "00:06:58.000")
        self.assertEqual(payloads[0]["end_time"], "00:07:16.880")
        self.assertEqual(
            payloads[0]["text"],
            "Yep. No, I don't think she should have had it though. "
            "Yeah, okay, good call. Um nope, stop, god damn it.",
        )

    def test_agent_payloads_do_not_merge_across_speaker_or_long_pause(self) -> None:
        turns = [
            {"turn_id": "1", "speaker_id": "dm", "speaker_name": "James", "character_name": None, "start_seconds": 0.0, "end_seconds": 1.0, "text_cleaned": "First."},
            {"turn_id": "2", "speaker_id": "kibiw", "speaker_name": "Player 1", "character_name": "Kibiw", "start_seconds": 1.5, "end_seconds": 2.0, "text_cleaned": "Interjection."},
            {"turn_id": "3", "speaker_id": "dm", "speaker_name": "James", "character_name": None, "start_seconds": 2.5, "end_seconds": 3.0, "text_cleaned": "Second."},
            {"turn_id": "4", "speaker_id": "dm", "speaker_name": "James", "character_name": None, "start_seconds": 14.0, "end_seconds": 15.0, "text_cleaned": "After a pause."},
        ]

        self.assertEqual(len(_agent_payloads(self.manifest, turns)), 4)
    def test_agent_turn_payload_labels_dm_and_omits_empty_turns(self) -> None:
        dm_turn = {
            "turn_id": "turn-2",
            "speaker_id": "dm",
            "speaker_name": "James",
            "character_name": None,
            "start_seconds": 0.0,
            "end_seconds": 4.25,
            "text_cleaned": "The door opens.",
        }
        self.assertEqual(_turn_payload(self.manifest, dm_turn)["speaker"], "DM")
        dm_turn["text_cleaned"] = "  "
        self.assertIsNone(_turn_payload(self.manifest, dm_turn))
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

    def test_normalize_parakeet_27_timestamp_segments(self) -> None:
        raw_output = {
            "chunks": [{
                "chunk_id": "dm_chunk_00001",
                "chunk_start_seconds": 58.0,
                "chunk_end_seconds": 118.0,
                "raw_result": {
                    "text": "First sentence. Second sentence.",
                    "timestamp": {
                        "segment": [
                            {"segment": "First sentence.", "start_offset": 1, "end_offset": 50, "start": 0.08, "end": 4.0},
                            {"segment": "Second sentence.", "start_offset": 82, "end_offset": 91, "start": 6.56, "end": 7.28},
                        ],
                        "word": [],
                    },
                },
            }],
        }

        turns = normalize_turns(
            manifest=self.manifest,
            speaker=self.speaker1,
            raw_output=raw_output,
            backend="parakeet",
            model="nvidia/parakeet-tdt-0.6b-v3",
        )

        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[0]["text_raw"], "First sentence.")
        self.assertAlmostEqual(turns[0]["start_seconds"], 58.08)
        self.assertAlmostEqual(turns[0]["end_seconds"], 62.0)
        self.assertEqual(turns[1]["text_raw"], "Second sentence.")
        self.assertAlmostEqual(turns[1]["start_seconds"], 64.56)
        self.assertAlmostEqual(turns[1]["end_seconds"], 65.28)
    def test_normalize_untimed_parakeet_text_uses_chunk_interval(self) -> None:
        raw_output = {
            "chunks": [{
                "chunk_id": "dm_chunk_00001",
                "chunk_start_seconds": 58.0,
                "chunk_end_seconds": 118.0,
                "raw_result": {
                    "text": "Speech without internal timestamps.",
                    "timestep": {"segment": []},
                },
            }],
        }

        turns = normalize_turns(
            manifest=self.manifest,
            speaker=self.speaker1,
            raw_output=raw_output,
            backend="parakeet",
            model="nvidia/parakeet-tdt-0.6b-v3",
        )

        self.assertEqual(turns[0]["start_seconds"], 58.0)
        self.assertEqual(turns[0]["end_seconds"], 118.0)
        self.assertEqual(turns[0]["duration_seconds"], 60.0)
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

    def test_apply_corrections_prefers_full_aliases_and_avoids_canonical_expansion(self) -> None:
        glossary = GlossaryConfig(entries={
            "npcs": [GlossaryEntry(
                canonical="Lady Saris",
                aliases=["Saris", "Ceres", "Lady Ceres"],
            )],
            "rules": [GlossaryEntry(
                canonical="Slight of Hand",
                aliases=["Slight"],
            )],
        })
        turns = [{
            "session_id": "test-session",
            "turn_id": "turn-accuracy",
            "text_raw": "Lady Ceres made a Slight of Hand check.",
        }]

        cleaned, _ = apply_corrections(turns, glossary)

        self.assertEqual(
            cleaned[0]["text_cleaned"],
            "Lady Saris made a Slight of Hand check.",
        )
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
        self.assertEqual(dm_rep["total_missing_seconds"], 0)
        self.assertEqual(dm_rep["estimated_silence_seconds"], 60.0)
        self.assertEqual(report["completeness_summary"]["status"], "complete")

    def test_completeness_treats_empty_successful_chunks_as_silence(self) -> None:
        turns = [{"speaker_id": "dm", "start_seconds": 10.0, "end_seconds": 20.0}]
        audio_reports = {"dm": {"duration_seconds": 100.0}, "kibiw": {"duration_seconds": 100.0}}
        chunks = [
            {"chunk_id": "chunk-1", "chunk_start_seconds": 0.0, "chunk_end_seconds": 60.0, "raw_result": {"text": "speech"}},
            {"chunk_id": "chunk-2", "chunk_start_seconds": 58.0, "chunk_end_seconds": 100.0, "raw_result": {"text": ""}},
        ]
        raw_outputs = {
            "dm": {"chunks": chunks, "failed_chunks": []},
            "kibiw": {"chunks": chunks, "failed_chunks": []},
        }

        report = compute_completeness_report(self.manifest, turns, audio_reports, raw_outputs)

        self.assertEqual(report["completeness_summary"]["status"], "complete")
        self.assertEqual(report["completeness_summary"]["total_missing_seconds"], 0)
        self.assertEqual(report["speakers"]["dm"]["chunks_empty"], 1)
        self.assertEqual(report["speakers"]["dm"]["estimated_silence_seconds"], 90.0)

    def test_completeness_warns_for_failed_chunk_processing_gap(self) -> None:
        self.manifest.speakers = [self.speaker1]
        audio_reports = {"dm": {"duration_seconds": 100.0}}
        raw_outputs = {"dm": {
            "chunks": [
                {"chunk_id": "chunk-1", "chunk_start_seconds": 0.0, "chunk_end_seconds": 60.0, "raw_result": {"text": "speech"}},
                {"chunk_id": "chunk-2", "chunk_start_seconds": 58.0, "chunk_end_seconds": 100.0, "raw_result": {"text": ""}},
            ],
            "failed_chunks": [{"chunk_id": "chunk-2", "start_seconds": 58.0, "end_seconds": 100.0}],
        }}

        report = compute_completeness_report(self.manifest, [], audio_reports, raw_outputs)

        dm_report = report["speakers"]["dm"]
        self.assertEqual(dm_report["chunks_expected"], 2)
        self.assertEqual(dm_report["chunks_successfully_transcribed"], 1)
        self.assertEqual(dm_report["chunks_failed"], 1)
        self.assertEqual(dm_report["missing_ranges"], [[60.0, 100.0]])
        self.assertEqual(dm_report["total_missing_seconds"], 40.0)
        self.assertEqual(report["completeness_summary"]["status"], "complete_with_warnings")

    def test_completeness_subtracts_retry_failure_from_parent_coverage(self) -> None:
        self.manifest.speakers = [self.speaker1]
        audio_reports = {"dm": {"duration_seconds": 60.0}}
        raw_outputs = {"dm": {
            "chunks": [{
                "chunk_id": "dm_chunk_00001",
                "chunk_start_seconds": 0.0,
                "chunk_end_seconds": 60.0,
                "raw_result": {"text": "partially transcribed"},
            }],
            "failed_chunks": [{
                "chunk_id": "dm_chunk_00001_retry_30s_00_retry_15s_00",
                "start_seconds": 0.0,
                "end_seconds": 15.0,
            }],
        }}

        report = compute_completeness_report(
            self.manifest, [], audio_reports, raw_outputs
        )

        dm_report = report["speakers"]["dm"]
        self.assertEqual(dm_report["chunks_failed"], 1)
        self.assertEqual(dm_report["covered_ranges"], [[15.0, 60.0]])
        self.assertEqual(dm_report["missing_ranges"], [[0.0, 15.0]])
        self.assertEqual(dm_report["total_missing_seconds"], 15.0)
        self.assertEqual(
            report["completeness_summary"]["status"], "complete_with_warnings"
        )

    def test_universal_table_chatter_is_stopword_noise(self) -> None:
        turns = []
        for index, term in enumerate(("Hang", "Look", "Hold", "Definitely"), start=1):
            turns.append({
                "turn_id": f"turn-{index}",
                "speaker_id": "dm",
                "text_cleaned": f"{term} on a moment.",
                "start_seconds": float(index),
                "end_seconds": float(index + 1),
            })

        candidates, noise = build_glossary_candidates(
            self.manifest, turns, words=[], corrections=[]
        )

        candidate_terms = {row["term"].lower() for row in candidates}
        noise_terms = {row["term"].lower() for row in noise}
        for term in ("hang", "look", "hold", "definitely"):
            self.assertNotIn(term, candidate_terms)
            self.assertIn(term, noise_terms)
    def test_glossary_candidates_exclude_components_of_known_terms(self) -> None:
        self.manifest.glossary.entries["spells"] = [
            GlossaryEntry(canonical="Hunger of Hadar", type="spell")
        ]
        self.manifest.glossary.entries["rules"] = [
            GlossaryEntry(canonical="Battle Medic", type="term")
        ]
        turns = [
            {
                "turn_id": "turn-hunger",
                "speaker_id": "dm",
                "text_cleaned": "The Hunger of Hadar filled the room.",
                "start_seconds": 1.0,
                "end_seconds": 2.0,
            },
            {
                "turn_id": "turn-battle",
                "speaker_id": "dm",
                "text_cleaned": "Battle Medic takes one action.",
                "start_seconds": 3.0,
                "end_seconds": 4.0,
            },
        ]

        candidates, noise = build_glossary_candidates(
            self.manifest, turns, words=[], corrections=[]
        )

        emitted_terms = {row["term"].lower() for row in candidates + noise}
        self.assertNotIn("hunger", emitted_terms)
        self.assertNotIn("battle", emitted_terms)
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

    def test_draft_export_removes_redundant_candidate_yaml(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            draft_dir = output_dir / "draft"
            draft_dir.mkdir()
            legacy = draft_dir / "glossary_candidates.yaml"
            legacy.write_text("legacy: true\n", encoding="utf-8")

            export_draft_outputs(output_dir, candidates=[], turns=[], words=[])

            self.assertFalse(legacy.exists())
            self.assertTrue((draft_dir / "unknown_terms_report.csv").exists())
            self.assertTrue((draft_dir / "candidate_glossary.yaml").exists())
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
            "dm": {
                "chunks": [{
                    "chunk_id": "chunk-1",
                    "chunk_start_seconds": 0.0,
                    "chunk_end_seconds": 152.0,
                    "raw_result": {"text": "dummy text"},
                }],
                "failed_chunks": [],
            },
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
