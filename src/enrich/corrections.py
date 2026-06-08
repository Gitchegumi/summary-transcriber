from __future__ import annotations

import re

from src.config.manifest import GlossaryConfig


DEFAULT_CORRECTIONS = {
    "water deep": "Waterdeep",
    "counter spell": "Counterspell",
    "deck save": "Dex save",
    "inside check": "Insight check",
}


def apply_corrections(turns: list[dict], glossary: GlossaryConfig) -> tuple[list[dict], list[dict]]:
    cleaned_turns = []
    corrections = []
    patterns = dict(DEFAULT_CORRECTIONS)

    # Prioritize glossary terms. Map both canonical and aliases.
    if glossary and hasattr(glossary, "entries"):
        for entries_list in glossary.entries.values():
            for entry in entries_list:
                if entry.canonical:
                    patterns[entry.canonical.lower()] = entry.canonical
                    for alias in entry.aliases:
                        if alias:
                            patterns[alias.lower()] = entry.canonical

    for turn in turns:
        updated = dict(turn)
        text = turn["text_raw"]
        cleaned = text
        turn_corrections = []
        for original, corrected in patterns.items():
            pattern = re.compile(rf"\b{re.escape(original)}\b", flags=re.IGNORECASE)
            if pattern.search(cleaned) and original.lower() != corrected.lower():
                before = cleaned
                cleaned = pattern.sub(corrected, cleaned)
                turn_corrections.append((before, cleaned, original, corrected))

        updated["text_cleaned"] = cleaned
        cleaned_turns.append(updated)
        for index, (before, after, original, corrected) in enumerate(turn_corrections, start=1):
            corrections.append(
                {
                    "session_id": turn["session_id"],
                    "turn_id": turn["turn_id"],
                    "correction_id": f"{turn['turn_id']}-correction-{index:03d}",
                    "original_text": before,
                    "corrected_text": after,
                    "correction_type": "glossary_rule",
                    "reason": f"Rule-based replacement: {original} -> {corrected}",
                    "confidence": 0.9,
                }
            )
    return cleaned_turns, corrections
