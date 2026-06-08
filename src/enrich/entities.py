from __future__ import annotations

import re

from src.config.manifest import GlossaryConfig


def extract_entities(turns: list[dict], glossary: GlossaryConfig) -> list[dict]:
    rows = []
    term_types = {
        "pc": glossary.pcs,
        "npc": glossary.npcs,
        "location": glossary.locations,
        "faction": glossary.factions,
        "item": glossary.items,
        "spell": glossary.spells,
        "rules_term": glossary.rules_terms,
        "custom_term": glossary.custom_terms,
    }
    for turn in turns:
        text = turn["text_cleaned"]
        for entity_type, terms in term_types.items():
            for term in terms:
                if re.search(rf"\b{re.escape(term)}\b", text, flags=re.IGNORECASE):
                    rows.append(
                        {
                            "session_id": turn["session_id"],
                            "turn_id": turn["turn_id"],
                            "entity_id": f"{turn['turn_id']}-{entity_type}-{len(rows) + 1:05d}",
                            "entity_text": term,
                            "entity_type": entity_type,
                            "start_seconds": turn["start_seconds"],
                            "end_seconds": turn["end_seconds"],
                            "source": "glossary_match",
                        }
                    )
    return rows
