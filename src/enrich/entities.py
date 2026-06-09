from __future__ import annotations

import re

from src.config.manifest import GlossaryConfig


def extract_entities(turns: list[dict], glossary: GlossaryConfig) -> list[dict]:
    rows = []
    if not glossary or not hasattr(glossary, "entries"):
        return rows

    for turn in turns:
        text = turn["text_cleaned"]
        for category, entries in glossary.entries.items():
            # Singularize or format category names to match standard NocoDB schema
            entity_type = category.rstrip("s")
            if entity_type == "rules_term":
                pass
            elif entity_type == "rules":
                entity_type = "rules_term"
            elif entity_type == "custom":
                entity_type = "custom_term"

            for entry in entries:
                if not entry.canonical:
                    continue
                
                # Check match for canonical name or aliases
                terms_to_check = [entry.canonical] + entry.aliases
                matched = False
                for term in terms_to_check:
                    if not term:
                        continue
                    if re.search(rf"\b{re.escape(term)}\b", text, flags=re.IGNORECASE):
                        matched = True
                        break
                
                if matched:
                    rows.append(
                        {
                            "session_id": turn["session_id"],
                            "turn_id": turn["turn_id"],
                            "entity_id": f"{turn['turn_id']}-{entity_type}-{len(rows) + 1:05d}",
                            "entity_text": entry.canonical,
                            "entity_type": entity_type,
                            "start_seconds": turn["start_seconds"],
                            "end_seconds": turn["end_seconds"],
                            "source": "glossary_match",
                        }
                    )
    return rows
