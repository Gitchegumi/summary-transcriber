from __future__ import annotations

from src.config.manifest import GlossaryConfig


def glossary_terms(glossary: GlossaryConfig) -> list[str]:
    return glossary.all_terms()
