from __future__ import annotations


def merge_turns(turns: list[dict]) -> list[dict]:
    return sorted(
        turns,
        key=lambda turn: (
            float(turn.get("start_seconds") or 0),
            float(turn.get("end_seconds") or 0),
            turn.get("speaker_id") or "",
        ),
    )
