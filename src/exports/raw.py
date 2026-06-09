from __future__ import annotations

import json
from pathlib import Path


def export_raw(raw_dir: Path, backend: str, speaker_id: str, payload: dict) -> Path:
    target_dir = raw_dir / backend
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{speaker_id}.raw.json"
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path
