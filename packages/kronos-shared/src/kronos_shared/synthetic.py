from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class SyntheticClassificationRow:
    image_path: str
    subset: str
    mode: str
    character_id: str
    portrait_path: str
    attack_type: str
    role: str
    level: int
    star_value: int
    star_color: str
    assist: bool
    seed: int
    background_kind: str
    scale_jitter: float
    card_box: list[int]
    portrait_box: list[int]
    star_box: list[int] | None = None
    source_split: str | None = None
    obstructions: list[dict[str, Any]] = field(default_factory=list)
    quality_policy: dict[str, Any] = field(default_factory=dict)
    global_effects: list[str] = field(default_factory=list)
    starter: bool = False
    leader: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SyntheticClassificationRow":
        normalized = dict(payload)
        if "subset" not in normalized:
            normalized["subset"] = str(normalized.get("split", ""))
        if "source_split" not in normalized and "split" in normalized:
            normalized["source_split"] = str(normalized["split"])
        normalized.setdefault("leader", False)
        normalized.pop("split", None)
        return cls(**normalized)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_synthetic_rows(path: Path) -> list[SyntheticClassificationRow]:
    rows: list[SyntheticClassificationRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(SyntheticClassificationRow.from_dict(json.loads(line)))
    return rows


def save_synthetic_rows(path: Path, rows: list[SyntheticClassificationRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
