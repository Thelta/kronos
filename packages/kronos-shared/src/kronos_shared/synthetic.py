from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class SyntheticClassificationRow:
    image_path: str
    subset: str
    mode: str
    character_id: str
    portrait_path: str | None
    attack_type: str | None
    role: str | None
    level: int | None
    star_value: int | None
    star_color: str | None
    assist: bool
    seed: int
    background_kind: str
    card_box: list[int]
    portrait_box: list[int]
    scale_jitter: float = 1.0
    star_box: list[int] | None = None
    source_split: str | None = None
    obstructions: list[dict[str, Any]] = field(default_factory=list)
    quality_policy: dict[str, Any] = field(default_factory=dict)
    global_effects: list[str] = field(default_factory=list)
    starter: bool = False
    leader: bool = False
    empty: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SkillClassificationRow:
    image_path: str
    subset: str
    mode: str
    character_id: str | None
    assist: bool
    seed: int
    background_kind: str
    card_box: list[int]
    portrait_box: list[int]
    skill_card_id: str | None = None
    portrait_path: str | None = None
    attack_type: str | None = None
    canonical_attack_type: str | None = None
    render_attack_type: str | None = None
    role: str | None = None
    level: int | None = None
    star_value: int | None = None
    star_color: str | None = None
    star_box: list[int] | None = None
    source_split: str | None = None
    obstructions: list[dict[str, Any]] = field(default_factory=list)
    quality_policy: dict[str, Any] = field(default_factory=dict)
    global_effects: list[str] = field(default_factory=list)
    starter: bool = False
    leader: bool = False
    empty: bool = False
    render_mode: str | None = None
    card_source: str | None = None
    costume_group_id: int | None = None

    @property
    def identity_key(self) -> str:
        return self.skill_card_id or self.character_id or "empty"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    if "subset" not in normalized:
        normalized["subset"] = str(normalized.get("split", ""))
    if "source_split" not in normalized and "split" in normalized:
        normalized["source_split"] = str(normalized["split"])
    normalized.setdefault("leader", False)
    normalized.setdefault("starter", False)
    normalized.setdefault("assist", False)
    normalized.setdefault("empty", False)
    normalized.pop("split", None)
    return normalized


def _filter_payload(cls: type[Any], payload: dict[str, Any]) -> dict[str, Any]:
    allowed = {item.name for item in fields(cls)}
    return {key: value for key, value in payload.items() if key in allowed}


def synthetic_row_from_dict(payload: dict[str, Any]) -> SyntheticClassificationRow:
    normalized = _filter_payload(SyntheticClassificationRow, _normalize_payload(payload))
    return SyntheticClassificationRow(**normalized)


def skill_row_from_dict(payload: dict[str, Any]) -> SkillClassificationRow:
    normalized = _filter_payload(SkillClassificationRow, _normalize_payload(payload))
    return SkillClassificationRow(**normalized)


def load_synthetic_rows(path: Path) -> list[SyntheticClassificationRow]:
    rows: list[SyntheticClassificationRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(synthetic_row_from_dict(json.loads(line)))
    return rows


def load_skill_rows(path: Path) -> list[SkillClassificationRow]:
    rows: list[SkillClassificationRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(skill_row_from_dict(json.loads(line)))
    return rows


def save_synthetic_rows(path: Path, rows: list[SyntheticClassificationRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")


def save_skill_rows(path: Path, rows: list[SkillClassificationRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
