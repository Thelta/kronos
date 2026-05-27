from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class YoloMergedDatasetConfig:
    output_dir: str
    synthetic_images_dir: str
    synthetic_labels_dir: str
    reviewed_real_dir: str
    review_status_file: str
    class_names: list[str] = field(default_factory=lambda: ["portrait"])
    train_split: float = 0.8
    seed: int = 7
    include_review_statuses: list[str] = field(default_factory=lambda: ["accepted", "rejected"])
    synthetic_image_patterns: list[str] = field(default_factory=lambda: ["*.jpg", "*.png"])
    reviewed_real_image_patterns: list[str] = field(default_factory=lambda: ["*_orig.jpg"])


@dataclass
class YoloTrainerConfig:
    weights: str = "yolo26n.pt"
    epochs: int = 100
    imgsz: int = 1280
    batch: int = 2
    patience: int = 15
    device: str = "cuda"
    project: str = "training_runs/yolo"
    name: str = "portrait_merged_yolo26n"
    save: bool = True
    plots: bool = True
    exist_ok: bool = True
    verbose: bool = True


@dataclass
class YoloTrainConfig:
    dataset: YoloMergedDatasetConfig
    trainer: YoloTrainerConfig = field(default_factory=YoloTrainerConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_override_value(raw_value: str) -> Any:
    lowered = raw_value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    try:
        if "." in raw_value:
            return float(raw_value)
        return int(raw_value)
    except ValueError:
        return raw_value


def apply_overrides(payload: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    for override in overrides:
        key_path, raw_value = override.split("=", maxsplit=1)
        keys = key_path.split(".")
        node = payload
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = parse_override_value(raw_value)
    return payload


def config_from_dict(payload: dict[str, Any]) -> YoloTrainConfig:
    return YoloTrainConfig(
        dataset=YoloMergedDatasetConfig(**payload["dataset"]),
        trainer=YoloTrainerConfig(**payload.get("trainer", {})),
    )


def load_config(config_path: Path, overrides: list[str]) -> YoloTrainConfig:
    payload = apply_overrides(load_json(config_path), overrides)
    return config_from_dict(payload)
