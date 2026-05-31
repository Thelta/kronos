from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DataConfig:
    manifest_path: str
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 0
    train_subset: str = "train"
    val_subset: str = "val"
    test_subset: str = "test"
    gallery_subset: str = "gallery"
    gallery_count_per_identity: int = 2
    pin_memory: bool = True
    tiny_card_degrade_prob: float = 0.0
    tiny_card_size_jitter: float = 0.12
    tiny_card_sizes: list[list[int]] = field(
        default_factory=lambda: [[63, 50], [75, 60], [76, 59], [62, 58], [90, 72], [110, 88]]
    )


@dataclass
class ModelConfig:
    model_name: str = "mobilenetv4_conv_small.e2400_r224_in1k"
    pretrained: bool = True
    embedding_dim: int = 256
    star_roi: list[float] = field(default_factory=lambda: [0.0, 0.68, 0.38, 1.0])
    assist_roi: list[float] = field(default_factory=lambda: [0.68, 0.0, 1.0, 0.32])
    star_box_expand_x: float = 0.2
    star_box_expand_y: float = 0.35
    star_box_train_prob: float = 0.7
    star_box_jitter: float = 0.12
    roi_hidden_dim: int = 128
    roi_input_size: int = 192
    star_lowres_prob: float = 0.65
    star_lowres_min_size: int = 16
    star_lowres_max_size: int = 28
    star_sequence_bins: int = 10
    star_locator_bins: int = 16
    star_strip_min_width: float = 0.35
    star_strip_max_width: float = 0.85


@dataclass
class ArcFaceConfig:
    margin: float = 0.3
    scale: float = 32.0


@dataclass
class OptimizerConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-4
    min_lr: float = 1e-5


@dataclass
class LossConfig:
    identity_weight: float = 1.0
    empty_weight: float = 0.2
    star_weight: float = 0.3
    star_color_weight: float = 0.2
    assist_weight: float = 0.2


@dataclass
class TrainerConfig:
    epochs: int = 10
    mixed_precision: str = "fp16"
    output_dir: str = "training_runs/kronos_arcface"
    resume_from: str | None = None
    seed: int = 7
    patience: int = 0


@dataclass
class TrainConfig:
    data: DataConfig = field(default_factory=lambda: DataConfig(manifest_path=""))
    model: ModelConfig = field(default_factory=ModelConfig)
    arcface: ArcFaceConfig = field(default_factory=ArcFaceConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    losses: LossConfig = field(default_factory=LossConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

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


def load_config(config_path: Path, overrides: list[str]) -> TrainConfig:
    payload = apply_overrides(load_json(config_path), overrides)
    return config_from_dict(payload)


def config_from_dict(payload: dict[str, Any]) -> TrainConfig:
    return TrainConfig(
        data=DataConfig(**payload["data"]),
        model=ModelConfig(**payload["model"]),
        arcface=ArcFaceConfig(**payload["arcface"]),
        optimizer=OptimizerConfig(**payload["optimizer"]),
        losses=LossConfig(**payload["losses"]),
        trainer=TrainerConfig(**payload["trainer"]),
    )
