from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .training_config import ArcFaceConfig, OptimizerConfig, TrainerConfig, apply_overrides, load_json


@dataclass
class SkillDataConfig:
    manifest_path: str
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 0
    train_sampler: str = "random"
    train_subset: str = "train"
    val_subset: str = "val"
    test_subset: str = "test"
    gallery_subset: str = "gallery"
    gallery_count_per_identity: int = 2
    pin_memory: bool = True


@dataclass
class SkillModelConfig:
    model_name: str = "mobilenetv4_conv_small.e2400_r224_in1k"
    pretrained: bool = True
    embedding_dim: int = 256
    head_hidden_dim: int = 128


@dataclass
class SkillLossConfig:
    identity_weight: float = 1.0
    empty_weight: float = 0.2


@dataclass
class SkillTrainConfig:
    data: SkillDataConfig = field(default_factory=lambda: SkillDataConfig(manifest_path=""))
    model: SkillModelConfig = field(default_factory=SkillModelConfig)
    arcface: ArcFaceConfig = field(default_factory=ArcFaceConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    losses: SkillLossConfig = field(default_factory=SkillLossConfig)
    trainer: TrainerConfig = field(default_factory=lambda: TrainerConfig(output_dir="training_runs/kronos_skill_arcface"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_skill_config(config_path: Path, overrides: list[str]) -> SkillTrainConfig:
    payload = apply_overrides(load_json(config_path), overrides)
    return skill_config_from_dict(payload)


def skill_config_from_dict(payload: dict[str, Any]) -> SkillTrainConfig:
    return SkillTrainConfig(
        data=SkillDataConfig(**payload["data"]),
        model=SkillModelConfig(**payload["model"]),
        arcface=ArcFaceConfig(**payload["arcface"]),
        optimizer=OptimizerConfig(**payload["optimizer"]),
        losses=SkillLossConfig(**payload["losses"]),
        trainer=TrainerConfig(**payload["trainer"]),
    )
