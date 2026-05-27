from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .yolo_config import YoloTrainConfig


@dataclass
class PreparedSample:
    source_name: str
    image_path: Path
    label_path: Path
    output_stem: str


def load_review_statuses(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {str(key): str(value) for key, value in payload.items()}


def gather_synthetic_samples(config: YoloTrainConfig) -> list[PreparedSample]:
    images_dir = Path(config.dataset.synthetic_images_dir)
    labels_dir = Path(config.dataset.synthetic_labels_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"Synthetic image directory does not exist: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Synthetic label directory does not exist: {labels_dir}")

    samples: list[PreparedSample] = []
    seen_images: set[Path] = set()
    for pattern in config.dataset.synthetic_image_patterns:
        for image_path in sorted(images_dir.glob(pattern)):
            if image_path in seen_images:
                continue
            seen_images.add(image_path)
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            samples.append(
                PreparedSample(
                    source_name="synthetic",
                    image_path=image_path,
                    label_path=label_path,
                    output_stem=f"syn_{image_path.stem}",
                )
            )
    return samples


def gather_reviewed_real_samples(config: YoloTrainConfig) -> list[PreparedSample]:
    review_dir = Path(config.dataset.reviewed_real_dir)
    status_path = Path(config.dataset.review_status_file)
    if not review_dir.exists():
        raise FileNotFoundError(f"Reviewed real directory does not exist: {review_dir}")
    if not status_path.exists():
        raise FileNotFoundError(f"Review status file does not exist: {status_path}")

    statuses = load_review_statuses(status_path)
    allowed = set(config.dataset.include_review_statuses)
    samples: list[PreparedSample] = []
    seen_images: set[Path] = set()
    for pattern in config.dataset.reviewed_real_image_patterns:
        for image_path in sorted(review_dir.glob(pattern)):
            if image_path in seen_images:
                continue
            seen_images.add(image_path)
            if statuses.get(image_path.name) not in allowed:
                continue
            label_stem = image_path.stem
            if not label_stem.endswith("_orig"):
                continue
            label_stem = label_stem[: -len("_orig")]
            label_path = review_dir / f"{label_stem}.txt"
            if not label_path.exists():
                continue
            samples.append(
                PreparedSample(
                    source_name="reviewed_real",
                    image_path=image_path,
                    label_path=label_path,
                    output_stem=f"real_{image_path.stem}",
                )
            )
    return samples


def split_samples(samples: list[PreparedSample], train_split: float, seed: int) -> tuple[list[PreparedSample], list[PreparedSample]]:
    if not 0.0 < float(train_split) < 1.0:
        raise ValueError(f"dataset.train_split must be between 0 and 1, got {train_split!r}")
    if len(samples) < 2:
        raise ValueError("Need at least two labeled images to create train/val splits.")

    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    train_count = int(len(shuffled) * float(train_split))
    train_count = max(1, min(len(shuffled) - 1, train_count))
    return shuffled[:train_count], shuffled[train_count:]


def reset_output_splits(output_dir: Path) -> None:
    for path in (
        output_dir / "images" / "train",
        output_dir / "images" / "val",
        output_dir / "labels" / "train",
        output_dir / "labels" / "val",
    ):
        if path.exists():
            shutil.rmtree(path)


def copy_samples(samples: list[PreparedSample], *, image_dir: Path, label_dir: Path) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        shutil.copy2(sample.image_path, image_dir / f"{sample.output_stem}{sample.image_path.suffix.lower()}")
        shutil.copy2(sample.label_path, label_dir / f"{sample.output_stem}.txt")


def write_data_yaml(output_dir: Path, class_names: list[str]) -> Path:
    yaml_path = output_dir / "data.yaml"
    names = ", ".join(f"'{name}'" for name in class_names)
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {output_dir.resolve().as_posix()}",
                "train: images/train",
                "val: images/val",
                f"nc: {len(class_names)}",
                f"names: [{names}]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def write_manifest(output_dir: Path, train_samples: list[PreparedSample], val_samples: list[PreparedSample], class_names: list[str]) -> Path:
    manifest_path = output_dir / "dataset_manifest.json"
    payload = {
        "output_dir": str(output_dir.resolve()),
        "class_names": class_names,
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "synthetic_count": sum(1 for sample in train_samples + val_samples if sample.source_name == "synthetic"),
        "reviewed_real_count": sum(1 for sample in train_samples + val_samples if sample.source_name == "reviewed_real"),
        "samples": [
            {
                "source_name": sample.source_name,
                "subset": subset,
                "image": str(sample.image_path),
                "label": str(sample.label_path),
                "output_stem": sample.output_stem,
            }
            for subset, subset_samples in (("train", train_samples), ("val", val_samples))
            for sample in subset_samples
        ],
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def prepare_yolo_dataset(config: YoloTrainConfig) -> dict[str, Any]:
    output_dir = Path(config.dataset.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    synthetic_samples = gather_synthetic_samples(config)
    reviewed_real_samples = gather_reviewed_real_samples(config)
    samples = synthetic_samples + reviewed_real_samples
    train_samples, val_samples = split_samples(samples, config.dataset.train_split, config.dataset.seed)

    reset_output_splits(output_dir)
    copy_samples(train_samples, image_dir=output_dir / "images" / "train", label_dir=output_dir / "labels" / "train")
    copy_samples(val_samples, image_dir=output_dir / "images" / "val", label_dir=output_dir / "labels" / "val")
    yaml_path = write_data_yaml(output_dir, config.dataset.class_names)
    manifest_path = write_manifest(output_dir, train_samples, val_samples, config.dataset.class_names)
    summary = {
        "dataset_dir": str(output_dir.resolve()),
        "data_yaml": str(yaml_path.resolve()),
        "manifest": str(manifest_path.resolve()),
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "synthetic_count": len(synthetic_samples),
        "reviewed_real_count": len(reviewed_real_samples),
        "total_count": len(samples),
    }
    print(json.dumps(summary, indent=2))
    return summary


def train_yolo_from_config(config: YoloTrainConfig, prepare: bool = True) -> Any:
    if prepare:
        prepare_yolo_dataset(config)

    data_yaml = Path(config.dataset.output_dir) / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    from ultralytics import YOLO

    model = YOLO(config.trainer.weights)
    results = model.train(
        data=str(data_yaml),
        epochs=config.trainer.epochs,
        imgsz=config.trainer.imgsz,
        batch=config.trainer.batch,
        patience=config.trainer.patience,
        device=config.trainer.device,
        project=config.trainer.project,
        name=config.trainer.name,
        save=config.trainer.save,
        plots=config.trainer.plots,
        exist_ok=config.trainer.exist_ok,
        verbose=config.trainer.verbose,
    )
    return results
