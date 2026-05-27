from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from kronos_shared import SyntheticClassificationRow, load_synthetic_rows


def load_rows(manifest_path: Path) -> list[SyntheticClassificationRow]:
    rows = load_synthetic_rows(manifest_path)
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return rows


def resolve_image_path(dataset_root: Path, image_path: str) -> Path:
    path = Path(image_path)
    if path.is_absolute():
        return path
    return dataset_root / path


def build_identity_index(rows: list[SyntheticClassificationRow]) -> list[str]:
    return sorted({row.character_id for row in rows})


def filter_rows(rows: list[SyntheticClassificationRow], subset: str) -> list[SyntheticClassificationRow]:
    return [row for row in rows if row.subset == subset]


def row_cleanliness_key(row: SyntheticClassificationRow) -> tuple[int, int, int, int, int, int]:
    quality = row.quality_policy
    jpeg_quality = int(quality.get("jpeg_quality", 100))
    resize_ratio = float(quality.get("resize_degrade_ratio", 1.0))
    translation = quality.get("translation_px", [0, 0])
    translation_magnitude = sum(abs(int(value)) for value in translation[:2])
    return (
        len(row.obstructions),
        1 if "blur_radius" in quality else 0,
        2 if jpeg_quality < 60 else (1 if jpeg_quality < 80 else 0),
        2 if resize_ratio < 0.7 else (1 if resize_ratio < 0.85 else 0),
        translation_magnitude,
        row.seed,
    )


def supports_gallery_subset(rows: list[SyntheticClassificationRow], gallery_subset: str) -> bool:
    return any(row.subset == gallery_subset for row in rows)


def prepare_runtime_subsets(
    rows: list[SyntheticClassificationRow],
    *,
    train_subset: str,
    val_subset: str,
    test_subset: str,
    gallery_subset: str,
    gallery_count_per_identity: int,
    seed: int,
) -> dict[str, list[SyntheticClassificationRow]]:
    if supports_gallery_subset(rows, gallery_subset):
        return {
            "gallery": filter_rows(rows, gallery_subset),
            "train_query": filter_rows(rows, "train_query") or filter_rows(rows, train_subset),
            "val_query": filter_rows(rows, "val_query") or filter_rows(rows, val_subset),
            "test_query": filter_rows(rows, "test_query") or filter_rows(rows, test_subset),
        }

    grouped_train: dict[str, list[SyntheticClassificationRow]] = {}
    for row in filter_rows(rows, train_subset):
        grouped_train.setdefault(row.character_id, []).append(row)

    rng = random.Random(seed)
    gallery_rows: list[SyntheticClassificationRow] = []
    train_query_rows: list[SyntheticClassificationRow] = []
    for character_id, character_rows in sorted(grouped_train.items()):
        if len(character_rows) <= gallery_count_per_identity:
            raise ValueError(
                f"Identity {character_id} has only {len(character_rows)} training variants; "
                f"need more than gallery_count_per_identity={gallery_count_per_identity}."
            )
        cleanest = sorted(character_rows, key=row_cleanliness_key)
        gallery_selected = cleanest[:gallery_count_per_identity]
        selected_ids = {id(row) for row in gallery_selected}
        remaining = [row for row in character_rows if id(row) not in selected_ids]
        rng.shuffle(remaining)
        gallery_rows.extend(gallery_selected)
        train_query_rows.extend(remaining)

    return {
        "gallery": gallery_rows,
        "train_query": train_query_rows,
        "val_query": filter_rows(rows, val_subset),
        "test_query": filter_rows(rows, test_subset),
    }


def degrade_tiny_card_image(
    image: Any,
    *,
    probability: float,
    sizes: list[list[int]],
    size_jitter: float,
    rng: random.Random | None = None,
) -> Any:
    from PIL import Image

    if probability <= 0.0 or not sizes:
        return image
    rng = rng or random
    if rng.random() > probability:
        return image

    original_size = image.size
    target_width, target_height = rng.choice(sizes)
    jitter = rng.uniform(1.0 - size_jitter, 1.0 + size_jitter)
    target_width = max(8, int(round(float(target_width) * jitter)))
    target_height = max(8, int(round(float(target_height) * jitter)))

    resample_options = [
        Image.Resampling.BILINEAR,
        Image.Resampling.BICUBIC,
        Image.Resampling.LANCZOS,
        Image.Resampling.NEAREST,
    ]
    downsample = rng.choice(resample_options)
    upsample = rng.choice(resample_options)
    return image.resize((target_width, target_height), downsample).resize(original_size, upsample)


def build_transforms(image_size: int) -> Any:
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def star_color_to_index(star_color: str) -> int:
    normalized = star_color.strip().lower()
    if normalized == "yellow":
        return 0
    if normalized == "blue":
        return 1
    raise ValueError(f"Unsupported star_color value: {star_color!r}")


def build_star_slot_indices(*, star_value: int, star_color: str) -> list[int]:
    if not 1 <= int(star_value) <= 5:
        raise ValueError(f"Unsupported star_value: {star_value!r}")
    filled_class = 1 if star_color_to_index(star_color) == 0 else 2
    filled_count = int(star_value)
    return [filled_class] * filled_count + [0] * (5 - filled_count)


def build_star_state_index(*, star_value: int, star_color: str) -> int:
    return star_color_to_index(star_color) * 5 + (int(star_value) - 1)


class SyntheticCardDataset:
    def __init__(
        self,
        *,
        rows: list[SyntheticClassificationRow],
        dataset_root: Path,
        character_to_index: dict[str, int],
        image_size: int,
        tiny_card_degrade_prob: float = 0.0,
        tiny_card_sizes: list[list[int]] | None = None,
        tiny_card_size_jitter: float = 0.12,
    ) -> None:
        self.rows = rows
        self.dataset_root = dataset_root
        self.character_to_index = character_to_index
        self.image_size = image_size
        self.tiny_card_degrade_prob = tiny_card_degrade_prob
        self.tiny_card_sizes = tiny_card_sizes or []
        self.tiny_card_size_jitter = tiny_card_size_jitter
        self.transform = build_transforms(image_size)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from PIL import Image

        row = self.rows[index]
        image = Image.open(resolve_image_path(self.dataset_root, row.image_path)).convert("RGB")
        width, height = image.size
        image = degrade_tiny_card_image(
            image,
            probability=self.tiny_card_degrade_prob,
            sizes=self.tiny_card_sizes,
            size_jitter=self.tiny_card_size_jitter,
        )
        scale_x = self.image_size / width
        scale_y = self.image_size / height
        card_box = row.card_box
        scaled_card_box = [
            float(card_box[0]) * scale_x,
            float(card_box[1]) * scale_y,
            float(card_box[2]) * scale_x,
            float(card_box[3]) * scale_y,
        ]
        if row.star_box is not None:
            star_box = row.star_box
            scaled_star_box = [
                float(star_box[0]) * scale_x,
                float(star_box[1]) * scale_y,
                float(star_box[2]) * scale_x,
                float(star_box[3]) * scale_y,
            ]
        else:
            scaled_star_box = [float("nan"), float("nan"), float("nan"), float("nan")]
        return {
            "image": self.transform(image),
            "identity_index": self.character_to_index[row.character_id],
            "star_index": int(row.star_value) - 1,
            "star_color_index": star_color_to_index(row.star_color),
            "star_state_index": build_star_state_index(star_value=row.star_value, star_color=row.star_color),
            "star_slot_indices": build_star_slot_indices(star_value=row.star_value, star_color=row.star_color),
            "assist": 1.0 if row.assist else 0.0,
            "card_box": scaled_card_box,
            "star_box": scaled_star_box,
            "row": row,
        }


def collate_samples(batch: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    return {
        "images": torch.stack([item["image"] for item in batch]),
        "identity_indices": torch.tensor([item["identity_index"] for item in batch], dtype=torch.long),
        "star_indices": torch.tensor([item["star_index"] for item in batch], dtype=torch.long),
        "star_color_indices": torch.tensor([item["star_color_index"] for item in batch], dtype=torch.long),
        "star_state_indices": torch.tensor([item["star_state_index"] for item in batch], dtype=torch.long),
        "star_slot_indices": torch.tensor([item["star_slot_indices"] for item in batch], dtype=torch.long),
        "assist": torch.tensor([item["assist"] for item in batch], dtype=torch.float32),
        "card_boxes": torch.tensor([item["card_box"] for item in batch], dtype=torch.float32),
        "star_boxes": torch.tensor([item["star_box"] for item in batch], dtype=torch.float32),
        "rows": [item["row"] for item in batch],
    }
