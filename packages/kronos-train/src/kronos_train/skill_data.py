from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from kronos_shared import SkillClassificationRow, load_skill_rows

from .data import EMPTY_CHARACTER_ID, build_train_transforms, build_transforms, resolve_image_path, row_cleanliness_key


def load_rows(manifest_path: Path) -> list[SkillClassificationRow]:
    rows = load_skill_rows(manifest_path)
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return rows


def is_empty_row(row: SkillClassificationRow) -> bool:
    return bool(row.empty) or row.identity_key.strip().lower() == EMPTY_CHARACTER_ID


def ensure_empty_examples(rows: list[SkillClassificationRow]) -> None:
    if not any(is_empty_row(row) for row in rows):
        raise ValueError("Manifest must contain at least one empty-labeled example.")


def build_identity_index(rows: list[SkillClassificationRow]) -> list[str]:
    ensure_empty_examples(rows)
    identities = sorted({row.identity_key for row in rows if not is_empty_row(row)})
    if not identities:
        raise ValueError("Manifest must contain at least one non-empty identity.")
    return identities


def filter_rows(rows: list[SkillClassificationRow], subset: str) -> list[SkillClassificationRow]:
    return [row for row in rows if row.subset == subset]


def supports_gallery_subset(rows: list[SkillClassificationRow], gallery_subset: str) -> bool:
    return any(row.subset == gallery_subset for row in rows)


def prepare_runtime_subsets(
    rows: list[SkillClassificationRow],
    *,
    train_subset: str,
    val_subset: str,
    test_subset: str,
    gallery_subset: str,
    gallery_count_per_identity: int,
    seed: int,
) -> dict[str, list[SkillClassificationRow]]:
    ensure_empty_examples(rows)
    if supports_gallery_subset(rows, gallery_subset):
        return {
            "gallery": [row for row in filter_rows(rows, gallery_subset) if not is_empty_row(row)],
            "train_query": filter_rows(rows, "train_query") or filter_rows(rows, train_subset),
            "val_query": filter_rows(rows, "val_query") or filter_rows(rows, val_subset),
            "test_query": filter_rows(rows, "test_query") or filter_rows(rows, test_subset),
        }

    train_rows = filter_rows(rows, train_subset)
    empty_train_rows = [row for row in train_rows if is_empty_row(row)]
    grouped_train: dict[str, list[SkillClassificationRow]] = {}
    for row in train_rows:
        if is_empty_row(row):
            continue
        grouped_train.setdefault(row.identity_key, []).append(row)

    rng = random.Random(seed)
    gallery_rows: list[SkillClassificationRow] = []
    train_query_rows: list[SkillClassificationRow] = []
    for identity_key, identity_rows in sorted(grouped_train.items()):
        if len(identity_rows) <= gallery_count_per_identity:
            raise ValueError(
                f"Identity {identity_key} has only {len(identity_rows)} training variants; "
                f"need more than gallery_count_per_identity={gallery_count_per_identity}."
            )
        cleanest = sorted(identity_rows, key=row_cleanliness_key)
        gallery_selected = cleanest[:gallery_count_per_identity]
        selected_ids = {id(row) for row in gallery_selected}
        remaining = [row for row in identity_rows if id(row) not in selected_ids]
        rng.shuffle(remaining)
        gallery_rows.extend(gallery_selected)
        train_query_rows.extend(remaining)

    train_query_rows.extend(empty_train_rows)
    return {
        "gallery": gallery_rows,
        "train_query": train_query_rows,
        "val_query": filter_rows(rows, val_subset),
        "test_query": filter_rows(rows, test_subset),
    }


class SkillClassificationDataset:
    def __init__(
        self,
        *,
        rows: list[SkillClassificationRow],
        dataset_root: Path,
        identity_to_index: dict[str, int],
        image_size: int,
        augment: bool = False,
    ) -> None:
        self.rows = rows
        self.dataset_root = dataset_root
        self.identity_to_index = identity_to_index
        self.image_size = image_size
        self.transform = build_train_transforms(image_size) if augment else build_transforms(image_size)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from PIL import Image

        row = self.rows[index]
        image = Image.open(resolve_image_path(self.dataset_root, row.image_path)).convert("RGB")
        empty = is_empty_row(row)
        width, height = image.size
        scale_x = self.image_size / width
        scale_y = self.image_size / height
        card_box = row.card_box
        scaled_card_box = [
            float(card_box[0]) * scale_x,
            float(card_box[1]) * scale_y,
            float(card_box[2]) * scale_x,
            float(card_box[3]) * scale_y,
        ]
        return {
            "image": self.transform(image),
            "identity_index": -1 if empty else self.identity_to_index[row.identity_key],
            "empty_label": 1.0 if empty else 0.0,
            "card_box": scaled_card_box,
            "row": row,
        }


def collate_samples(batch: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    identity_indices = torch.tensor([item["identity_index"] for item in batch], dtype=torch.long)
    return {
        "images": torch.stack([item["image"] for item in batch]),
        "identity_indices": identity_indices,
        "empty_labels": torch.tensor([item["empty_label"] for item in batch], dtype=torch.float32),
        "non_empty_mask": identity_indices >= 0,
        "card_boxes": torch.tensor([item["card_box"] for item in batch], dtype=torch.float32),
        "rows": [item["row"] for item in batch],
    }


class BalancedIdentityModeSampler:
    def __init__(self, rows: list[SkillClassificationRow], *, seed: int) -> None:
        self.rows = rows
        self.seed = int(seed)
        self._iteration_count = 0

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        buckets: dict[tuple[str, str], list[int]] = {}
        empty_indices: list[int] = []
        for index, row in enumerate(self.rows):
            if is_empty_row(row):
                empty_indices.append(index)
                continue
            buckets.setdefault((row.identity_key, row.render_mode or ""), []).append(index)

        rng = random.Random(self.seed + self._iteration_count)
        self._iteration_count += 1
        for indices in buckets.values():
            rng.shuffle(indices)
        rng.shuffle(empty_indices)

        active_keys = list(buckets.keys())
        ordered_indices: list[int] = []
        empty_insert_every = max(1, len(self.rows) // len(empty_indices)) if empty_indices else None
        emitted_non_empty = 0

        while active_keys:
            rng.shuffle(active_keys)
            next_active_keys: list[tuple[str, str]] = []
            for key in active_keys:
                bucket = buckets[key]
                if not bucket:
                    continue
                ordered_indices.append(bucket.pop())
                emitted_non_empty += 1
                if empty_indices and empty_insert_every is not None and emitted_non_empty % empty_insert_every == 0:
                    ordered_indices.append(empty_indices.pop())
                if bucket:
                    next_active_keys.append(key)
            active_keys = next_active_keys

        while empty_indices:
            ordered_indices.append(empty_indices.pop())
        return iter(ordered_indices)
