from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from kronos_shared import SyntheticClassificationRow, load_synthetic_rows, save_synthetic_rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repartition a shittim synthetic manifest for closed-set retrieval.")
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gallery-count", type=int, default=4)
    parser.add_argument("--train-count", type=int, default=32)
    parser.add_argument("--val-count", type=int, default=8)
    parser.add_argument("--test-count", type=int, default=8)
    return parser.parse_args(argv)


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


def repartition_rows(
    rows: list[SyntheticClassificationRow],
    *,
    seed: int,
    gallery_count: int,
    train_count: int,
    val_count: int,
    test_count: int,
) -> list[SyntheticClassificationRow]:
    required = gallery_count + train_count + val_count + test_count
    grouped: dict[str, list[SyntheticClassificationRow]] = {}
    for row in rows:
        grouped.setdefault(row.character_id, []).append(row)

    rng = random.Random(seed)
    prepared: list[SyntheticClassificationRow] = []
    for character_id in sorted(grouped):
        variants = list(grouped[character_id])
        if len(variants) < required:
            raise ValueError(
                f"Identity {character_id} only has {len(variants)} variants, but {required} are required."
            )

        cleanest = sorted(variants, key=row_cleanliness_key)
        gallery_rows = cleanest[:gallery_count]
        gallery_ids = {id(row) for row in gallery_rows}
        remaining = [row for row in variants if id(row) not in gallery_ids]
        rng.shuffle(remaining)

        train_rows = remaining[:train_count]
        val_rows = remaining[train_count : train_count + val_count]
        test_rows = remaining[train_count + val_count : train_count + val_count + test_count]
        extra_train_rows = remaining[train_count + val_count + test_count :]

        subsets = (
            ("gallery", gallery_rows),
            ("train_query", train_rows + extra_train_rows),
            ("val_query", val_rows),
            ("test_query", test_rows),
        )
        for subset, subset_rows in subsets:
            for row in subset_rows:
                prepared.append(replace(row, subset=subset))

    return prepared


def summarize_rows(rows: list[SyntheticClassificationRow]) -> dict[str, object]:
    subset_counts: dict[str, int] = {}
    identity_counts: dict[str, int] = {}
    for row in rows:
        subset_counts[row.subset] = subset_counts.get(row.subset, 0) + 1
        identity_counts[row.character_id] = identity_counts.get(row.character_id, 0) + 1
    return {
        "row_count": len(rows),
        "identity_count": len(identity_counts),
        "subset_counts": subset_counts,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rows = load_synthetic_rows(args.input_manifest)
    prepared = repartition_rows(
        rows,
        seed=args.seed,
        gallery_count=args.gallery_count,
        train_count=args.train_count,
        val_count=args.val_count,
        test_count=args.test_count,
    )
    save_synthetic_rows(args.output_manifest, prepared)
    print(json.dumps(summarize_rows(prepared), indent=2))


if __name__ == "__main__":
    main()
