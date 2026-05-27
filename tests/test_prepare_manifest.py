from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "kronos-shared" / "src"))
sys.path.insert(0, str(ROOT / "packages" / "kronos-train" / "src"))

from kronos_shared import SyntheticClassificationRow  # noqa: E402
from kronos_train.data import prepare_runtime_subsets  # noqa: E402
from kronos_train.prep import repartition_rows  # noqa: E402


def make_row(character_id: str, seed: int, obstruction_count: int = 0) -> SyntheticClassificationRow:
    return SyntheticClassificationRow(
        image_path=f"images/{character_id}_{seed}.jpg",
        subset="train",
        mode="classification",
        character_id=character_id,
        portrait_path=f"portraits/{character_id}.png",
        attack_type="Explosion",
        role="Supporter",
        level=90,
        star_value=3,
        star_color="yellow",
        assist=False,
        starter=False,
        seed=seed,
        background_kind="gradient",
        scale_jitter=1.0,
        card_box=[0, 0, 10, 10],
        portrait_box=[0, 0, 8, 8],
        source_split="train",
        obstructions=[{"kind": "glare"}] * obstruction_count,
        quality_policy={"jpeg_quality": 90, "translation_px": [0, 0]},
        global_effects=[],
    )


class PrepareManifestTests(unittest.TestCase):
    def test_repartition_assigns_each_identity_to_all_required_subsets(self) -> None:
        rows = []
        for character_id in ("1001", "1002"):
            for seed in range(15):
                rows.append(make_row(character_id, seed, obstruction_count=1 if seed > 2 else 0))

        prepared = repartition_rows(
            rows,
            seed=7,
            gallery_count=2,
            train_count=5,
            val_count=3,
            test_count=3,
        )

        by_character: dict[str, dict[str, int]] = {}
        for row in prepared:
            by_character.setdefault(row.character_id, {})
            by_character[row.character_id][row.subset] = by_character[row.character_id].get(row.subset, 0) + 1

        for counts in by_character.values():
            self.assertEqual(counts["gallery"], 2)
            self.assertEqual(counts["val_query"], 3)
            self.assertEqual(counts["test_query"], 3)
            self.assertGreaterEqual(counts["train_query"], 5)

    def test_cleanest_rows_are_reserved_for_gallery(self) -> None:
        rows = [make_row("1001", seed, obstruction_count=1 if seed >= 2 else 0) for seed in range(8)]
        prepared = repartition_rows(
            rows,
            seed=1,
            gallery_count=2,
            train_count=2,
            val_count=2,
            test_count=2,
        )
        gallery_seeds = sorted(row.seed for row in prepared if row.subset == "gallery")
        self.assertEqual(gallery_seeds, [0, 1])

    def test_runtime_subset_preparation_uses_raw_train_val_test_manifest(self) -> None:
        rows = []
        for character_id in ("1001", "1002"):
            for seed in range(6):
                rows.append(make_row(character_id, seed, obstruction_count=1 if seed > 1 else 0))
                rows[-1].subset = "train"
            rows.append(make_row(character_id, 100, obstruction_count=1))
            rows[-1].subset = "val"
            rows.append(make_row(character_id, 101, obstruction_count=1))
            rows[-1].subset = "test"

        prepared = prepare_runtime_subsets(
            rows,
            train_subset="train",
            val_subset="val",
            test_subset="test",
            gallery_subset="gallery",
            gallery_count_per_identity=2,
            seed=7,
        )

        self.assertEqual(len(prepared["gallery"]), 4)
        self.assertEqual(len(prepared["train_query"]), 8)
        self.assertEqual(len(prepared["val_query"]), 2)
        self.assertEqual(len(prepared["test_query"]), 2)


if __name__ == "__main__":
    unittest.main()
