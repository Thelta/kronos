from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "kronos-shared" / "src"))
sys.path.insert(0, str(ROOT / "packages" / "kronos-train" / "src"))

from kronos_shared import SkillClassificationRow, load_skill_rows  # noqa: E402
from kronos_train.skill_data import BalancedIdentityModeSampler, build_identity_index, is_empty_row, prepare_runtime_subsets  # noqa: E402


def make_skill_row(identity: str | None, seed: int, subset: str = "train", *, empty: bool = False) -> SkillClassificationRow:
    return SkillClassificationRow(
        image_path=f"images/{identity or 'empty'}_{seed}.jpg",
        subset=subset,
        mode="classification",
        character_id=None if empty else identity,
        skill_card_id=None if empty else identity,
        assist=False,
        seed=seed,
        background_kind="gradient",
        card_box=[0, 0, 10, 10],
        portrait_box=[0, 0, 8, 8],
        obstructions=[],
        quality_policy={"jpeg_quality": 90, "translation_px": [0, 0]},
        empty=empty,
    )


class SkillDataTests(unittest.TestCase):
    def test_load_skill_rows_prefers_skill_card_id_with_character_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "manifest.jsonl"
            path.write_text(
                '{"image_path":"images/one.jpg","split":"train","mode":"classification",'
                '"character_id":"fallback","skill_card_id":null,"assist":false,"seed":1,'
                '"background_kind":"gradient","card_box":[0,0,10,10],"portrait_box":[0,0,8,8]}\n',
                encoding="utf-8",
            )
            rows = load_skill_rows(path)
        self.assertEqual(rows[0].identity_key, "fallback")

    def test_build_identity_index_excludes_empty(self) -> None:
        rows = [make_skill_row("Skill_A", 1), make_skill_row(None, 2, empty=True)]
        self.assertEqual(build_identity_index(rows), ["Skill_A"])

    def test_prepare_runtime_subsets_keeps_empty_out_of_gallery(self) -> None:
        rows = [make_skill_row("Skill_A", seed) for seed in range(4)]
        rows.extend(make_skill_row("Skill_B", seed + 10) for seed in range(4))
        rows.append(make_skill_row(None, 100, empty=True))

        prepared = prepare_runtime_subsets(
            rows,
            train_subset="train",
            val_subset="val",
            test_subset="test",
            gallery_subset="gallery",
            gallery_count_per_identity=1,
            seed=7,
        )

        self.assertEqual(len(prepared["gallery"]), 2)
        self.assertFalse(any(is_empty_row(row) for row in prepared["gallery"]))
        self.assertTrue(any(is_empty_row(row) for row in prepared["train_query"]))

    def test_balanced_sampler_interleaves_identity_mode_buckets(self) -> None:
        rows: list[SkillClassificationRow] = []
        for identity in ("Skill_A", "Skill_B"):
            for render_mode in ("full_color", "full_color_flash"):
                row = make_skill_row(identity, len(rows) + 1)
                row.render_mode = render_mode
                rows.append(row)
        rows.append(make_skill_row(None, 100, empty=True))

        sampler = BalancedIdentityModeSampler(rows, seed=7)
        order = list(iter(sampler))
        first_four = [rows[index] for index in order[:4]]

        self.assertEqual(len(order), len(rows))
        self.assertEqual(len({(row.identity_key, row.render_mode) for row in first_four}), 4)


if __name__ == "__main__":
    unittest.main()
