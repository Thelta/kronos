from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "kronos-shared" / "src"))
sys.path.insert(0, str(ROOT / "packages" / "kronos-train" / "src"))

from kronos_shared import SkillClassificationRow  # noqa: E402
from kronos_train.skill_pipeline import analyze_skill_confusions, evaluate_skill_checkpoint, load_canonical_records, top_unique_identities  # noqa: E402
from kronos_train.skill_training_config import SkillTrainConfig  # noqa: E402


def make_skill_row(identity: str | None, seed: int, subset: str = "test", *, empty: bool = False) -> SkillClassificationRow:
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
        render_mode="full_color_flash" if not empty else "empty",
        obstructions=[],
        quality_policy={"jpeg_quality": 90, "translation_px": [0, 0]},
        empty=empty,
    )


class SkillPipelineTests(unittest.TestCase):
    def test_load_canonical_records_accepts_sparse_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir)
            images_dir = temp_path / "images"
            images_dir.mkdir()
            (images_dir / "sample.png").write_bytes(b"png")
            manifest_path = temp_path / "manifest.jsonl"
            manifest_path.write_text(
                json.dumps({"image_path": "images/sample.png", "character_id": "Skill_A", "skill_card_id": "Skill_A"}) + "\n",
                encoding="utf-8",
            )

            records = load_canonical_records(manifest_path)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].identity_key, "Skill_A")
        self.assertEqual(records[0].image_path, images_dir / "sample.png")

    def test_top_unique_identities_collapses_duplicate_gallery_hits(self) -> None:
        import torch

        records = [
            type("Record", (), {"identity_key": "Skill_A"})(),
            type("Record", (), {"identity_key": "Skill_A"})(),
            type("Record", (), {"identity_key": "Skill_B"})(),
            type("Record", (), {"identity_key": "Skill_C"})(),
        ]

        identities = top_unique_identities(torch.tensor([0.9, 0.8, 0.7, 0.6]), records, limit=3)

        self.assertEqual(identities, ["Skill_A", "Skill_B", "Skill_C"])

    def test_evaluate_skill_checkpoint_can_use_canonical_manifest(self) -> None:
        import torch

        config = SkillTrainConfig()
        config.data.manifest_path = "D:/fake/skill_cls/manifest.jsonl"
        rows = [make_skill_row("Skill_A", 1), make_skill_row(None, 2, empty=True)]
        query_loader = object()
        output = io.StringIO()

        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "canonical.jsonl"
            manifest_path.write_text(
                json.dumps({"image_path": "images/sample.png", "character_id": "Skill_A", "skill_card_id": "Skill_A"}) + "\n",
                encoding="utf-8",
            )
            with (
                patch("kronos_train.skill_pipeline.load_model_from_checkpoint", return_value=(object(), config, ["Skill_A"], torch.device("cpu"))),
                patch("kronos_train.skill_pipeline.load_rows", return_value=rows),
                patch("kronos_train.skill_pipeline.prepare_runtime_subsets", return_value={"gallery": [rows[0]], "train_query": rows, "val_query": rows, "test_query": rows}),
                patch("kronos_train.skill_pipeline.SkillClassificationDataset", return_value=object()),
                patch("torch.utils.data.DataLoader", return_value=query_loader),
                patch(
                    "kronos_train.skill_pipeline.evaluate_retrieval_against_canonical",
                    return_value={"all": {"count": 1, "retrieval_top1": 1.0, "retrieval_top3": 1.0}, "gallery_type": "canonical_image"},
                ) as evaluate_mock,
                redirect_stdout(output),
            ):
                evaluate_skill_checkpoint(Path("checkpoint.pt"), "test_query", "cpu", canonical_manifest_path=manifest_path)

        self.assertEqual(json.loads(output.getvalue())["gallery_type"], "canonical_image")
        self.assertEqual(Path(evaluate_mock.call_args.kwargs["canonical_manifest_path"]), manifest_path)

    def test_analyze_skill_confusions_can_use_canonical_manifest(self) -> None:
        import torch

        config = SkillTrainConfig()
        config.data.manifest_path = "D:/fake/skill_cls/manifest.jsonl"
        rows = [make_skill_row("Skill_A", 1), make_skill_row(None, 2, empty=True)]
        query_loader = object()
        output = io.StringIO()

        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "canonical.jsonl"
            manifest_path.write_text(
                json.dumps({"image_path": "images/sample.png", "character_id": "Skill_A", "skill_card_id": "Skill_A"}) + "\n",
                encoding="utf-8",
            )
            with (
                patch("kronos_train.skill_pipeline.load_model_from_checkpoint", return_value=(object(), config, ["Skill_A"], torch.device("cpu"))),
                patch("kronos_train.skill_pipeline.load_rows", return_value=rows),
                patch("kronos_train.skill_pipeline.prepare_runtime_subsets", return_value={"gallery": [rows[0]], "train_query": rows, "val_query": rows, "test_query": rows}),
                patch("kronos_train.skill_pipeline.SkillClassificationDataset", return_value=object()),
                patch("torch.utils.data.DataLoader", return_value=query_loader),
                patch(
                    "kronos_train.skill_pipeline.analyze_retrieval_confusions_against_canonical",
                    return_value={"gallery_type": "canonical_image", "sample_count": 1, "overall_closest_competitors": []},
                ) as analyze_mock,
                redirect_stdout(output),
            ):
                analyze_skill_confusions(Path("checkpoint.pt"), "test_query", "cpu", canonical_manifest_path=manifest_path)

        self.assertEqual(json.loads(output.getvalue())["gallery_type"], "canonical_image")
        self.assertEqual(Path(analyze_mock.call_args.kwargs["canonical_manifest_path"]), manifest_path)


if __name__ == "__main__":
    unittest.main()
