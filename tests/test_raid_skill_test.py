from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.raid_skill_test import (  # noqa: E402
    CanonicalImageRecord,
    CanonicalIndex,
    PortraitDetection,
    QueryPrediction,
    build_similarity_debug_stats,
    build_top_matches,
    collect_input_images,
    crop_to_box,
    load_canonical_records,
    process_image,
    run_cli,
    sort_detections,
)


class RaidSkillTestScriptTests(unittest.TestCase):
    def test_load_canonical_records_accepts_subset_none_and_missing_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            images_dir = temp_path / "images"
            images_dir.mkdir()
            Image.new("RGB", (12, 12), color=(255, 0, 0)).save(images_dir / "sample.png")
            manifest_path = temp_path / "manifest.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "image_path": "images/sample.png",
                        "subset": None,
                        "mode": "canonical",
                        "character_id": "Skill_Portrait_Airi",
                        "skill_card_id": "Skill_Portrait_Airi",
                        "assist": False,
                        "seed": 1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            records = load_canonical_records(manifest_path)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].skill_card_id, "Skill_Portrait_Airi")

    def test_load_canonical_records_resolves_image_paths_relative_to_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            images_dir = temp_path / "nested" / "images"
            images_dir.mkdir(parents=True)
            image_path = images_dir / "sample.png"
            Image.new("RGB", (12, 12), color=(255, 0, 0)).save(image_path)
            manifest_path = temp_path / "nested" / "manifest.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "image_path": "images/sample.png",
                        "subset": None,
                        "mode": "canonical",
                        "character_id": "Skill_Portrait_Airi",
                        "skill_card_id": "Skill_Portrait_Airi",
                        "assist": False,
                        "seed": 1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            records = load_canonical_records(manifest_path)

        self.assertEqual(records[0].image_path, image_path)

    def test_build_top_matches_returns_highest_similarity_identity(self) -> None:
        import torch

        records = [
            CanonicalImageRecord(skill_card_id="Skill_A", character_id="Skill_A", image_path=Path("a.png")),
            CanonicalImageRecord(skill_card_id="Skill_B", character_id="Skill_B", image_path=Path("b.png")),
        ]

        matches = build_top_matches(torch.tensor([0.2, 0.9]), records, topk=2)

        self.assertEqual(matches[0]["skill_card_id"], "Skill_B")
        self.assertEqual(matches[1]["skill_card_id"], "Skill_A")

    def test_build_similarity_debug_stats_summarizes_modes_and_margin(self) -> None:
        import torch

        records = [
            CanonicalImageRecord(
                skill_card_id="Skill_A",
                character_id="Skill_A",
                image_path=Path("a0.png"),
                render_mode="full_color",
            ),
            CanonicalImageRecord(
                skill_card_id="Skill_A",
                character_id="Skill_A",
                image_path=Path("a1.png"),
                render_mode="full_color_flash",
            ),
            CanonicalImageRecord(
                skill_card_id="Skill_B",
                character_id="Skill_B",
                image_path=Path("b0.png"),
                render_mode="full_color",
            ),
        ]

        similarities = torch.tensor([0.80, 0.76, 0.55])
        top_matches = build_top_matches(similarities, records, topk=3)

        stats = build_similarity_debug_stats(similarities, records, top_matches)

        self.assertEqual(stats["top_identity_summaries"][0]["skill_card_id"], "Skill_A")
        self.assertAlmostEqual(stats["top_identity_summaries"][0]["best_similarity"], 0.80)
        self.assertIn("full_color_flash", stats["top_identity_summaries"][0]["render_modes"])
        self.assertEqual(stats["topk_identity_votes"][0]["count"], 2)
        self.assertAlmostEqual(stats["best_other_identity_similarity"], 0.55)
        self.assertAlmostEqual(stats["top1_margin"], 0.25)

    def test_empty_prediction_path_skips_ranking(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "frame.png"
            output_dir = temp_path / "out"
            Image.new("RGB", (24, 24), color=(0, 0, 0)).save(image_path)
            runtime = SimpleNamespace(checkpoint_path=temp_path / "checkpoint.pt")
            canonical_index = CanonicalIndex(
                records=[CanonicalImageRecord(skill_card_id="Skill_A", character_id="Skill_A", image_path=image_path)],
                embeddings=None,
            )

            payload = process_image(
                image_path=image_path,
                output_dir=output_dir,
                detector=lambda path, threshold: [PortraitDetection(box_xyxy=(0, 0, 12, 12), confidence=0.9)],
                runtime=runtime,
                canonical_index=canonical_index,
                canonical_manifest_path=temp_path / "manifest.jsonl",
                yolo_weights_path=temp_path / "weights.pt",
                topk=3,
                conf_threshold=0.25,
                overlay_writer=lambda output_path, image, detections: None,
                predict_crop_fn=lambda crop, runtime, canonical_index, topk: QueryPrediction(
                    predicted_empty=True,
                    predicted_empty_probability=0.9,
                    predicted_skill_card_id=None,
                    predicted_character_id=None,
                    top_matches=[],
                    debug_stats={"status": "empty"},
                ),
            )

        detection = payload["detections"][0]
        self.assertTrue(detection["predicted_empty"])
        self.assertEqual(detection["top_matches"], [])
        self.assertEqual(detection["debug_stats"]["status"], "empty")

    def test_sort_detections_orders_by_center_y_then_center_x(self) -> None:
        detections = [
            PortraitDetection(box_xyxy=(40, 30, 50, 40), confidence=0.9),
            PortraitDetection(box_xyxy=(10, 10, 20, 20), confidence=0.9),
            PortraitDetection(box_xyxy=(30, 10, 40, 20), confidence=0.9),
        ]

        ordered = sort_detections(detections)

        self.assertEqual([item.box_xyxy for item in ordered], [(10, 10, 20, 20), (30, 10, 40, 20), (40, 30, 50, 40)])

    def test_collect_input_images_sorts_directory_matches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "b.png").write_bytes(b"b")
            (temp_path / "a.png").write_bytes(b"a")
            (temp_path / "c.jpg").write_bytes(b"c")

            images = collect_input_images(temp_path, "*.png")

        self.assertEqual([path.name for path in images], ["a.png", "b.png"])

    def test_crop_to_box_trims_bottom_to_canonical_aspect_ratio(self) -> None:
        image = Image.new("RGB", (300, 300), color=(10, 20, 30))

        crop = crop_to_box(image, (10, 20, 171, 175))

        self.assertEqual(crop.size, (161, 126))

    def test_process_image_writes_json_crops_and_calls_overlay_writer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "raid.png"
            output_dir = temp_path / "out"
            Image.new("RGB", (40, 40), color=(32, 64, 96)).save(image_path)
            runtime = SimpleNamespace(checkpoint_path=temp_path / "checkpoint.pt")
            canonical_index = CanonicalIndex(
                records=[CanonicalImageRecord(skill_card_id="Skill_A", character_id="Skill_A", image_path=image_path)],
                embeddings=None,
            )
            overlay_calls: list[tuple[Path, int]] = []

            payload = process_image(
                image_path=image_path,
                output_dir=output_dir,
                detector=lambda path, threshold: [PortraitDetection(box_xyxy=(5, 6, 20, 22), confidence=0.77)],
                runtime=runtime,
                canonical_index=canonical_index,
                canonical_manifest_path=temp_path / "manifest.jsonl",
                yolo_weights_path=temp_path / "weights.pt",
                topk=2,
                conf_threshold=0.25,
                overlay_writer=lambda output_path, image, detections: overlay_calls.append((output_path, len(detections))),
                predict_crop_fn=lambda crop, runtime, canonical_index, topk: QueryPrediction(
                    predicted_empty=False,
                    predicted_empty_probability=0.1,
                    predicted_skill_card_id="Skill_A",
                    predicted_character_id="Skill_A",
                    top_matches=[
                        {
                            "rank": 1,
                            "skill_card_id": "Skill_A",
                            "character_id": "Skill_A",
                            "canonical_image_path": str(image_path.resolve()),
                            "similarity": 0.99,
                        }
                    ],
                    debug_stats={
                        "top1_margin": 0.44,
                        "topk_identity_votes": [{"skill_card_id": "Skill_A", "count": 1, "best_similarity": 0.99}],
                    },
                ),
            )

            json_payload = json.loads(Path(payload["json_path"]).read_text(encoding="utf-8"))
            crop_exists = Path(json_payload["detections"][0]["crop_path"]).exists()

        self.assertEqual(len(json_payload["detections"]), 1)
        self.assertTrue(crop_exists)
        self.assertEqual(json_payload["detections"][0]["predicted_skill_card_id"], "Skill_A")
        self.assertAlmostEqual(json_payload["detections"][0]["debug_stats"]["top1_margin"], 0.44)
        self.assertEqual(overlay_calls[0][0].name, "raid.raid-skill.overlay.png")
        self.assertEqual(overlay_calls[0][1], 1)

    def test_run_cli_processes_sorted_folder_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_dir = temp_path / "frames"
            input_dir.mkdir()
            Image.new("RGB", (10, 10)).save(input_dir / "b.png")
            Image.new("RGB", (10, 10)).save(input_dir / "a.png")
            output_dir = temp_path / "out"
            processed: list[str] = []
            args = Namespace(
                input=str(input_dir),
                output=str(output_dir),
                glob="*.png",
                yolo_weights=temp_path / "weights.pt",
                skill_checkpoint=temp_path / "checkpoint.pt",
                canonical_manifest=temp_path / "manifest.jsonl",
                device="cpu",
                topk=2,
                conf_threshold=0.25,
            )

            original_module = sys.modules["tools.raid_skill_test"]
            original_load_skill_runtime = original_module.load_skill_runtime
            original_load_canonical_records = original_module.load_canonical_records
            original_build_canonical_index = original_module.build_canonical_index
            original_create_yolo_detector = original_module.create_yolo_detector
            original_process_image = original_module.process_image
            try:
                original_module.load_skill_runtime = lambda checkpoint, device: SimpleNamespace(
                    resolved_device_name="cpu",
                    checkpoint_path=Path(checkpoint),
                )
                original_module.load_canonical_records = lambda manifest: [
                    CanonicalImageRecord(skill_card_id="Skill_A", character_id="Skill_A", image_path=input_dir / "a.png")
                ]
                original_module.build_canonical_index = lambda records, runtime: CanonicalIndex(records=records, embeddings=None)
                original_module.create_yolo_detector = lambda weights, device_name: (lambda image_path, conf_threshold: [])
                original_module.process_image = lambda **kwargs: processed.append(kwargs["image_path"].name) or {"json_path": "ignored"}

                exit_code = run_cli(args)
            finally:
                original_module.load_skill_runtime = original_load_skill_runtime
                original_module.load_canonical_records = original_load_canonical_records
                original_module.build_canonical_index = original_build_canonical_index
                original_module.create_yolo_detector = original_create_yolo_detector
                original_module.process_image = original_process_image

        self.assertEqual(exit_code, 0)
        self.assertEqual(processed, ["a.png", "b.png"])


if __name__ == "__main__":
    unittest.main()
