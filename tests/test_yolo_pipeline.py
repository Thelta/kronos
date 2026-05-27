from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "kronos-train" / "src"))

from kronos_train.yolo_config import YoloMergedDatasetConfig, YoloTrainConfig  # noqa: E402
from kronos_train.yolo_pipeline import prepare_yolo_dataset  # noqa: E402


class YoloPipelineTests(unittest.TestCase):
    def test_prepare_yolo_dataset_builds_merged_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            synthetic_images = root / "synthetic" / "images"
            synthetic_labels = root / "synthetic" / "labels"
            real_review = root / "review"
            synthetic_images.mkdir(parents=True)
            synthetic_labels.mkdir(parents=True)
            real_review.mkdir(parents=True)

            for stem in ("syn_a", "syn_b"):
                (synthetic_images / f"{stem}.jpg").write_bytes(b"jpg")
                (synthetic_labels / f"{stem}.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")

            for stem, status in (("real_keep_orig", "accepted"), ("real_skip_orig", "pending")):
                (real_review / f"{stem}.jpg").write_bytes(b"jpg")
                (real_review / f"{stem.replace('_orig', '')}.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")

            (real_review / "review_status.json").write_text(
                json.dumps(
                    {
                        "real_keep_orig.jpg": "accepted",
                        "real_skip_orig.jpg": "pending",
                    }
                ),
                encoding="utf-8",
            )

            config = YoloTrainConfig(
                dataset=YoloMergedDatasetConfig(
                    output_dir=str(root / "out"),
                    synthetic_images_dir=str(synthetic_images),
                    synthetic_labels_dir=str(synthetic_labels),
                    reviewed_real_dir=str(real_review),
                    review_status_file=str(real_review / "review_status.json"),
                    seed=3,
                )
            )

            summary = prepare_yolo_dataset(config)

            output_dir = Path(summary["dataset_dir"])
            self.assertEqual(summary["total_count"], 3)
            self.assertEqual(summary["synthetic_count"], 2)
            self.assertEqual(summary["reviewed_real_count"], 1)
            self.assertTrue((output_dir / "data.yaml").exists())
            self.assertTrue((output_dir / "dataset_manifest.json").exists())

            images = list((output_dir / "images" / "train").glob("*")) + list((output_dir / "images" / "val").glob("*"))
            self.assertEqual(len(images), 3)
            self.assertTrue(any(path.stem == "real_real_keep_orig" for path in images))
            self.assertFalse(any("skip" in path.stem for path in images))

    def test_prepare_yolo_dataset_clears_stale_split_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            images_dir = root / "synthetic" / "images"
            labels_dir = root / "synthetic" / "labels"
            review_dir = root / "review"
            images_dir.mkdir(parents=True)
            labels_dir.mkdir(parents=True)
            review_dir.mkdir(parents=True)

            for stem in ("a", "b"):
                (images_dir / f"{stem}.jpg").write_bytes(b"jpg")
                (labels_dir / f"{stem}.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")

            (review_dir / "real_keep_orig.jpg").write_bytes(b"jpg")
            (review_dir / "real_keep.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
            (review_dir / "review_status.json").write_text(
                json.dumps({"real_keep_orig.jpg": "accepted"}),
                encoding="utf-8",
            )

            config = YoloTrainConfig(
                dataset=YoloMergedDatasetConfig(
                    output_dir=str(root / "out"),
                    synthetic_images_dir=str(images_dir),
                    synthetic_labels_dir=str(labels_dir),
                    reviewed_real_dir=str(review_dir),
                    review_status_file=str(review_dir / "review_status.json"),
                    seed=1,
                )
            )

            prepare_yolo_dataset(config)
            stale_file = root / "out" / "images" / "train" / "stale.jpg"
            stale_file.write_bytes(b"stale")

            prepare_yolo_dataset(config)

            self.assertFalse(stale_file.exists())


if __name__ == "__main__":
    unittest.main()
