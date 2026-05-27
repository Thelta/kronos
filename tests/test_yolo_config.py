from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "kronos-train" / "src"))

from kronos_train.yolo_config import apply_overrides, config_from_dict  # noqa: E402


class YoloConfigTests(unittest.TestCase):
    def test_config_from_dict_reads_merged_inputs(self) -> None:
        config = config_from_dict(
            {
                "dataset": {
                    "output_dir": "tmp/out",
                    "synthetic_images_dir": "synthetic/images",
                    "synthetic_labels_dir": "synthetic/labels",
                    "reviewed_real_dir": "review",
                    "review_status_file": "review/review_status.json",
                },
                "trainer": {
                    "weights": "yolo26n.pt",
                    "epochs": 50,
                },
            }
        )

        self.assertEqual(config.dataset.synthetic_images_dir, "synthetic/images")
        self.assertEqual(config.dataset.reviewed_real_dir, "review")
        self.assertEqual(config.dataset.include_review_statuses, ["accepted", "rejected"])
        self.assertEqual(config.trainer.weights, "yolo26n.pt")

    def test_apply_overrides_updates_merged_paths(self) -> None:
        payload = {
            "dataset": {
                "output_dir": "tmp/out",
                "synthetic_images_dir": "a",
                "synthetic_labels_dir": "b",
                "reviewed_real_dir": "c",
                "review_status_file": "d",
            },
            "trainer": {},
        }

        updated = apply_overrides(
            payload,
            [
                "dataset.reviewed_real_dir=D:/data/review",
                "dataset.output_dir=training_runs/yolo",
                "trainer.epochs=25",
            ],
        )

        self.assertEqual(updated["dataset"]["reviewed_real_dir"], "D:/data/review")
        self.assertEqual(updated["dataset"]["output_dir"], "training_runs/yolo")
        self.assertEqual(updated["trainer"]["epochs"], 25)


if __name__ == "__main__":
    unittest.main()
