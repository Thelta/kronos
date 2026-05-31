from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "kronos-shared" / "src"))
sys.path.insert(0, str(ROOT / "packages" / "kronos-train" / "src"))

from kronos_shared import SyntheticClassificationRow  # noqa: E402
from kronos_train.pipeline import predict_image  # noqa: E402
from kronos_train.training_config import TrainConfig  # noqa: E402


class DummyEmptyModel:
    def __call__(self, images, labels=None, card_boxes=None, star_boxes=None, return_debug=False):  # noqa: ANN001
        import torch

        return {"empty_logits": torch.tensor([10.0], dtype=torch.float32, device=images.device)}


def make_card_row(character_id: str, *, empty: bool = False) -> SyntheticClassificationRow:
    return SyntheticClassificationRow(
        image_path="images/sample.jpg",
        subset="train",
        mode="classification",
        character_id=character_id,
        portrait_path=None if empty else "portraits/sample.png",
        attack_type=None if empty else "Explosion",
        role=None if empty else "Supporter",
        level=None if empty else 90,
        star_value=None if empty else 3,
        star_color=None if empty else "yellow",
        assist=False,
        starter=False,
        seed=1,
        background_kind="gradient",
        card_box=[0, 0, 10, 10],
        portrait_box=[0, 0, 8, 8],
        empty=empty,
    )


class PipelineTests(unittest.TestCase):
    def test_predict_image_returns_empty_without_gallery_lookup(self) -> None:
        if find_spec("torch") is None:
            self.skipTest("torch is not installed in this environment")
        from PIL import Image
        import torch

        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "sample.png"
            Image.new("RGB", (32, 32), color=(255, 255, 255)).save(image_path)

            config = TrainConfig()
            config.data.manifest_path = str(Path(tmp_dir) / "manifest.jsonl")
            config.data.image_size = 32
            rows = [make_card_row("1001"), make_card_row("empty", empty=True)]

            captured = io.StringIO()
            with (
                patch("kronos_train.pipeline.load_model_from_checkpoint", return_value=(DummyEmptyModel(), config, ["1001"], torch.device("cpu"))),
                patch("kronos_train.pipeline.load_rows", return_value=rows),
                patch(
                    "kronos_train.pipeline.prepare_runtime_subsets",
                    return_value={"gallery": [rows[0]], "train_query": rows, "val_query": rows, "test_query": rows},
                ),
                redirect_stdout(captured),
            ):
                predict_image(Path("checkpoint.pt"), image_path, "cpu")

        payload = json.loads(captured.getvalue())
        self.assertEqual(payload["predicted_character_id"], "empty")
        self.assertTrue(payload["predicted_empty"])
        self.assertEqual(payload["top_matches"], [])


if __name__ == "__main__":
    unittest.main()
