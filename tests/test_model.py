from __future__ import annotations

import sys
import unittest
from importlib.util import find_spec
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "kronos-train" / "src"))

from kronos_train.model import build_model  # noqa: E402
from kronos_train.skill_model import build_skill_model  # noqa: E402
from kronos_train.skill_training_config import SkillTrainConfig  # noqa: E402
from kronos_train.training_config import TrainConfig  # noqa: E402


class ModelTests(unittest.TestCase):
    def test_mobilenetv4_forward_with_roi_heads(self) -> None:
        if find_spec("torch") is None:
            self.skipTest("torch is not installed in this environment")
        import torch

        config = TrainConfig()
        config.model.pretrained = False
        model = build_model(config, num_classes=10)
        outputs = model(torch.randn(4, 3, 224, 224), torch.tensor([0, 1, 2, 3], dtype=torch.long))

        self.assertEqual(tuple(outputs["embedding"].shape), (4, 256))
        self.assertEqual(tuple(outputs["identity_logits"].shape), (4, 10))
        self.assertEqual(tuple(outputs["empty_logits"].shape), (4,))
        self.assertEqual(tuple(outputs["star_state_logits"].shape), (4, 10))
        self.assertEqual(tuple(outputs["assist_logits"].shape), (4,))

    def test_skill_model_forward_with_empty_head(self) -> None:
        if find_spec("torch") is None:
            self.skipTest("torch is not installed in this environment")
        import torch

        config = SkillTrainConfig()
        config.model.pretrained = False
        model = build_skill_model(config, num_classes=7)
        outputs = model(torch.randn(3, 3, 224, 224), torch.tensor([0, 1, 2], dtype=torch.long))

        self.assertEqual(tuple(outputs["embedding"].shape), (3, 256))
        self.assertEqual(tuple(outputs["identity_logits"].shape), (3, 7))
        self.assertEqual(tuple(outputs["empty_logits"].shape), (3,))


if __name__ == "__main__":
    unittest.main()
