from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "kronos-train" / "src"))

from kronos_train.data import degrade_tiny_card_image, build_star_slot_indices, star_color_to_index  # noqa: E402


class DataTests(unittest.TestCase):
    def test_star_color_to_index(self) -> None:
        self.assertEqual(star_color_to_index("yellow"), 0)
        self.assertEqual(star_color_to_index("blue"), 1)
        self.assertEqual(star_color_to_index(" Blue "), 1)

    def test_build_star_slot_indices(self) -> None:
        self.assertEqual(build_star_slot_indices(star_value=3, star_color="yellow"), [1, 1, 1, 0, 0])
        self.assertEqual(build_star_slot_indices(star_value=2, star_color="blue"), [2, 2, 0, 0, 0])

    def test_degrade_tiny_card_image_preserves_original_size(self) -> None:
        from PIL import Image
        import random

        image = Image.new("RGB", (275, 275), color=(255, 255, 255))
        degraded = degrade_tiny_card_image(
            image,
            probability=1.0,
            sizes=[[63, 50]],
            size_jitter=0.0,
            rng=random.Random(7),
        )

        self.assertEqual(degraded.size, image.size)


if __name__ == "__main__":
    unittest.main()
