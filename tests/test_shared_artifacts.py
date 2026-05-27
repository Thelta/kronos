from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "kronos-shared" / "src"))

from kronos_shared import (  # noqa: E402
    GalleryArtifact,
    GalleryPrototype,
    SyntheticClassificationRow,
    load_gallery_artifact,
    load_synthetic_rows,
    save_gallery_artifact,
    save_synthetic_rows,
)


class SharedArtifactTests(unittest.TestCase):
    def test_gallery_artifact_round_trip(self) -> None:
        artifact = GalleryArtifact(
            model_name="mobilenetv4",
            embedding_dim=256,
            prototype_strategy="mean_l2",
            prototypes=[GalleryPrototype(character_id="1001", embedding=[0.1, 0.2], sample_count=3)],
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "gallery.json"
            save_gallery_artifact(path, artifact)
            loaded = load_gallery_artifact(path)
        self.assertEqual(loaded.model_name, "mobilenetv4")
        self.assertEqual(loaded.prototypes[0].character_id, "1001")

    def test_synthetic_rows_accept_legacy_split_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "manifest.jsonl"
            rows = [
                SyntheticClassificationRow(
                    image_path="images/sample.jpg",
                    subset="train_query",
                    mode="classification",
                    character_id="1001",
                    portrait_path="portraits/1001.png",
                    attack_type="Explosion",
                    role="Supporter",
                    level=90,
                    star_value=3,
                    star_color="yellow",
                    assist=False,
                    starter=False,
                    seed=1,
                    background_kind="gradient",
                    scale_jitter=1.0,
                    card_box=[0, 0, 10, 10],
                    portrait_box=[0, 0, 8, 8],
                )
            ]
            save_synthetic_rows(path, rows)
            loaded = load_synthetic_rows(path)
        self.assertEqual(loaded[0].subset, "train_query")


if __name__ == "__main__":
    unittest.main()
