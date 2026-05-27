from __future__ import annotations

from pathlib import Path

from kronos_shared import GalleryArtifact, load_gallery_artifact


def load_identity_gallery(path: str | Path) -> GalleryArtifact:
    return load_gallery_artifact(Path(path))
