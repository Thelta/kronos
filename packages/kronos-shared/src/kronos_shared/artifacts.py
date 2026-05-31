from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(slots=True)
class GalleryPrototype:
    character_id: str
    embedding: list[float]
    sample_count: int


@dataclass(slots=True)
class GalleryArtifact:
    model_name: str
    embedding_dim: int
    prototype_strategy: str
    prototypes: list[GalleryPrototype] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict) -> "GalleryArtifact":
        return cls(
            model_name=payload["model_name"],
            embedding_dim=int(payload["embedding_dim"]),
            prototype_strategy=payload["prototype_strategy"],
            prototypes=[GalleryPrototype(**item) for item in payload.get("prototypes", [])],
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class RecognitionMatch:
    character_id: str
    similarity: float


@dataclass(slots=True)
class RecognitionResult:
    character_id: str
    similarity: float | None
    star_value: int | None
    assist: bool | None
    predicted_empty: bool = False
    predicted_empty_probability: float = 0.0
    top_matches: list[RecognitionMatch] = field(default_factory=list)


def load_gallery_artifact(path: Path) -> GalleryArtifact:
    with path.open("r", encoding="utf-8") as handle:
        return GalleryArtifact.from_dict(json.load(handle))


def save_gallery_artifact(path: Path, artifact: GalleryArtifact) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(artifact.to_dict(), handle, indent=2, ensure_ascii=False)
