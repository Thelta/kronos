from __future__ import annotations

from typing import Any

from .model import ArcMarginProduct


def build_skill_model(config: Any, num_classes: int) -> Any:
    import timm
    import torch.nn as nn
    import torch.nn.functional as F

    class SkillArcFaceModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = timm.create_model(
                config.model.model_name,
                pretrained=config.model.pretrained,
                num_classes=0,
                global_pool="avg",
            )
            feature_dim = int(getattr(self.backbone, "head_hidden_size", self.backbone.num_features))
            hidden = max(1, int(config.model.head_hidden_dim))
            self.embedding_layer = nn.Linear(feature_dim, config.model.embedding_dim, bias=False)
            self.embedding_bn = nn.BatchNorm1d(config.model.embedding_dim)
            self.empty_head = nn.Sequential(
                nn.Linear(feature_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, 1),
            )
            self.arcface = ArcMarginProduct(
                in_features=config.model.embedding_dim,
                out_features=num_classes,
                scale=config.arcface.scale,
                margin=config.arcface.margin,
            ).module

        def forward(self, images: Any, labels: Any | None = None) -> dict[str, Any]:
            feature_map = self.backbone.forward_features(images)
            if isinstance(feature_map, (list, tuple)):
                feature_map = feature_map[-1]
            pooled_features = self.backbone.forward_head(feature_map, pre_logits=True)
            if isinstance(pooled_features, (list, tuple)):
                pooled_features = pooled_features[-1]
            if pooled_features.ndim > 2:
                pooled_features = pooled_features.flatten(1)

            embedding = self.embedding_bn(self.embedding_layer(pooled_features))
            normalized_embedding = F.normalize(embedding, dim=1)
            return {
                "embedding": normalized_embedding,
                "identity_logits": self.arcface(normalized_embedding, labels),
                "empty_logits": self.empty_head(pooled_features).squeeze(1),
            }

    return SkillArcFaceModel()
