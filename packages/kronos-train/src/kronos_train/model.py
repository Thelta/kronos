from __future__ import annotations

import math
import random
from typing import Any


class ArcMarginProduct:
    def __init__(self, *, in_features: int, out_features: int, scale: float, margin: float) -> None:
        import torch
        from torch import nn

        class _Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.empty(out_features, in_features))
                nn.init.xavier_uniform_(self.weight)
                self.scale = scale
                self.margin = margin
                self.cos_m = math.cos(margin)
                self.sin_m = math.sin(margin)
                self.th = math.cos(math.pi - margin)
                self.mm = math.sin(math.pi - margin) * margin

            def forward(self, embeddings: Any, labels: Any | None) -> Any:
                import torch
                import torch.nn.functional as F

                # ArcFace is numerically sensitive under autocast. Keep it in fp32 and
                # clamp cosine slightly inside [-1, 1] to avoid sqrt/acos edge NaNs.
                with torch.autocast(device_type=embeddings.device.type, enabled=False):
                    normalized_embeddings = F.normalize(embeddings.float(), dim=1)
                    normalized_weight = F.normalize(self.weight.float(), dim=1)
                    cosine = F.linear(normalized_embeddings, normalized_weight).clamp(-0.999999, 0.999999)
                    if labels is None:
                        return cosine * self.scale

                    sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0.0, 1.0))
                    phi = cosine * self.cos_m - sine * self.sin_m
                    phi = torch.where(cosine > self.th, phi, cosine - self.mm)
                    one_hot = F.one_hot(labels, num_classes=cosine.shape[1]).float()
                    logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
                    return logits * self.scale

        self.module = _Module()


def build_model(config: Any, num_classes: int) -> Any:
    import timm
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def validate_roi(name: str, roi: list[float]) -> tuple[float, float, float, float]:
        if len(roi) != 4:
            raise ValueError(f"{name} must contain four normalized coordinates [x0, y0, x1, y1].")
        x0, y0, x1, y1 = (float(value) for value in roi)
        if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
            raise ValueError(f"{name} must satisfy 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1.")
        return x0, y0, x1, y1

    class PixelRoiEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )

        def forward(self, images: Any) -> Any:
            return self.features(images)

    class AttentionPool2d(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.attention_logits = nn.Conv2d(channels, 1, kernel_size=1)

        def forward(self, feature_map: Any, return_attention: bool = False) -> Any:
            batch_size, channels, height, width = feature_map.shape
            attention = self.attention_logits(feature_map).view(batch_size, 1, height * width)
            attention = torch.softmax(attention, dim=2)
            features = feature_map.view(batch_size, channels, height * width)
            pooled = torch.sum(features * attention, dim=2)
            if return_attention:
                return pooled, attention.view(batch_size, 1, height, width)
            return pooled

    class MobileNetArcFaceModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = timm.create_model(
                config.model.model_name,
                pretrained=config.model.pretrained,
                num_classes=0,
                global_pool="avg",
            )
            feature_dim = int(getattr(self.backbone, "head_hidden_size", self.backbone.num_features))
            hidden = int(config.model.roi_hidden_dim)
            empty_hidden = hidden // 2 if hidden >= 2 else 1
            self.embedding_layer = nn.Linear(feature_dim, config.model.embedding_dim, bias=False)
            self.embedding_bn = nn.BatchNorm1d(config.model.embedding_dim)
            self.empty_head = nn.Sequential(
                nn.Linear(feature_dim, empty_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(empty_hidden, 1),
            )
            self.star_roi = validate_roi("model.star_roi", config.model.star_roi)
            self.assist_roi = validate_roi("model.assist_roi", config.model.assist_roi)
            self.roi_input_size = int(config.model.roi_input_size)
            self.star_box_expand_x = float(config.model.star_box_expand_x)
            self.star_box_expand_y = float(config.model.star_box_expand_y)
            self.star_box_train_prob = float(config.model.star_box_train_prob)
            self.star_box_jitter = float(config.model.star_box_jitter)
            self.star_lowres_prob = float(config.model.star_lowres_prob)
            self.star_lowres_min_size = int(config.model.star_lowres_min_size)
            self.star_lowres_max_size = int(config.model.star_lowres_max_size)
            self.star_encoder = PixelRoiEncoder()
            self.star_pool = AttentionPool2d(channels=128)
            self.star_head = nn.Sequential(
                nn.Linear(128, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1),
                nn.Linear(hidden, 10),
            )
            self.assist_encoder = PixelRoiEncoder()
            self.assist_head = nn.Sequential(
                nn.Linear(128, hidden // 2 if hidden >= 2 else 1),
                nn.ReLU(inplace=True),
                nn.Linear(hidden // 2 if hidden >= 2 else 1, 1),
            )
            self.arcface = ArcMarginProduct(
                in_features=config.model.embedding_dim,
                out_features=num_classes,
                scale=config.arcface.scale,
                margin=config.arcface.margin,
            ).module

        def crop_roi(self, images: Any, card_boxes: Any | None, roi: tuple[float, float, float, float]) -> Any:
            height, width = int(images.shape[-2]), int(images.shape[-1])
            x0, y0, x1, y1 = roi
            if card_boxes is None:
                default_boxes = images.new_tensor([0.0, 0.0, float(width), float(height)]).unsqueeze(0)
                card_boxes = default_boxes.repeat(images.shape[0], 1)

            roi_crops = []
            for image, box in zip(images, card_boxes):
                card_left = min(width - 1, max(0, int(math.floor(float(box[0].item())))))
                card_top = min(height - 1, max(0, int(math.floor(float(box[1].item())))))
                card_right = max(card_left + 1, min(width, int(math.ceil(float(box[2].item())))))
                card_bottom = max(card_top + 1, min(height, int(math.ceil(float(box[3].item())))))
                card_width = max(1, card_right - card_left)
                card_height = max(1, card_bottom - card_top)

                left = min(width - 1, max(0, int(math.floor(card_left + x0 * card_width))))
                top = min(height - 1, max(0, int(math.floor(card_top + y0 * card_height))))
                right = max(left + 1, min(width, int(math.ceil(card_left + x1 * card_width))))
                bottom = max(top + 1, min(height, int(math.ceil(card_top + y1 * card_height))))

                roi_image = image[:, top:bottom, left:right].unsqueeze(0)
                roi_crops.append(
                    F.interpolate(
                        roi_image,
                        size=(self.roi_input_size, self.roi_input_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                )

            return torch.cat(roi_crops, dim=0)
        def crop_boxes(self, images: Any, boxes: Any) -> Any:
            height, width = int(images.shape[-2]), int(images.shape[-1])
            roi_crops = []
            for image, box in zip(images, boxes):
                left = min(width - 1, max(0, int(math.floor(float(box[0].item())))))
                top = min(height - 1, max(0, int(math.floor(float(box[1].item())))))
                right = max(left + 1, min(width, int(math.ceil(float(box[2].item())))))
                bottom = max(top + 1, min(height, int(math.ceil(float(box[3].item())))))
                roi_image = image[:, top:bottom, left:right].unsqueeze(0)
                roi_crops.append(
                    F.interpolate(
                        roi_image,
                        size=(self.roi_input_size, self.roi_input_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                )
            return torch.cat(roi_crops, dim=0)

        def build_star_training_boxes(self, images: Any, card_boxes: Any | None, star_boxes: Any | None) -> Any | None:
            if not self.training or star_boxes is None:
                return None
            height, width = int(images.shape[-2]), int(images.shape[-1])
            if card_boxes is None:
                default_boxes = images.new_tensor([0.0, 0.0, float(width), float(height)]).unsqueeze(0)
                card_boxes = default_boxes.repeat(images.shape[0], 1)

            training_boxes = []
            has_any = False
            for card_box, star_box in zip(card_boxes, star_boxes):
                if not bool(torch.isfinite(star_box).all().item()) or random.random() > self.star_box_train_prob:
                    training_boxes.append(star_box.new_tensor([float("nan")] * 4))
                    continue

                star_left = float(star_box[0].item())
                star_top = float(star_box[1].item())
                star_right = float(star_box[2].item())
                star_bottom = float(star_box[3].item())
                star_width = max(1.0, star_right - star_left)
                star_height = max(1.0, star_bottom - star_top)
                center_x = 0.5 * (star_left + star_right)
                center_y = 0.5 * (star_top + star_bottom)

                center_x += random.uniform(-self.star_box_jitter, self.star_box_jitter) * star_width
                center_y += random.uniform(-self.star_box_jitter, self.star_box_jitter) * star_height
                half_width = 0.5 * star_width * (1.0 + (2.0 * self.star_box_expand_x))
                half_height = 0.5 * star_height * (1.0 + (2.0 * self.star_box_expand_y))

                card_left = float(card_box[0].item())
                card_top = float(card_box[1].item())
                card_right = float(card_box[2].item())
                card_bottom = float(card_box[3].item())

                left = max(0.0, min(center_x - half_width, card_right - 1.0))
                top = max(0.0, min(center_y - half_height, card_bottom - 1.0))
                right = min(float(width), max(center_x + half_width, left + 1.0))
                bottom = min(float(height), max(center_y + half_height, top + 1.0))

                left = max(card_left, left)
                top = max(card_top, top)
                right = min(card_right, right)
                bottom = min(card_bottom, bottom)

                if right <= left:
                    right = min(card_right, left + 1.0)
                if bottom <= top:
                    bottom = min(card_bottom, top + 1.0)

                training_boxes.append(star_box.new_tensor([left, top, right, bottom]))
                has_any = True

            if not has_any:
                return None
            return torch.stack(training_boxes, dim=0)

        def apply_star_lowres_augmentation(self, star_images: Any) -> Any:
            if not self.training or self.star_lowres_prob <= 0.0:
                return star_images

            degraded = []
            min_size = max(4, min(self.star_lowres_min_size, self.star_lowres_max_size))
            max_size = max(min_size, self.star_lowres_max_size)
            for image in star_images:
                image = image.unsqueeze(0)
                if random.random() < self.star_lowres_prob:
                    target_size = random.randint(min_size, max_size)
                    lowres = F.interpolate(image, size=(target_size, target_size), mode="area")
                    image = F.interpolate(
                        lowres,
                        size=(self.roi_input_size, self.roi_input_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                degraded.append(image)
            return torch.cat(degraded, dim=0)
        def forward(
            self,
            images: Any,
            labels: Any | None = None,
            card_boxes: Any | None = None,
            star_boxes: Any | None = None,
            return_debug: bool = False,
        ) -> dict[str, Any]:
            feature_map = self.backbone.forward_features(images)
            if isinstance(feature_map, (list, tuple)):
                feature_map = feature_map[-1]
            if feature_map.ndim != 4:
                raise ValueError(
                    f"Expected backbone.forward_features() to return a 4D feature map, got shape {tuple(feature_map.shape)}."
                )
            pooled_features = self.backbone.forward_head(feature_map, pre_logits=True)
            if isinstance(pooled_features, (list, tuple)):
                pooled_features = pooled_features[-1]
            if pooled_features.ndim > 2:
                pooled_features = torch.flatten(pooled_features, 1)

            embedding = self.embedding_bn(self.embedding_layer(pooled_features))
            normalized_embedding = F.normalize(embedding, dim=1)
            identity_logits = self.arcface(normalized_embedding, labels)
            star_images = self.crop_roi(images, card_boxes, self.star_roi)
            star_training_boxes = self.build_star_training_boxes(images, card_boxes, star_boxes)
            if star_training_boxes is not None:
                replacement_mask = torch.isfinite(star_training_boxes).all(dim=1)
                if bool(replacement_mask.any().item()):
                    star_images = star_images.clone()
                    star_images[replacement_mask] = self.crop_boxes(images[replacement_mask], star_training_boxes[replacement_mask])
            star_images = self.apply_star_lowres_augmentation(star_images)
            assist_images = self.crop_roi(images, card_boxes, self.assist_roi)
            star_feature_map = self.star_encoder(star_images)
            assist_feature_map = self.assist_encoder(assist_images)
            star_attention_map = None
            if return_debug:
                star_features, star_attention_map = self.star_pool(star_feature_map, return_attention=True)
            else:
                star_features = self.star_pool(star_feature_map)
            assist_features = F.adaptive_avg_pool2d(assist_feature_map, output_size=1).flatten(1)
            outputs = {
                "embedding": normalized_embedding,
                "identity_logits": identity_logits,
                "empty_logits": self.empty_head(pooled_features).squeeze(1),
                "star_state_logits": self.star_head(star_features),
                "assist_logits": self.assist_head(assist_features).squeeze(1),
            }
            if return_debug:
                outputs["star_attention_map"] = star_attention_map
                outputs["star_images"] = star_images
            return outputs

    return MobileNetArcFaceModel()
