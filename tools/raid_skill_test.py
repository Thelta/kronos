from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


def _ensure_workspace_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    shared_src = root / "packages" / "kronos-shared" / "src"
    train_src = root / "packages" / "kronos-train" / "src"
    analyzer_src = root / "src"
    for path in (shared_src, train_src, analyzer_src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


_ensure_workspace_paths()

DEFAULT_YOLO_WEIGHTS = Path("training_runs/yolo/portrait_merged_yolo26n/weights/best.pt")
DEFAULT_SKILL_CHECKPOINT = Path("training_runs/kronos_skill_arcface/best.pt")
DEFAULT_CANONICAL_MANIFEST = Path(r"D:\kronos-training\skill_canonical\manifest.jsonl")
DEFAULT_GLOB = "*.png"
DEFAULT_TOPK = 5
DEFAULT_CONF_THRESHOLD = 0.25
CANONICAL_BATCH_SIZE = 32
CANONICAL_CARD_WIDTH = 224.0
CANONICAL_CARD_HEIGHT = 176.0
CANONICAL_CARD_ASPECT_RATIO = CANONICAL_CARD_WIDTH / CANONICAL_CARD_HEIGHT


@dataclass(frozen=True)
class CanonicalImageRecord:
    skill_card_id: str
    character_id: str | None
    image_path: Path
    render_mode: str | None = None
    variant_index: int | None = None


@dataclass(frozen=True)
class PortraitDetection:
    box_xyxy: tuple[int, int, int, int]
    confidence: float


@dataclass(frozen=True)
class QueryPrediction:
    predicted_empty: bool
    predicted_empty_probability: float
    predicted_skill_card_id: str | None
    predicted_character_id: str | None
    top_matches: list[dict[str, Any]]
    debug_stats: dict[str, Any] | None = None


@dataclass(frozen=True)
class CanonicalIndex:
    records: list[CanonicalImageRecord]
    embeddings: Any


@dataclass(frozen=True)
class SkillRuntime:
    model: Any
    device: Any
    transform: Any
    resolved_device_name: str
    checkpoint_path: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="raid-skill-test")
    parser.add_argument("--input", required=True, help="Image file or directory of images.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--glob", default=DEFAULT_GLOB, help="Glob used when --input points to a directory.")
    parser.add_argument("--yolo-weights", type=Path, default=DEFAULT_YOLO_WEIGHTS)
    parser.add_argument("--skill-checkpoint", type=Path, default=DEFAULT_SKILL_CHECKPOINT)
    parser.add_argument("--canonical-manifest", type=Path, default=DEFAULT_CANONICAL_MANIFEST)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument("--conf-threshold", type=float, default=DEFAULT_CONF_THRESHOLD)
    parser.add_argument("--flash-recovery", action="store_true", help="Try gamma/contrast recovery on bright crops and pick the better variant.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_cli(args)


def run_cli(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_input_images(input_path, args.glob)
    runtime = load_skill_runtime(args.skill_checkpoint, args.device)
    canonical_records = load_canonical_records(args.canonical_manifest)
    canonical_index = build_canonical_index(canonical_records, runtime)
    detector = create_yolo_detector(args.yolo_weights, runtime.resolved_device_name)

    predict_fn = (
        predict_crop_with_flash_recovery if args.flash_recovery else predict_crop
    )

    for image_path in image_paths:
        payload = process_image(
            image_path=image_path,
            output_dir=output_dir,
            detector=detector,
            runtime=runtime,
            canonical_index=canonical_index,
            canonical_manifest_path=args.canonical_manifest,
            yolo_weights_path=args.yolo_weights,
            topk=args.topk,
            conf_threshold=args.conf_threshold,
            predict_crop_fn=predict_fn,
        )
        print(f"{image_path.name} -> {payload['json_path']}")

    return 0


def collect_input_images(input_path: Path, pattern: str) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(path for path in input_path.glob(pattern) if path.is_file())
    raise FileNotFoundError(f"Input not found: {input_path}")


def load_canonical_records(manifest_path: Path) -> list[CanonicalImageRecord]:
    dataset_root = manifest_path.parent
    records: list[CanonicalImageRecord] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            identity = payload.get("skill_card_id") or payload.get("character_id")
            if not identity:
                continue
            image_path = payload.get("image_path")
            if not image_path:
                continue
            records.append(
                CanonicalImageRecord(
                    skill_card_id=str(identity),
                    character_id=payload.get("character_id"),
                    image_path=resolve_manifest_image_path(dataset_root, str(image_path)),
                    render_mode=payload.get("render_mode"),
                    variant_index=payload.get("variant_index"),
                )
            )
    if not records:
        raise ValueError(f"Canonical manifest has no usable identity rows: {manifest_path}")
    return records


def resolve_manifest_image_path(dataset_root: Path, image_path: str) -> Path:
    path = Path(image_path)
    if path.is_absolute():
        return path
    return dataset_root / path


def load_skill_runtime(checkpoint_path: Path, requested_device_name: str) -> SkillRuntime:
    import torch

    from kronos_train.data import build_transforms
    from kronos_train.skill_pipeline import load_model_from_checkpoint

    resolved_device_name = requested_device_name
    if requested_device_name != "cpu" and not torch.cuda.is_available():
        resolved_device_name = "cpu"

    model, config, _, device = load_model_from_checkpoint(checkpoint_path, resolved_device_name)
    transform = build_transforms(config.data.image_size)
    return SkillRuntime(
        model=model,
        device=device,
        transform=transform,
        resolved_device_name=device.type,
        checkpoint_path=checkpoint_path,
    )


def build_canonical_index(
    records: list[CanonicalImageRecord],
    runtime: SkillRuntime,
    *,
    batch_size: int = CANONICAL_BATCH_SIZE,
) -> CanonicalIndex:
    from PIL import Image
    import torch

    embeddings: list[Any] = []
    batch_tensors: list[Any] = []

    with torch.no_grad():
        for record in records:
            image = Image.open(record.image_path).convert("RGB")
            batch_tensors.append(runtime.transform(image))
            if len(batch_tensors) >= batch_size:
                embeddings.append(run_embedding_batch(batch_tensors, runtime))
                batch_tensors = []
        if batch_tensors:
            embeddings.append(run_embedding_batch(batch_tensors, runtime))

    return CanonicalIndex(records=records, embeddings=torch.cat(embeddings, dim=0))


def run_embedding_batch(batch_tensors: list[Any], runtime: SkillRuntime) -> Any:
    import torch

    stacked = torch.stack(batch_tensors).to(runtime.device)
    outputs = runtime.model(stacked, None)
    return outputs["embedding"].detach()


def create_yolo_detector(weights_path: Path, device_name: str) -> Callable[[Path, float], list[PortraitDetection]]:
    from ultralytics import YOLO

    model = YOLO(str(weights_path))

    def detect(image_path: Path, conf_threshold: float) -> list[PortraitDetection]:
        results = model.predict(
            source=str(image_path),
            conf=conf_threshold,
            device=device_name,
            verbose=False,
        )
        return parse_yolo_detections(results[0])

    return detect


def parse_yolo_detections(result: Any) -> list[PortraitDetection]:
    names = getattr(result, "names", {}) or {}
    portrait_class_ids = {int(class_id) for class_id, name in names.items() if name == "portrait"}
    detections: list[PortraitDetection] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return detections

    for box in boxes:
        class_id = int(box.cls.item())
        if portrait_class_ids and class_id not in portrait_class_ids:
            continue
        xyxy = box.xyxy[0].tolist()
        detections.append(
            PortraitDetection(
                box_xyxy=tuple(int(round(value)) for value in xyxy),
                confidence=float(box.conf.item()),
            )
        )
    return sort_detections(detections)


def sort_detections(detections: list[PortraitDetection]) -> list[PortraitDetection]:
    return sorted(detections, key=lambda item: (box_center_y(item.box_xyxy), box_center_x(item.box_xyxy)))


def box_center_x(box_xyxy: tuple[int, int, int, int]) -> float:
    return (float(box_xyxy[0]) + float(box_xyxy[2])) / 2.0


def box_center_y(box_xyxy: tuple[int, int, int, int]) -> float:
    return (float(box_xyxy[1]) + float(box_xyxy[3])) / 2.0


def process_image(
    *,
    image_path: Path,
    output_dir: Path,
    detector: Callable[[Path, float], list[PortraitDetection]],
    runtime: SkillRuntime,
    canonical_index: CanonicalIndex,
    canonical_manifest_path: Path,
    yolo_weights_path: Path,
    topk: int,
    conf_threshold: float,
    overlay_writer: Callable[[Path, Any, list[dict[str, Any]]], None] | None = None,
    predict_crop_fn: Callable[[Any, SkillRuntime, CanonicalIndex, int], QueryPrediction] | None = None,
) -> dict[str, Any]:
    from PIL import Image

    overlay_writer = overlay_writer or write_overlay_image
    predict_crop_fn = predict_crop_fn or predict_crop
    output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    detections = sort_detections(detector(image_path, conf_threshold))

    serialized_detections: list[dict[str, Any]] = []
    for index, detection in enumerate(detections):
        crop = crop_to_box(image, detection.box_xyxy)
        crop_path = crops_dir / f"{image_path.stem}.portrait_{index:02d}.png"
        crop.save(crop_path)
        prediction = predict_crop_fn(crop, runtime, canonical_index, topk)
        serialized_detections.append(
            {
                "index": index,
                "box_xyxy": list(detection.box_xyxy),
                "confidence": detection.confidence,
                "crop_path": str(crop_path.resolve()),
                "predicted_empty": prediction.predicted_empty,
                "predicted_empty_probability": prediction.predicted_empty_probability,
                "predicted_skill_card_id": prediction.predicted_skill_card_id,
                "predicted_character_id": prediction.predicted_character_id,
                "top_matches": prediction.top_matches,
                "debug_stats": prediction.debug_stats or {},
            }
        )

    json_path = output_dir / f"{image_path.stem}.raid-skill.json"
    overlay_path = output_dir / f"{image_path.stem}.raid-skill.overlay.png"
    payload = {
        "input_path": str(image_path.resolve()),
        "image_size": {"width": image.width, "height": image.height},
        "yolo_weights_path": str(yolo_weights_path.resolve()),
        "skill_checkpoint_path": str(runtime.checkpoint_path.resolve()),
        "canonical_manifest_path": str(canonical_manifest_path.resolve()),
        "canonical_image_count": len(canonical_index.records),
        "json_path": str(json_path.resolve()),
        "overlay_path": str(overlay_path.resolve()),
        "detections": serialized_detections,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    overlay_writer(overlay_path, image, serialized_detections)
    return payload


def crop_to_box(image: Any, box_xyxy: tuple[int, int, int, int]) -> Any:
    left, top, right, bottom = box_xyxy
    clipped_left = max(0, left)
    clipped_top = max(0, top)
    clipped_right = min(image.width, right)
    clipped_bottom = min(image.height, bottom)
    clipped_width = max(1, clipped_right - clipped_left)
    clipped_height = max(1, clipped_bottom - clipped_top)

    # Trim from the bottom so the recognition crop better matches the
    # canonical skill-card aspect ratio and excludes trailing UI strips.
    target_height = int(round(clipped_width / CANONICAL_CARD_ASPECT_RATIO))
    if 0 < target_height < clipped_height:
        clipped_bottom = clipped_top + target_height

    clipped = (
        clipped_left,
        clipped_top,
        clipped_right,
        clipped_bottom,
    )
    return image.crop(clipped)


def predict_crop(crop_image: Any, runtime: SkillRuntime, canonical_index: CanonicalIndex, topk: int) -> QueryPrediction:
    import torch

    with torch.no_grad():
        tensor = runtime.transform(crop_image).unsqueeze(0).to(runtime.device)
        outputs = runtime.model(tensor, None)
        empty_probability = float(outputs["empty_logits"].sigmoid().item())
        if empty_probability >= 0.5:
            return QueryPrediction(
                predicted_empty=True,
                predicted_empty_probability=empty_probability,
                predicted_skill_card_id=None,
                predicted_character_id=None,
                top_matches=[],
                debug_stats={"status": "empty"},
            )

        similarities = torch.matmul(outputs["embedding"], canonical_index.embeddings.T).squeeze(0)
        top_matches = build_top_matches(similarities, canonical_index.records, topk)
        debug_stats = build_similarity_debug_stats(similarities, canonical_index.records, top_matches)
        best_match = top_matches[0] if top_matches else None
        return QueryPrediction(
            predicted_empty=False,
            predicted_empty_probability=empty_probability,
            predicted_skill_card_id=best_match["skill_card_id"] if best_match is not None else None,
            predicted_character_id=best_match["character_id"] if best_match is not None else None,
            top_matches=top_matches,
            debug_stats=debug_stats,
        )


def build_top_matches(similarities: Any, records: list[CanonicalImageRecord], topk: int) -> list[dict[str, Any]]:
    import torch

    if not records:
        return []
    k = max(1, min(int(topk), len(records)))
    top_values, top_indices = torch.topk(similarities, k=k)
    matches: list[dict[str, Any]] = []
    for rank, (value, index) in enumerate(zip(top_values.tolist(), top_indices.tolist()), start=1):
        record = records[int(index)]
        matches.append(
            {
                "rank": rank,
                "skill_card_id": record.skill_card_id,
                "character_id": record.character_id,
                "canonical_image_path": str(record.image_path.resolve()),
                "similarity": float(value),
            }
        )
    return matches


def build_similarity_debug_stats(
    similarities: Any,
    records: list[CanonicalImageRecord],
    top_matches: list[dict[str, Any]],
    *,
    top_identity_limit: int = 5,
) -> dict[str, Any]:
    values = [float(value) for value in similarities.detach().cpu().tolist()]
    identity_summaries: dict[str, dict[str, Any]] = {}
    for value, record in zip(values, records):
        identity_summary = identity_summaries.setdefault(
            record.skill_card_id,
            {
                "skill_card_id": record.skill_card_id,
                "character_id": record.character_id,
                "sample_count": 0,
                "best_similarity": -math.inf,
                "similarity_sum": 0.0,
                "render_modes": {},
            },
        )
        identity_summary["sample_count"] += 1
        identity_summary["best_similarity"] = max(identity_summary["best_similarity"], value)
        identity_summary["similarity_sum"] += value

        render_mode = record.render_mode or "unknown"
        mode_summary = identity_summary["render_modes"].setdefault(
            render_mode,
            {
                "sample_count": 0,
                "best_similarity": -math.inf,
                "similarity_sum": 0.0,
            },
        )
        mode_summary["sample_count"] += 1
        mode_summary["best_similarity"] = max(mode_summary["best_similarity"], value)
        mode_summary["similarity_sum"] += value

    top_identities = sorted(
        identity_summaries.values(),
        key=lambda item: (item["best_similarity"], item["similarity_sum"] / max(1, item["sample_count"])),
        reverse=True,
    )[:top_identity_limit]

    top_identity_payload: list[dict[str, Any]] = []
    for rank, summary in enumerate(top_identities, start=1):
        render_modes = {
            name: {
                "sample_count": mode["sample_count"],
                "best_similarity": mode["best_similarity"],
                "mean_similarity": mode["similarity_sum"] / max(1, mode["sample_count"]),
            }
            for name, mode in sorted(summary["render_modes"].items())
        }
        top_identity_payload.append(
            {
                "rank": rank,
                "skill_card_id": summary["skill_card_id"],
                "character_id": summary["character_id"],
                "sample_count": summary["sample_count"],
                "best_similarity": summary["best_similarity"],
                "mean_similarity": summary["similarity_sum"] / max(1, summary["sample_count"]),
                "render_modes": render_modes,
            }
        )

    best_other_similarity = 0.0
    top1_margin = 0.0
    if len(top_identity_payload) >= 2:
        best_other_similarity = float(top_identity_payload[1]["best_similarity"])
        top1_margin = float(top_identity_payload[0]["best_similarity"] - best_other_similarity)

    return {
        "similarity_distribution": summarize_similarity_distribution(values),
        "topk_identity_votes": build_topk_identity_votes(top_matches),
        "top_identity_summaries": top_identity_payload,
        "best_other_identity_similarity": best_other_similarity,
        "top1_margin": top1_margin,
    }


def summarize_similarity_distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    sorted_values = sorted(values)
    return {
        "count": len(values),
        "min": sorted_values[0],
        "max": sorted_values[-1],
        "mean": mean,
        "std": math.sqrt(variance),
        "p95": percentile(sorted_values, 0.95),
        "p99": percentile(sorted_values, 0.99),
    }


def percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = (len(sorted_values) - 1) * fraction
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return sorted_values[lower]
    weight = index - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def build_topk_identity_votes(top_matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    votes: dict[str, dict[str, Any]] = {}
    for match in top_matches:
        identity = str(match["skill_card_id"])
        vote = votes.setdefault(
            identity,
            {
                "skill_card_id": identity,
                "character_id": match.get("character_id"),
                "count": 0,
                "best_similarity": float("-inf"),
            },
        )
        vote["count"] += 1
        vote["best_similarity"] = max(vote["best_similarity"], float(match["similarity"]))
    return sorted(votes.values(), key=lambda item: (item["count"], item["best_similarity"]), reverse=True)


def write_overlay_image(output_path: Path, image: Any, detections: list[dict[str, Any]]) -> None:
    from PIL import ImageDraw

    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for detection in detections:
        box = detection["box_xyxy"]
        draw.rectangle(box, outline=(255, 64, 64), width=3)
        label = build_overlay_label(detection)
        label_x = max(0, int(box[0]))
        label_y = max(0, int(box[1]) - 14)
        draw.text((label_x, label_y), label, fill=(255, 255, 0))
    overlay.save(output_path)


def build_overlay_label(detection: dict[str, Any]) -> str:
    if detection["predicted_empty"]:
        return f"#{detection['index']} {detection['confidence']:.2f} empty"
    best_similarity = detection["top_matches"][0]["similarity"] if detection["top_matches"] else 0.0
    identity = detection["predicted_skill_card_id"] or "unknown"
    return f"#{detection['index']} {detection['confidence']:.2f} {identity} {best_similarity:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
