from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kronos_shared import GalleryArtifact, GalleryPrototype, save_gallery_artifact

from .skill_data import (
    BalancedIdentityModeSampler,
    SkillClassificationDataset,
    build_identity_index,
    collate_samples,
    ensure_empty_examples,
    is_empty_row,
    load_rows,
    prepare_runtime_subsets,
)
from .skill_model import build_skill_model
from .skill_training_config import SkillTrainConfig, skill_config_from_dict


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        import torch
    except ImportError:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(config: SkillTrainConfig) -> tuple[Any, Any, Any, list[str], Path]:
    from torch.utils.data import DataLoader

    manifest_path = Path(config.data.manifest_path)
    dataset_root = manifest_path.parent
    rows = load_rows(manifest_path)
    ensure_empty_examples(rows)
    identities = build_identity_index(rows)
    identity_to_index = {identity: index for index, identity in enumerate(identities)}
    prepared = prepare_runtime_subsets(
        rows,
        train_subset=config.data.train_subset,
        val_subset=config.data.val_subset,
        test_subset=config.data.test_subset,
        gallery_subset=config.data.gallery_subset,
        gallery_count_per_identity=config.data.gallery_count_per_identity,
        seed=config.trainer.seed,
    )
    gallery_rows = prepared["gallery"]
    train_rows = prepared["train_query"]
    val_rows = prepared["val_query"]
    if not gallery_rows or not train_rows or not val_rows:
        raise ValueError("Manifest must yield non-empty gallery, train_query, and val_query subsets.")

    common = {
        "dataset_root": dataset_root,
        "identity_to_index": identity_to_index,
        "image_size": config.data.image_size,
    }
    gallery_dataset = SkillClassificationDataset(rows=gallery_rows, **common)
    train_dataset = SkillClassificationDataset(rows=train_rows, augment=True, **common)
    val_dataset = SkillClassificationDataset(rows=val_rows, **common)

    loader_kwargs = {
        "batch_size": config.data.batch_size,
        "num_workers": config.data.num_workers,
        "pin_memory": config.data.pin_memory,
        "collate_fn": collate_samples,
    }
    if config.data.train_sampler == "balanced_identity_mode":
        train_loader = DataLoader(
            train_dataset,
            shuffle=False,
            sampler=BalancedIdentityModeSampler(train_rows, seed=config.trainer.seed),
            **loader_kwargs,
        )
    else:
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    gallery_loader = DataLoader(gallery_dataset, shuffle=False, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    return train_loader, gallery_loader, val_loader, identities, dataset_root


def build_optimizer(config: SkillTrainConfig, model: Any) -> tuple[Any, Any]:
    import torch

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
    )

    def lr_lambda(step: int) -> float:
        total_steps = max(1, config.trainer.epochs)
        progress = min(1.0, step / total_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_factor = config.optimizer.min_lr / config.optimizer.lr
        return min_factor + (1.0 - min_factor) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler


def move_batch_to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {
        "images": batch["images"].to(device, non_blocking=True),
        "identity_indices": batch["identity_indices"].to(device, non_blocking=True),
        "empty_labels": batch["empty_labels"].to(device, non_blocking=True),
        "non_empty_mask": batch["non_empty_mask"].to(device, non_blocking=True),
        "card_boxes": batch["card_boxes"].to(device, non_blocking=True),
        "rows": batch["rows"],
    }


def compute_losses(config: SkillTrainConfig, outputs: dict[str, Any], batch: dict[str, Any]) -> tuple[Any, dict[str, float]]:
    import torch.nn.functional as F

    non_empty_mask = batch["non_empty_mask"]
    zero = outputs["empty_logits"].new_zeros(())
    empty_loss = F.binary_cross_entropy_with_logits(outputs["empty_logits"], batch["empty_labels"])
    if bool(non_empty_mask.any().item()):
        identity_loss = F.cross_entropy(outputs["identity_logits"][non_empty_mask], batch["identity_indices"][non_empty_mask])
    else:
        identity_loss = zero
    total = (config.losses.identity_weight * identity_loss) + (config.losses.empty_weight * empty_loss)
    return total, {
        "identity_loss": float(identity_loss.detach().cpu()),
        "empty_loss": float(empty_loss.detach().cpu()),
    }


def train_one_epoch(*, model: Any, loader: Any, optimizer: Any, device: Any, config: SkillTrainConfig) -> dict[str, float]:
    import contextlib
    import torch

    model.train()
    use_fp16 = device.type == "cuda" and config.trainer.mixed_precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16) if use_fp16 else None
    autocast_context = (
        lambda: torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_fp16
        else contextlib.nullcontext()
    )

    totals = {"loss": 0.0, "identity_loss": 0.0, "empty_loss": 0.0}
    total_items = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        safe_identity_indices = batch["identity_indices"].masked_fill(~batch["non_empty_mask"], 0)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context():
            outputs = model(batch["images"], safe_identity_indices)
            loss, parts = compute_losses(config, outputs, batch)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = batch["images"].shape[0]
        total_items += batch_size
        totals["loss"] += float(loss.detach().cpu()) * batch_size
        totals["identity_loss"] += parts["identity_loss"] * batch_size
        totals["empty_loss"] += parts["empty_loss"] * batch_size

    return {key: value / total_items for key, value in totals.items()}


def compute_gallery_embeddings(model: Any, loader: Any, identities: list[str], device: Any) -> tuple[Any, dict[str, int]]:
    import torch

    model.eval()
    grouped: dict[str, list[Any]] = {identity: [] for identity in identities}
    counts: dict[str, int] = {identity: 0 for identity in identities}
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(batch["images"], None)
            embeddings = outputs["embedding"].detach().cpu()
            for row, embedding in zip(batch["rows"], embeddings):
                grouped[row.identity_key].append(embedding)
                counts[row.identity_key] += 1

    prototype_vectors = []
    for identity in identities:
        stacked = torch.stack(grouped[identity])
        prototype = torch.nn.functional.normalize(stacked.mean(dim=0), dim=0)
        prototype_vectors.append(prototype)
    return torch.stack(prototype_vectors).to(device), counts


@dataclass(frozen=True)
class CanonicalSkillRecord:
    identity_key: str
    image_path: Path


def load_canonical_records(manifest_path: Path) -> list[CanonicalSkillRecord]:
    dataset_root = manifest_path.parent
    records: list[CanonicalSkillRecord] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            identity = payload.get("skill_card_id") or payload.get("character_id")
            image_path = payload.get("image_path")
            if not identity or not image_path:
                continue
            path = Path(image_path)
            if not path.is_absolute():
                path = dataset_root / path
            records.append(CanonicalSkillRecord(identity_key=str(identity), image_path=path))
    if not records:
        raise ValueError(f"Canonical manifest has no usable rows: {manifest_path}")
    return records


def run_embedding_tensor_batch(batch_tensors: list[Any], model: Any, device: Any) -> Any:
    import torch

    stacked = torch.stack(batch_tensors).to(device)
    with torch.no_grad():
        outputs = model(stacked, None)
    return outputs["embedding"].detach()


def compute_canonical_gallery_embeddings(
    records: list[CanonicalSkillRecord],
    config: SkillTrainConfig,
    model: Any,
    device: Any,
) -> tuple[list[CanonicalSkillRecord], Any, dict[str, int]]:
    from PIL import Image
    import torch

    from .data import build_transforms

    transform = build_transforms(config.data.image_size)
    grouped_counts: dict[str, int] = {}
    embedding_batches: list[Any] = []
    batch_tensors: list[Any] = []
    for record in records:
        image = Image.open(record.image_path).convert("RGB")
        batch_tensors.append(transform(image))
        grouped_counts[record.identity_key] = grouped_counts.get(record.identity_key, 0) + 1
        if len(batch_tensors) >= config.data.batch_size:
            embedding_batches.append(run_embedding_tensor_batch(batch_tensors, model, device))
            batch_tensors = []
    if batch_tensors:
        embedding_batches.append(run_embedding_tensor_batch(batch_tensors, model, device))
    return records, torch.cat(embedding_batches, dim=0), grouped_counts


def top_unique_identities(similarities: Any, records: list[CanonicalSkillRecord], limit: int = 3) -> list[str]:
    import torch

    ordered_indices = torch.argsort(similarities, descending=True).tolist()
    identities: list[str] = []
    seen: set[str] = set()
    for index in ordered_indices:
        identity = records[int(index)].identity_key
        if identity in seen:
            continue
        seen.add(identity)
        identities.append(identity)
        if len(identities) >= limit:
            break
    return identities


def canonical_identity_scores(similarities: Any, records: list[CanonicalSkillRecord]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for value, record in zip(similarities.detach().cpu().tolist(), records):
        value = float(value)
        current = scores.get(record.identity_key)
        if current is None or value > current:
            scores[record.identity_key] = value
    return scores


def prototype_identity_scores(similarities: Any, identities: list[str]) -> dict[str, float]:
    return {identity: float(value) for identity, value in zip(identities, similarities.detach().cpu().tolist())}


def row_tags(row: Any) -> list[str]:
    if is_empty_row(row):
        return []
    tags = ["all", "obstructed" if row.obstructions else "clean"]
    quality = row.quality_policy
    if "blur_radius" in quality:
        tags.append("blurred")
    if int(quality.get("jpeg_quality", 100)) <= 60:
        tags.append("jpeg_low")
    if "jpeg_quality" in quality and int(quality.get("jpeg_quality", 100)) > 60:
        tags.append("jpeg_present")
    return tags


def evaluate_retrieval(*, model: Any, gallery_loader: Any, query_loader: Any, identities: list[str], device: Any) -> dict[str, Any]:
    import torch

    prototypes, prototype_counts = compute_gallery_embeddings(model, gallery_loader, identities, device)
    index_to_identity = {index: identity for index, identity in enumerate(identities)}

    model.eval()
    counters: dict[str, dict[str, float]] = {}
    empty_total = 0.0
    empty_correct = 0.0
    empty_true = 0.0
    empty_predicted = 0.0
    empty_true_positive = 0.0

    def ensure(tag: str) -> dict[str, float]:
        if tag not in counters:
            counters[tag] = {"count": 0.0, "top1": 0.0, "top3": 0.0}
        return counters[tag]

    with torch.no_grad():
        for batch in query_loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(batch["images"], None)
            similarities = outputs["embedding"] @ prototypes.T
            topk_indices = similarities.topk(k=min(3, len(identities)), dim=1).indices
            empty_predictions = (outputs["empty_logits"].sigmoid() >= 0.5).long()

            for item_index, row in enumerate(batch["rows"]):
                row_is_empty = is_empty_row(row)
                predicted_empty = bool(int(empty_predictions[item_index].detach().cpu()))
                empty_total += 1.0
                empty_true += 1.0 if row_is_empty else 0.0
                empty_predicted += 1.0 if predicted_empty else 0.0
                empty_true_positive += 1.0 if row_is_empty and predicted_empty else 0.0
                empty_correct += 1.0 if row_is_empty == predicted_empty else 0.0
                if row_is_empty:
                    continue

                top_identities = [] if predicted_empty else [index_to_identity[int(index)] for index in topk_indices[item_index]]
                for tag in row_tags(row):
                    bucket = ensure(tag)
                    bucket["count"] += 1.0
                    bucket["top1"] += 1.0 if top_identities and top_identities[0] == row.identity_key else 0.0
                    bucket["top3"] += 1.0 if row.identity_key in top_identities else 0.0

    metrics = {
        tag: {
            "count": int(values["count"]),
            "retrieval_top1": values["top1"] / values["count"],
            "retrieval_top3": values["top3"] / values["count"],
        }
        for tag, values in counters.items()
        if values["count"] > 0
    }
    metrics["empty"] = {
        "count": int(empty_total),
        "accuracy": empty_correct / empty_total if empty_total > 0 else 0.0,
        "precision": (empty_true_positive / empty_predicted) if empty_predicted > 0 else 0.0,
        "recall": (empty_true_positive / empty_true) if empty_true > 0 else 0.0,
    }
    metrics["gallery_sample_count"] = prototype_counts
    return metrics


def evaluate_retrieval_against_canonical(
    *,
    model: Any,
    config: SkillTrainConfig,
    query_loader: Any,
    canonical_manifest_path: Path,
    device: Any,
) -> dict[str, Any]:
    import torch

    records = load_canonical_records(canonical_manifest_path)
    records, gallery_embeddings, gallery_counts = compute_canonical_gallery_embeddings(records, config, model, device)

    model.eval()
    counters: dict[str, dict[str, float]] = {}
    empty_total = 0.0
    empty_correct = 0.0
    empty_true = 0.0
    empty_predicted = 0.0
    empty_true_positive = 0.0

    def ensure(tag: str) -> dict[str, float]:
        if tag not in counters:
            counters[tag] = {"count": 0.0, "top1": 0.0, "top3": 0.0}
        return counters[tag]

    with torch.no_grad():
        for batch in query_loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(batch["images"], None)
            similarities = outputs["embedding"] @ gallery_embeddings.T
            empty_predictions = (outputs["empty_logits"].sigmoid() >= 0.5).long()

            for item_index, row in enumerate(batch["rows"]):
                row_is_empty = is_empty_row(row)
                predicted_empty = bool(int(empty_predictions[item_index].detach().cpu()))
                empty_total += 1.0
                empty_true += 1.0 if row_is_empty else 0.0
                empty_predicted += 1.0 if predicted_empty else 0.0
                empty_true_positive += 1.0 if row_is_empty and predicted_empty else 0.0
                empty_correct += 1.0 if row_is_empty == predicted_empty else 0.0
                if row_is_empty:
                    continue

                top_identities = [] if predicted_empty else top_unique_identities(similarities[item_index], records, limit=3)
                for tag in row_tags(row):
                    bucket = ensure(tag)
                    bucket["count"] += 1.0
                    bucket["top1"] += 1.0 if top_identities and top_identities[0] == row.identity_key else 0.0
                    bucket["top3"] += 1.0 if row.identity_key in top_identities else 0.0

    metrics = {
        tag: {
            "count": int(values["count"]),
            "retrieval_top1": values["top1"] / values["count"],
            "retrieval_top3": values["top3"] / values["count"],
        }
        for tag, values in counters.items()
        if values["count"] > 0
    }
    metrics["empty"] = {
        "count": int(empty_total),
        "accuracy": empty_correct / empty_total if empty_total > 0 else 0.0,
        "precision": (empty_true_positive / empty_predicted) if empty_predicted > 0 else 0.0,
        "recall": (empty_true_positive / empty_true) if empty_true > 0 else 0.0,
    }
    metrics["gallery_type"] = "canonical_image"
    metrics["gallery_manifest_path"] = str(canonical_manifest_path)
    metrics["gallery_image_count"] = len(records)
    metrics["gallery_sample_count"] = gallery_counts
    return metrics


def summarize_confusion_analysis(
    *,
    gallery_type: str,
    subset: str,
    total_non_empty: int,
    top1_correct: int,
    nearest_competitor_counts: dict[str, int],
    pair_stats: dict[tuple[str, str], dict[str, float]],
    per_identity_stats: dict[str, dict[str, Any]],
    topn: int,
) -> dict[str, Any]:
    overall_competitors = [
        {"skill_card_id": identity, "count": count}
        for identity, count in sorted(nearest_competitor_counts.items(), key=lambda item: (-item[1], item[0]))[:topn]
    ]
    confusion_pairs = []
    for (true_identity, predicted_identity), stats in sorted(
        pair_stats.items(),
        key=lambda item: (-int(item[1]["count"]), item[0][0], item[0][1]),
    )[:topn]:
        confusion_pairs.append(
            {
                "true_skill_card_id": true_identity,
                "predicted_skill_card_id": predicted_identity,
                "count": int(stats["count"]),
                "mean_predicted_score": stats["predicted_score_sum"] / max(1.0, stats["count"]),
                "mean_true_score": stats["true_score_sum"] / max(1.0, stats["count"]),
                "mean_margin": stats["margin_sum"] / max(1.0, stats["count"]),
            }
        )

    identity_payload = []
    for identity, stats in sorted(per_identity_stats.items()):
        sample_count = int(stats["sample_count"])
        competitor_payload = []
        for competitor, competitor_stats in sorted(
            stats["competitors"].items(),
            key=lambda item: (-int(item[1]["count"]), item[0]),
        )[:topn]:
            competitor_payload.append(
                {
                    "skill_card_id": competitor,
                    "count": int(competitor_stats["count"]),
                    "rate": competitor_stats["count"] / max(1.0, stats["sample_count"]),
                    "mean_competitor_score": competitor_stats["competitor_score_sum"] / max(1.0, competitor_stats["count"]),
                    "mean_true_score": competitor_stats["true_score_sum"] / max(1.0, competitor_stats["count"]),
                    "mean_margin": competitor_stats["margin_sum"] / max(1.0, competitor_stats["count"]),
                }
            )
        identity_payload.append(
            {
                "skill_card_id": identity,
                "sample_count": sample_count,
                "top1_accuracy": stats["correct_top1"] / max(1.0, stats["sample_count"]),
                "mean_true_rank": stats["true_rank_sum"] / max(1.0, stats["sample_count"]),
                "closest_competitors": competitor_payload,
            }
        )

    return {
        "subset": subset,
        "gallery_type": gallery_type,
        "sample_count": total_non_empty,
        "top1_accuracy": top1_correct / max(1, total_non_empty),
        "overall_closest_competitors": overall_competitors,
        "confusion_pairs": confusion_pairs,
        "identities": identity_payload,
    }


def analyze_retrieval_confusions(
    *,
    model: Any,
    gallery_loader: Any,
    query_loader: Any,
    identities: list[str],
    device: Any,
    subset: str,
    topn: int,
) -> dict[str, Any]:
    prototypes, _ = compute_gallery_embeddings(model, gallery_loader, identities, device)
    model.eval()

    total_non_empty = 0
    top1_correct = 0
    nearest_competitor_counts: dict[str, int] = {}
    pair_stats: dict[tuple[str, str], dict[str, float]] = {}
    per_identity_stats: dict[str, dict[str, Any]] = {}

    import torch

    with torch.no_grad():
        for batch in query_loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(batch["images"], None)
            similarity_batch = outputs["embedding"] @ prototypes.T
            empty_predictions = (outputs["empty_logits"].sigmoid() >= 0.5).long()

            for item_index, row in enumerate(batch["rows"]):
                if is_empty_row(row) or bool(int(empty_predictions[item_index].detach().cpu())):
                    continue
                scores = prototype_identity_scores(similarity_batch[item_index], identities)
                update_confusion_stats(
                    scores=scores,
                    true_identity=row.identity_key,
                    nearest_competitor_counts=nearest_competitor_counts,
                    pair_stats=pair_stats,
                    per_identity_stats=per_identity_stats,
                )
                total_non_empty += 1
                ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
                if ordered and ordered[0][0] == row.identity_key:
                    top1_correct += 1

    return summarize_confusion_analysis(
        gallery_type="prototype",
        subset=subset,
        total_non_empty=total_non_empty,
        top1_correct=top1_correct,
        nearest_competitor_counts=nearest_competitor_counts,
        pair_stats=pair_stats,
        per_identity_stats=per_identity_stats,
        topn=topn,
    )


def analyze_retrieval_confusions_against_canonical(
    *,
    model: Any,
    config: SkillTrainConfig,
    query_loader: Any,
    canonical_manifest_path: Path,
    device: Any,
    subset: str,
    topn: int,
) -> dict[str, Any]:
    records = load_canonical_records(canonical_manifest_path)
    records, gallery_embeddings, _ = compute_canonical_gallery_embeddings(records, config, model, device)
    model.eval()

    total_non_empty = 0
    top1_correct = 0
    nearest_competitor_counts: dict[str, int] = {}
    pair_stats: dict[tuple[str, str], dict[str, float]] = {}
    per_identity_stats: dict[str, dict[str, Any]] = {}

    import torch

    with torch.no_grad():
        for batch in query_loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(batch["images"], None)
            similarity_batch = outputs["embedding"] @ gallery_embeddings.T
            empty_predictions = (outputs["empty_logits"].sigmoid() >= 0.5).long()

            for item_index, row in enumerate(batch["rows"]):
                if is_empty_row(row) or bool(int(empty_predictions[item_index].detach().cpu())):
                    continue
                scores = canonical_identity_scores(similarity_batch[item_index], records)
                update_confusion_stats(
                    scores=scores,
                    true_identity=row.identity_key,
                    nearest_competitor_counts=nearest_competitor_counts,
                    pair_stats=pair_stats,
                    per_identity_stats=per_identity_stats,
                )
                total_non_empty += 1
                ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
                if ordered and ordered[0][0] == row.identity_key:
                    top1_correct += 1

    summary = summarize_confusion_analysis(
        gallery_type="canonical_image",
        subset=subset,
        total_non_empty=total_non_empty,
        top1_correct=top1_correct,
        nearest_competitor_counts=nearest_competitor_counts,
        pair_stats=pair_stats,
        per_identity_stats=per_identity_stats,
        topn=topn,
    )
    summary["gallery_manifest_path"] = str(canonical_manifest_path)
    return summary


def update_confusion_stats(
    *,
    scores: dict[str, float],
    true_identity: str,
    nearest_competitor_counts: dict[str, int],
    pair_stats: dict[tuple[str, str], dict[str, float]],
    per_identity_stats: dict[str, dict[str, Any]],
) -> None:
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    predicted_identity, predicted_score = ordered[0]
    true_score = float(scores[true_identity])
    true_rank = next(index for index, (identity, _) in enumerate(ordered, start=1) if identity == true_identity)
    competitor_identity, competitor_score = next(
        (identity, score) for identity, score in ordered if identity != true_identity
    )
    margin = float(competitor_score - true_score)

    nearest_competitor_counts[competitor_identity] = nearest_competitor_counts.get(competitor_identity, 0) + 1
    identity_stats = per_identity_stats.setdefault(
        true_identity,
        {"sample_count": 0.0, "correct_top1": 0.0, "true_rank_sum": 0.0, "competitors": {}},
    )
    identity_stats["sample_count"] += 1.0
    identity_stats["correct_top1"] += 1.0 if predicted_identity == true_identity else 0.0
    identity_stats["true_rank_sum"] += float(true_rank)
    competitor_stats = identity_stats["competitors"].setdefault(
        competitor_identity,
        {"count": 0.0, "competitor_score_sum": 0.0, "true_score_sum": 0.0, "margin_sum": 0.0},
    )
    competitor_stats["count"] += 1.0
    competitor_stats["competitor_score_sum"] += competitor_score
    competitor_stats["true_score_sum"] += true_score
    competitor_stats["margin_sum"] += margin

    if predicted_identity != true_identity:
        pair = pair_stats.setdefault(
            (true_identity, predicted_identity),
            {"count": 0.0, "predicted_score_sum": 0.0, "true_score_sum": 0.0, "margin_sum": 0.0},
        )
        pair["count"] += 1.0
        pair["predicted_score_sum"] += predicted_score
        pair["true_score_sum"] += true_score
        pair["margin_sum"] += float(predicted_score - true_score)


def save_checkpoint(
    *,
    output_dir: Path,
    filename: str,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    epoch: int,
    best_top1: float,
    config: SkillTrainConfig,
    identities: list[str],
) -> None:
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_top1": best_top1,
            "config": config.to_dict(),
            "identities": identities,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        },
        output_dir / filename,
    )


def load_checkpoint(checkpoint_path: Path, device: Any) -> tuple[SkillTrainConfig, list[str], dict[str, Any]]:
    import torch

    payload = torch.load(checkpoint_path, map_location=device)
    config = skill_config_from_dict(payload["config"])
    return config, list(payload["identities"]), payload


def train_skill_from_config(config: SkillTrainConfig, device_name: str, dry_run: bool = False) -> None:
    import torch

    set_seed(config.trainer.seed)
    train_loader, gallery_loader, val_loader, identities, _ = build_loaders(config)
    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")
    model = build_skill_model(config, len(identities)).to(device)
    optimizer, scheduler = build_optimizer(config, model)
    start_epoch = 0
    if config.trainer.resume_from:
        payload = torch.load(config.trainer.resume_from, map_location=device)
        model.load_state_dict(payload["model_state"], strict=True)
        if "optimizer_state" in payload:
            optimizer.load_state_dict(payload["optimizer_state"])
        if "scheduler_state" in payload:
            scheduler.load_state_dict(payload["scheduler_state"])
        start_epoch = int(payload.get("epoch", -1)) + 1

    best_top1 = -1.0
    epochs_without_improvement = 0
    patience = config.trainer.patience
    output_dir = Path(config.trainer.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = start_epoch + 1 if dry_run else config.trainer.epochs
    for epoch in range(start_epoch, epochs):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            config=config,
        )
        val_metrics = evaluate_retrieval(
            model=model,
            gallery_loader=gallery_loader,
            query_loader=val_loader,
            identities=identities,
            device=device,
        )
        val_top1 = float(val_metrics["all"]["retrieval_top1"])
        is_best = val_top1 > best_top1
        if is_best:
            best_top1 = val_top1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        scheduler.step()
        save_checkpoint(
            output_dir=output_dir,
            filename="last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_top1=best_top1,
            config=config,
            identities=identities,
        )
        if is_best:
            save_checkpoint(
                output_dir=output_dir,
                filename="best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_top1=best_top1,
                config=config,
                identities=identities,
            )
        with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"epoch": epoch, "train": train_metrics, "val": val_metrics}) + "\n")
        print(json.dumps({"epoch": epoch, "train": train_metrics, "val": val_metrics, "best_top1": best_top1}, indent=2))
        if patience > 0 and epochs_without_improvement >= patience:
            print(f"Early stopping: no improvement for {patience} epochs.")
            break


def load_model_from_checkpoint(checkpoint_path: Path, device_name: str) -> tuple[Any, SkillTrainConfig, list[str], Any]:
    import torch

    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")
    config, identities, payload = load_checkpoint(checkpoint_path, device)
    model = build_skill_model(config, len(identities)).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model, config, identities, device


def export_skill_gallery(checkpoint_path: Path, output_path: Path, device_name: str) -> None:
    from torch.utils.data import DataLoader

    model, config, identities, device = load_model_from_checkpoint(checkpoint_path, device_name)
    manifest_path = Path(config.data.manifest_path)
    dataset_root = manifest_path.parent
    rows = load_rows(manifest_path)
    ensure_empty_examples(rows)
    identity_to_index = {identity: index for index, identity in enumerate(identities)}
    prepared = prepare_runtime_subsets(
        rows,
        train_subset=config.data.train_subset,
        val_subset=config.data.val_subset,
        test_subset=config.data.test_subset,
        gallery_subset=config.data.gallery_subset,
        gallery_count_per_identity=config.data.gallery_count_per_identity,
        seed=config.trainer.seed,
    )
    gallery_dataset = SkillClassificationDataset(
        rows=prepared["gallery"],
        dataset_root=dataset_root,
        identity_to_index=identity_to_index,
        image_size=config.data.image_size,
    )
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        collate_fn=collate_samples,
    )
    prototypes, counts = compute_gallery_embeddings(model, gallery_loader, identities, device)
    artifact = GalleryArtifact(
        model_name=config.model.model_name,
        embedding_dim=config.model.embedding_dim,
        prototype_strategy="mean_l2",
        prototypes=[
            GalleryPrototype(
                character_id=identity,
                embedding=prototypes[index].detach().cpu().tolist(),
                sample_count=counts[identity],
            )
            for index, identity in enumerate(identities)
        ],
    )
    save_gallery_artifact(output_path, artifact)
    print(json.dumps({"output": str(output_path), "identity_count": len(identities)}, indent=2))


def evaluate_skill_checkpoint(
    checkpoint_path: Path,
    subset: str,
    device_name: str,
    output_path: Path | None = None,
    canonical_manifest_path: Path | None = None,
) -> None:
    from torch.utils.data import DataLoader

    model, config, identities, device = load_model_from_checkpoint(checkpoint_path, device_name)
    manifest_path = Path(config.data.manifest_path)
    dataset_root = manifest_path.parent
    rows = load_rows(manifest_path)
    ensure_empty_examples(rows)
    identity_to_index = {identity: index for index, identity in enumerate(identities)}
    prepared = prepare_runtime_subsets(
        rows,
        train_subset=config.data.train_subset,
        val_subset=config.data.val_subset,
        test_subset=config.data.test_subset,
        gallery_subset=config.data.gallery_subset,
        gallery_count_per_identity=config.data.gallery_count_per_identity,
        seed=config.trainer.seed,
    )
    query_dataset = SkillClassificationDataset(
        rows=prepared[subset],
        dataset_root=dataset_root,
        identity_to_index=identity_to_index,
        image_size=config.data.image_size,
    )
    loader_kwargs = {
        "batch_size": config.data.batch_size,
        "shuffle": False,
        "num_workers": config.data.num_workers,
        "pin_memory": config.data.pin_memory,
        "collate_fn": collate_samples,
    }
    query_loader = DataLoader(query_dataset, **loader_kwargs)
    if canonical_manifest_path is not None:
        metrics = evaluate_retrieval_against_canonical(
            model=model,
            config=config,
            query_loader=query_loader,
            canonical_manifest_path=canonical_manifest_path,
            device=device,
        )
    else:
        gallery_dataset = SkillClassificationDataset(
            rows=prepared["gallery"],
            dataset_root=dataset_root,
            identity_to_index=identity_to_index,
            image_size=config.data.image_size,
        )
        gallery_loader = DataLoader(gallery_dataset, **loader_kwargs)
        metrics = evaluate_retrieval(
            model=model,
            gallery_loader=gallery_loader,
            query_loader=query_loader,
            identities=identities,
            device=device,
        )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def analyze_skill_confusions(
    checkpoint_path: Path,
    subset: str,
    device_name: str,
    output_path: Path | None = None,
    canonical_manifest_path: Path | None = None,
    topn: int = 10,
) -> None:
    from torch.utils.data import DataLoader

    model, config, identities, device = load_model_from_checkpoint(checkpoint_path, device_name)
    manifest_path = Path(config.data.manifest_path)
    dataset_root = manifest_path.parent
    rows = load_rows(manifest_path)
    ensure_empty_examples(rows)
    identity_to_index = {identity: index for index, identity in enumerate(identities)}
    prepared = prepare_runtime_subsets(
        rows,
        train_subset=config.data.train_subset,
        val_subset=config.data.val_subset,
        test_subset=config.data.test_subset,
        gallery_subset=config.data.gallery_subset,
        gallery_count_per_identity=config.data.gallery_count_per_identity,
        seed=config.trainer.seed,
    )
    query_dataset = SkillClassificationDataset(
        rows=prepared[subset],
        dataset_root=dataset_root,
        identity_to_index=identity_to_index,
        image_size=config.data.image_size,
    )
    loader_kwargs = {
        "batch_size": config.data.batch_size,
        "shuffle": False,
        "num_workers": config.data.num_workers,
        "pin_memory": config.data.pin_memory,
        "collate_fn": collate_samples,
    }
    query_loader = DataLoader(query_dataset, **loader_kwargs)
    if canonical_manifest_path is not None:
        summary = analyze_retrieval_confusions_against_canonical(
            model=model,
            config=config,
            query_loader=query_loader,
            canonical_manifest_path=canonical_manifest_path,
            device=device,
            subset=subset,
            topn=topn,
        )
    else:
        gallery_dataset = SkillClassificationDataset(
            rows=prepared["gallery"],
            dataset_root=dataset_root,
            identity_to_index=identity_to_index,
            image_size=config.data.image_size,
        )
        gallery_loader = DataLoader(gallery_dataset, **loader_kwargs)
        summary = analyze_retrieval_confusions(
            model=model,
            gallery_loader=gallery_loader,
            query_loader=query_loader,
            identities=identities,
            device=device,
            subset=subset,
            topn=topn,
        )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def predict_skill_image(checkpoint_path: Path, image_path: Path, device_name: str, topk: int = 5) -> None:
    from PIL import Image
    import torch
    from torch.utils.data import DataLoader

    model, config, identities, device = load_model_from_checkpoint(checkpoint_path, device_name)
    manifest_path = Path(config.data.manifest_path)
    dataset_root = manifest_path.parent
    rows = load_rows(manifest_path)
    ensure_empty_examples(rows)
    identity_to_index = {identity: index for index, identity in enumerate(identities)}
    prepared = prepare_runtime_subsets(
        rows,
        train_subset=config.data.train_subset,
        val_subset=config.data.val_subset,
        test_subset=config.data.test_subset,
        gallery_subset=config.data.gallery_subset,
        gallery_count_per_identity=config.data.gallery_count_per_identity,
        seed=config.trainer.seed,
    )

    from .data import build_transforms

    image = Image.open(image_path).convert("RGB")
    transform = build_transforms(config.data.image_size)
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor, None)
        empty_probability = float(outputs["empty_logits"].sigmoid().item())
        predicted_empty = empty_probability >= 0.5

    if predicted_empty:
        print(
            json.dumps(
                {
                    "image": str(image_path),
                    "predicted_character_id": "empty",
                    "predicted_empty": True,
                    "predicted_empty_probability": empty_probability,
                    "top_matches": [],
                },
                indent=2,
            )
        )
        return

    gallery_dataset = SkillClassificationDataset(
        rows=prepared["gallery"],
        dataset_root=dataset_root,
        identity_to_index=identity_to_index,
        image_size=config.data.image_size,
    )
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        collate_fn=collate_samples,
    )
    prototypes, _ = compute_gallery_embeddings(model, gallery_loader, identities, device)
    with torch.no_grad():
        similarities = (outputs["embedding"] @ prototypes.T).squeeze(0)
        k = max(1, min(int(topk), len(identities)))
        top_values, top_indices = similarities.topk(k=k)
    print(
        json.dumps(
            {
                "image": str(image_path),
                "predicted_character_id": identities[int(top_indices[0].item())],
                "predicted_empty": False,
                "predicted_empty_probability": empty_probability,
                "top_matches": [
                    {
                        "character_id": identities[int(index.item())],
                        "similarity": float(value.item()),
                    }
                    for value, index in zip(top_values, top_indices)
                ],
            },
            indent=2,
        )
    )
