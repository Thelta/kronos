from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

from kronos_shared import GalleryArtifact, GalleryPrototype, SyntheticClassificationRow, save_gallery_artifact

from .data import (
    SyntheticCardDataset,
    build_identity_index,
    collate_samples,
    load_rows,
    prepare_runtime_subsets,
)
from .model import build_model
from .training_config import TrainConfig, config_from_dict


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


def build_loaders(config: TrainConfig) -> tuple[Any, Any, Any, list[str], Path]:
    from torch.utils.data import DataLoader

    manifest_path = Path(config.data.manifest_path)
    dataset_root = manifest_path.parent
    rows = load_rows(manifest_path)
    identities = build_identity_index(rows)
    character_to_index = {character_id: index for index, character_id in enumerate(identities)}
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
        "character_to_index": character_to_index,
        "image_size": config.data.image_size,
    }
    gallery_dataset = SyntheticCardDataset(rows=gallery_rows, **common)
    train_dataset = SyntheticCardDataset(
        rows=train_rows,
        tiny_card_degrade_prob=config.data.tiny_card_degrade_prob,
        tiny_card_sizes=config.data.tiny_card_sizes,
        tiny_card_size_jitter=config.data.tiny_card_size_jitter,
        **common,
    )
    val_dataset = SyntheticCardDataset(rows=val_rows, **common)

    loader_kwargs = {
        "batch_size": config.data.batch_size,
        "num_workers": config.data.num_workers,
        "pin_memory": config.data.pin_memory,
        "collate_fn": collate_samples,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    gallery_loader = DataLoader(gallery_dataset, shuffle=False, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    return train_loader, gallery_loader, val_loader, identities, dataset_root


def build_query_loader(config: TrainConfig, subset: str) -> tuple[Any, Any, list[str]]:
    from torch.utils.data import DataLoader

    manifest_path = Path(config.data.manifest_path)
    dataset_root = manifest_path.parent
    rows = load_rows(manifest_path)
    identities = build_identity_index(rows)
    character_to_index = {character_id: index for index, character_id in enumerate(identities)}
    prepared = prepare_runtime_subsets(
        rows,
        train_subset=config.data.train_subset,
        val_subset=config.data.val_subset,
        test_subset=config.data.test_subset,
        gallery_subset=config.data.gallery_subset,
        gallery_count_per_identity=config.data.gallery_count_per_identity,
        seed=config.trainer.seed,
    )
    query_rows = prepared[subset]
    if not query_rows:
        raise ValueError(f"Subset not found or empty: {subset}")
    query_dataset = SyntheticCardDataset(
        rows=query_rows,
        dataset_root=dataset_root,
        character_to_index=character_to_index,
        image_size=config.data.image_size,
    )
    query_loader = DataLoader(
        query_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        collate_fn=collate_samples,
    )
    return query_loader, dataset_root, identities


def build_optimizer(config: TrainConfig, model: Any) -> tuple[Any, Any]:
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
    moved = {
        "images": batch["images"].to(device, non_blocking=True),
        "identity_indices": batch["identity_indices"].to(device, non_blocking=True),
        "star_indices": batch["star_indices"].to(device, non_blocking=True),
        "star_color_indices": batch["star_color_indices"].to(device, non_blocking=True),
        "star_state_indices": batch["star_state_indices"].to(device, non_blocking=True),
        "star_slot_indices": batch["star_slot_indices"].to(device, non_blocking=True),
        "assist": batch["assist"].to(device, non_blocking=True),
        "card_boxes": batch["card_boxes"].to(device, non_blocking=True),
        "star_boxes": batch["star_boxes"].to(device, non_blocking=True),
        "rows": batch["rows"],
    }
    return moved


def compute_losses(config: TrainConfig, outputs: dict[str, Any], batch: dict[str, Any]) -> tuple[Any, dict[str, float]]:
    import torch.nn.functional as F

    identity_loss = F.cross_entropy(outputs["identity_logits"], batch["identity_indices"])
    star_loss = F.cross_entropy(outputs["star_state_logits"], batch["star_state_indices"])
    assist_loss = F.binary_cross_entropy_with_logits(outputs["assist_logits"], batch["assist"])
    total = (
        config.losses.identity_weight * identity_loss
        + (config.losses.star_weight + config.losses.star_color_weight) * star_loss
        + config.losses.assist_weight * assist_loss
    )
    return total, {
        "identity_loss": float(identity_loss.detach().cpu()),
        "star_loss": float(star_loss.detach().cpu()),
        "assist_loss": float(assist_loss.detach().cpu()),
    }


def train_one_epoch(
    *,
    model: Any,
    loader: Any,
    optimizer: Any,
    device: Any,
    config: TrainConfig,
) -> dict[str, float]:
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

    totals = {"loss": 0.0, "identity_loss": 0.0, "star_loss": 0.0, "assist_loss": 0.0}
    total_items = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context():
            outputs = model(batch["images"], batch["identity_indices"], batch["card_boxes"], batch["star_boxes"])
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
        for key in ("identity_loss", "star_loss", "assist_loss"):
            totals[key] += parts[key] * batch_size

    return {key: value / total_items for key, value in totals.items()}


def compute_gallery_embeddings(model: Any, loader: Any, identities: list[str], device: Any) -> tuple[Any, dict[str, int]]:
    import torch

    model.eval()
    grouped: dict[str, list[Any]] = {character_id: [] for character_id in identities}
    counts: dict[str, int] = {character_id: 0 for character_id in identities}
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(batch["images"], None, batch["card_boxes"], batch["star_boxes"])
            embeddings = outputs["embedding"].detach().cpu()
            for row, embedding in zip(batch["rows"], embeddings):
                grouped[row.character_id].append(embedding)
                counts[row.character_id] += 1

    prototype_vectors = []
    for character_id in identities:
        stacked = torch.stack(grouped[character_id])
        prototype = torch.nn.functional.normalize(stacked.mean(dim=0), dim=0)
        prototype_vectors.append(prototype)
    return torch.stack(prototype_vectors).to(device), counts


def row_tags(row: SyntheticClassificationRow) -> list[str]:
    tags = ["all", "obstructed" if row.obstructions else "clean"]
    quality = row.quality_policy
    if "blur_radius" in quality:
        tags.append("blurred")
    if int(quality.get("jpeg_quality", 100)) <= 60:
        tags.append("jpeg_low")
    if "jpeg_quality" in quality and int(quality.get("jpeg_quality", 100)) > 60:
        tags.append("jpeg_present")
    return tags


def evaluate_retrieval(
    *,
    model: Any,
    gallery_loader: Any,
    query_loader: Any,
    identities: list[str],
    device: Any,
) -> dict[str, Any]:
    import torch

    prototypes, prototype_counts = compute_gallery_embeddings(model, gallery_loader, identities, device)
    index_to_character = {index: character_id for index, character_id in enumerate(identities)}

    model.eval()
    counters: dict[str, dict[str, float]] = {}

    def ensure(tag: str) -> dict[str, float]:
        if tag not in counters:
            counters[tag] = {
                "count": 0.0,
                "top1": 0.0,
                "top3": 0.0,
                "star": 0.0,
                "star_color": 0.0,
                "assist": 0.0,
            }
        return counters[tag]

    with torch.no_grad():
        for batch in query_loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(batch["images"], None, batch["card_boxes"], batch["star_boxes"])
            embeddings = outputs["embedding"]
            similarities = embeddings @ prototypes.T
            topk_indices = similarities.topk(k=min(3, len(identities)), dim=1).indices
            star_predictions, star_color_predictions = derive_star_predictions(outputs["star_state_logits"])
            assist_predictions = (outputs["assist_logits"].sigmoid() >= 0.5).long()

            for item_index, row in enumerate(batch["rows"]):
                target_character = row.character_id
                top_characters = [index_to_character[int(index)] for index in topk_indices[item_index]]
                star_correct = int(int(star_predictions[item_index].detach().cpu()) == int(row.star_value))
                star_color_correct = int(
                    int(star_color_predictions[item_index].detach().cpu())
                    == (1 if row.star_color.strip().lower() == "blue" else 0)
                )
                assist_correct = int(int(assist_predictions[item_index].detach().cpu()) == int(bool(row.assist)))
                for tag in row_tags(row):
                    bucket = ensure(tag)
                    bucket["count"] += 1.0
                    bucket["top1"] += 1.0 if top_characters[0] == target_character else 0.0
                    bucket["top3"] += 1.0 if target_character in top_characters else 0.0
                    bucket["star"] += float(star_correct)
                    bucket["star_color"] += float(star_color_correct)
                    bucket["assist"] += float(assist_correct)

    metrics = {
        tag: {
            "count": int(values["count"]),
            "retrieval_top1": values["top1"] / values["count"],
            "retrieval_top3": values["top3"] / values["count"],
            "star_accuracy": values["star"] / values["count"],
            "star_color_accuracy": values["star_color"] / values["count"],
            "assist_accuracy": values["assist"] / values["count"],
        }
        for tag, values in counters.items()
        if values["count"] > 0
    }
    metrics["gallery_sample_count"] = prototype_counts
    return metrics


def derive_star_predictions(star_state_logits: Any) -> tuple[Any, Any]:
    state_predictions = star_state_logits.argmax(dim=1)
    star_count = (state_predictions % 5) + 1
    star_color = state_predictions // 5
    return star_count, star_color


def star_state_label(state_index: int) -> str:
    color = "blue" if state_index >= 5 else "yellow"
    value = (state_index % 5) + 1
    return f"{color}_{value}"


def save_checkpoint(
    *,
    output_dir: Path,
    filename: str,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    epoch: int,
    best_top1: float,
    config: TrainConfig,
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


def load_checkpoint(checkpoint_path: Path, device: Any) -> tuple[TrainConfig, list[str], dict[str, Any]]:
    import torch

    payload = torch.load(checkpoint_path, map_location=device)
    config = config_from_dict(payload["config"])
    return config, list(payload["identities"]), payload


def train_from_config(config: TrainConfig, device_name: str, dry_run: bool = False) -> None:
    import torch

    set_seed(config.trainer.seed)
    train_loader, gallery_loader, val_loader, identities, _ = build_loaders(config)
    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")
    model = build_model(config, len(identities)).to(device)
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
        is_best = val_top1 >= best_top1
        best_top1 = max(best_top1, val_top1)
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


def load_model_from_checkpoint(checkpoint_path: Path, device_name: str) -> tuple[Any, TrainConfig, list[str], Any]:
    import torch

    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")
    config, identities, payload = load_checkpoint(checkpoint_path, device)
    model = build_model(config, len(identities)).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model, config, identities, device


def export_gallery(checkpoint_path: Path, output_path: Path, device_name: str) -> None:
    from torch.utils.data import DataLoader

    model, config, identities, device = load_model_from_checkpoint(checkpoint_path, device_name)
    manifest_path = Path(config.data.manifest_path)
    dataset_root = manifest_path.parent
    rows = load_rows(manifest_path)
    character_to_index = {character_id: index for index, character_id in enumerate(identities)}
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
    gallery_dataset = SyntheticCardDataset(
        rows=gallery_rows,
        dataset_root=dataset_root,
        character_to_index=character_to_index,
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
                character_id=character_id,
                embedding=prototypes[index].detach().cpu().tolist(),
                sample_count=counts[character_id],
            )
            for index, character_id in enumerate(identities)
        ],
    )
    save_gallery_artifact(output_path, artifact)
    print(json.dumps({"output": str(output_path), "identity_count": len(identities)}, indent=2))


def evaluate_checkpoint(checkpoint_path: Path, subset: str, device_name: str, output_path: Path | None = None) -> None:
    from torch.utils.data import DataLoader

    model, config, identities, device = load_model_from_checkpoint(checkpoint_path, device_name)
    manifest_path = Path(config.data.manifest_path)
    dataset_root = manifest_path.parent
    rows = load_rows(manifest_path)
    character_to_index = {character_id: index for index, character_id in enumerate(identities)}
    prepared = prepare_runtime_subsets(
        rows,
        train_subset=config.data.train_subset,
        val_subset=config.data.val_subset,
        test_subset=config.data.test_subset,
        gallery_subset=config.data.gallery_subset,
        gallery_count_per_identity=config.data.gallery_count_per_identity,
        seed=config.trainer.seed,
    )

    gallery_dataset = SyntheticCardDataset(
        rows=prepared["gallery"],
        dataset_root=dataset_root,
        character_to_index=character_to_index,
        image_size=config.data.image_size,
    )
    query_dataset = SyntheticCardDataset(
        rows=prepared[subset],
        dataset_root=dataset_root,
        character_to_index=character_to_index,
        image_size=config.data.image_size,
    )
    loader_kwargs = {
        "batch_size": config.data.batch_size,
        "shuffle": False,
        "num_workers": config.data.num_workers,
        "pin_memory": config.data.pin_memory,
        "collate_fn": collate_samples,
    }
    gallery_loader = DataLoader(gallery_dataset, **loader_kwargs)
    query_loader = DataLoader(query_dataset, **loader_kwargs)
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


def dump_star_crops(config: TrainConfig, subset: str, output_dir: Path, limit: int) -> None:
    from PIL import Image
    import torch
    from torch.utils.data import DataLoader

    def crop_with_box(image_tensor: Any, card_box: Any) -> Any:
        height, width = int(image_tensor.shape[-2]), int(image_tensor.shape[-1])
        x0, y0, x1, y1 = (float(value) for value in config.model.star_roi)
        card_left = min(width - 1, max(0, int(math.floor(float(card_box[0])))))
        card_top = min(height - 1, max(0, int(math.floor(float(card_box[1])))))
        card_right = max(card_left + 1, min(width, int(math.ceil(float(card_box[2])))))
        card_bottom = max(card_top + 1, min(height, int(math.ceil(float(card_box[3])))))
        card_width = max(1, card_right - card_left)
        card_height = max(1, card_bottom - card_top)
        left = min(width - 1, max(0, int(math.floor(card_left + x0 * card_width))))
        top = min(height - 1, max(0, int(math.floor(card_top + y0 * card_height))))
        right = max(left + 1, min(width, int(math.ceil(card_left + x1 * card_width))))
        bottom = max(top + 1, min(height, int(math.ceil(card_top + y1 * card_height))))
        return image_tensor[:, top:bottom, left:right]

    manifest_path = Path(config.data.manifest_path)
    dataset_root = manifest_path.parent
    rows = load_rows(manifest_path)
    identities = build_identity_index(rows)
    character_to_index = {character_id: index for index, character_id in enumerate(identities)}
    prepared = prepare_runtime_subsets(
        rows,
        train_subset=config.data.train_subset,
        val_subset=config.data.val_subset,
        test_subset=config.data.test_subset,
        gallery_subset=config.data.gallery_subset,
        gallery_count_per_identity=config.data.gallery_count_per_identity,
        seed=config.trainer.seed,
    )
    target_rows = prepared[subset]
    dataset = SyntheticCardDataset(
        rows=target_rows,
        dataset_root=dataset_root,
        character_to_index=character_to_index,
        image_size=config.data.image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=min(config.data.batch_size, max(1, limit)),
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=False,
        collate_fn=collate_samples,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    saved = 0
    for batch in loader:
        images = batch["images"].cpu()
        card_boxes = batch["card_boxes"].cpu()
        for image_tensor, card_box, row in zip(images, card_boxes, batch["rows"]):
            crop = crop_with_box(image_tensor, card_box.tolist())
            crop = (crop * std) + mean
            crop = crop.clamp(0.0, 1.0)
            rgb = (crop.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
            path = output_dir / f"{saved:03d}_{row.character_id}_v{row.star_value}_{row.star_color}.png"
            Image.fromarray(rgb).save(path)
            saved += 1
            if saved >= limit:
                print(json.dumps({"output_dir": str(output_dir), "saved": saved, "subset": subset}, indent=2))
                return

    print(json.dumps({"output_dir": str(output_dir), "saved": saved, "subset": subset}, indent=2))


def predict_image(
    checkpoint_path: Path,
    image_path: Path,
    device_name: str,
    topk: int = 5,
    pad_ratio: float = 0.0,
    pad_mode: str = "edge",
    dump_dir: Path | None = None,
) -> None:
    from PIL import Image
    import numpy as np
    import torch

    def pad_image(image: Image.Image, ratio: float, mode: str) -> Image.Image:
        if ratio <= 0.0:
            return image
        array = np.asarray(image)
        pad_x = max(1, int(round(array.shape[1] * ratio)))
        pad_y = max(1, int(round(array.shape[0] * ratio)))
        if mode == "constant":
            padded = np.pad(
                array,
                ((pad_y, pad_y), (pad_x, pad_x), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        else:
            np_mode = "edge" if mode == "edge" else "reflect"
            padded = np.pad(array, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode=np_mode)
        return Image.fromarray(padded)

    model, config, identities, device = load_model_from_checkpoint(checkpoint_path, device_name)
    manifest_path = Path(config.data.manifest_path)
    dataset_root = manifest_path.parent
    rows = load_rows(manifest_path)
    character_to_index = {character_id: index for index, character_id in enumerate(identities)}
    prepared = prepare_runtime_subsets(
        rows,
        train_subset=config.data.train_subset,
        val_subset=config.data.val_subset,
        test_subset=config.data.test_subset,
        gallery_subset=config.data.gallery_subset,
        gallery_count_per_identity=config.data.gallery_count_per_identity,
        seed=config.trainer.seed,
    )

    from torch.utils.data import DataLoader

    gallery_dataset = SyntheticCardDataset(
        rows=prepared["gallery"],
        dataset_root=dataset_root,
        character_to_index=character_to_index,
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

    image = pad_image(Image.open(image_path).convert("RGB"), float(pad_ratio), pad_mode)
    width, height = image.size
    from .data import build_transforms

    transform = build_transforms(config.data.image_size)
    image_tensor = transform(image).unsqueeze(0).to(device)
    scale_x = config.data.image_size / width
    scale_y = config.data.image_size / height
    card_box = torch.tensor(
        [[0.0, 0.0, float(width) * scale_x, float(height) * scale_y]],
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        outputs = model(image_tensor, None, card_box, return_debug=dump_dir is not None)
        similarities = (outputs["embedding"] @ prototypes.T).squeeze(0)
        k = max(1, min(int(topk), len(identities)))
        top_values, top_indices = similarities.topk(k=k)
        star_counts, star_colors = derive_star_predictions(outputs["star_state_logits"])
        star_probabilities = outputs["star_state_logits"].softmax(dim=1).squeeze(0)
        star_state_index = int(star_probabilities.argmax().item())
        star_confidence = float(star_probabilities[star_state_index].item())
        star_top_values, star_top_indices = star_probabilities.topk(k=min(3, star_probabilities.shape[0]))
        assist_prediction = bool((outputs["assist_logits"].sigmoid() >= 0.5).item())

    result = {
        "image": str(image_path),
        "predicted_character_id": identities[int(top_indices[0].item())],
        "predicted_star_state": star_state_label(star_state_index),
        "predicted_star_value": int(star_counts[0].item()),
        "predicted_star_color": "blue" if int(star_colors[0].item()) == 1 else "yellow",
        "predicted_star_confidence": star_confidence,
        "predicted_assist": assist_prediction,
        "top_star_states": [
            {
                "state": star_state_label(int(index.item())),
                "probability": float(value.item()),
            }
            for value, index in zip(star_top_values, star_top_indices)
        ],
        "top_matches": [
            {
                "character_id": identities[int(index.item())],
                "similarity": float(value.item()),
            }
            for value, index in zip(top_values, top_indices)
        ],
    }
    if dump_dir is not None:
        def crop_from_card(image_obj: Image.Image, roi: list[float]) -> Image.Image:
            import math

            x0, y0, x1, y1 = (float(value) for value in roi)
            image_width, image_height = image_obj.size
            left = min(image_width - 1, max(0, int(math.floor(x0 * image_width))))
            top = min(image_height - 1, max(0, int(math.floor(y0 * image_height))))
            right = max(left + 1, min(image_width, int(math.ceil(x1 * image_width))))
            bottom = max(top + 1, min(image_height, int(math.ceil(y1 * image_height))))
            return image_obj.crop((left, top, right, bottom))

        def tensor_to_pil(image_tensor: Any) -> Image.Image:
            tensor = image_tensor.detach().cpu()
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            tensor = (tensor * std) + mean
            tensor = tensor.clamp(0.0, 1.0)
            array = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
            return Image.fromarray(array)

        dump_dir.mkdir(parents=True, exist_ok=True)
        image.save(dump_dir / "full.png")
        crop_from_card(image, list(config.model.star_roi)).save(dump_dir / "star.png")
        crop_from_card(image, list(config.model.assist_roi)).save(dump_dir / "assist.png")
        if "star_images" in outputs:
            star_model_input = tensor_to_pil(outputs["star_images"][0])
            star_model_input.save(dump_dir / "star_model_input.png")
        if "star_attention_map" in outputs:
            attention_map = outputs["star_attention_map"][0, 0].detach().cpu()
            attention_map = attention_map / attention_map.max().clamp_min(1e-8)
            attention_rgb = (attention_map.numpy() * 255.0).round().astype("uint8")
            attention_image = Image.fromarray(attention_rgb, mode="L")
            attention_image = attention_image.resize(star_model_input.size, Image.Resampling.BILINEAR)
            attention_image.save(dump_dir / "star_attention.png")
            base = np.asarray(star_model_input).astype(np.float32)
            heat = np.asarray(attention_image).astype(np.float32) / 255.0
            overlay = base.copy()
            overlay[..., 0] = np.clip((0.55 * overlay[..., 0]) + (0.45 * 255.0 * heat), 0.0, 255.0)
            overlay[..., 1] = np.clip(0.70 * overlay[..., 1], 0.0, 255.0)
            overlay[..., 2] = np.clip(0.70 * overlay[..., 2], 0.0, 255.0)
            Image.fromarray(overlay.round().astype("uint8")).save(dump_dir / "star_attention_overlay.png")
        (dump_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
