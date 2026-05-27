from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import dump_star_crops, evaluate_checkpoint, export_gallery, predict_image, train_from_config
from .prep import main as prepare_manifest_main
from .training_config import load_config
from .yolo_config import load_config as load_yolo_config
from .yolo_pipeline import prepare_yolo_dataset, train_yolo_from_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kronos-train")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare-manifest", help="Repartition a shittim synthetic manifest.")
    prepare_parser.add_argument("--input-manifest", type=Path, required=True)
    prepare_parser.add_argument("--output-manifest", type=Path, required=True)
    prepare_parser.add_argument("--seed", type=int, default=7)
    prepare_parser.add_argument("--gallery-count", type=int, default=4)
    prepare_parser.add_argument("--train-count", type=int, default=32)
    prepare_parser.add_argument("--val-count", type=int, default=8)
    prepare_parser.add_argument("--test-count", type=int, default=8)
    prepare_parser.set_defaults(handler=handle_prepare_manifest)

    train_parser = subparsers.add_parser("train", help="Train MobileNetV4-small ArcFace model on prepared synthetic data.")
    train_parser.add_argument("--config", type=Path, required=True)
    train_parser.add_argument("--override", action="append", default=[])
    train_parser.add_argument("--device", default="cuda")
    train_parser.add_argument("--dry-run", action="store_true")
    train_parser.set_defaults(handler=handle_train)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate retrieval and auxiliary heads from a checkpoint.")
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.add_argument("--subset", default="test_query")
    evaluate_parser.add_argument("--device", default="cuda")
    evaluate_parser.add_argument("--output", type=Path, default=None)
    evaluate_parser.set_defaults(handler=handle_evaluate)

    export_parser = subparsers.add_parser("export-gallery", help="Export gallery prototypes from a checkpoint.")
    export_parser.add_argument("--checkpoint", type=Path, required=True)
    export_parser.add_argument("--output", type=Path, required=True)
    export_parser.add_argument("--device", default="cuda")
    export_parser.set_defaults(handler=handle_export_gallery)

    dump_parser = subparsers.add_parser("dump-star-crops", help="Save debug star ROI crops from a manifest subset.")
    dump_parser.add_argument("--config", type=Path, required=True)
    dump_parser.add_argument("--override", action="append", default=[])
    dump_parser.add_argument("--subset", default="val_query")
    dump_parser.add_argument("--output-dir", type=Path, required=True)
    dump_parser.add_argument("--limit", type=int, default=20)
    dump_parser.set_defaults(handler=handle_dump_star_crops)

    predict_parser = subparsers.add_parser("predict-image", help="Run one image through a checkpoint and print predictions.")
    predict_parser.add_argument("--checkpoint", type=Path, required=True)
    predict_parser.add_argument("--image", type=Path, required=True)
    predict_parser.add_argument("--device", default="cuda")
    predict_parser.add_argument("--topk", type=int, default=5)
    predict_parser.add_argument("--pad-ratio", type=float, default=0.0)
    predict_parser.add_argument("--pad-mode", choices=("edge", "reflect", "constant"), default="edge")
    predict_parser.add_argument("--dump-dir", type=Path, default=None)
    predict_parser.set_defaults(handler=handle_predict_image)

    yolo_prepare_parser = subparsers.add_parser("prepare-yolo", help="Prepare the merged YOLO26 dataset from synthetic and reviewed-real inputs.")
    yolo_prepare_parser.add_argument("--config", type=Path, required=True)
    yolo_prepare_parser.add_argument("--override", action="append", default=[])
    yolo_prepare_parser.set_defaults(handler=handle_prepare_yolo)

    yolo_train_parser = subparsers.add_parser("train-yolo", help="Train the merged YOLO26 detector from config.")
    yolo_train_parser.add_argument("--config", type=Path, required=True)
    yolo_train_parser.add_argument("--override", action="append", default=[])
    yolo_train_parser.add_argument("--skip-prepare", action="store_true")
    yolo_train_parser.set_defaults(handler=handle_train_yolo)
    return parser


def handle_prepare_manifest(args: argparse.Namespace) -> int:
    prepare_manifest_main(
        [
            "--input-manifest",
            str(args.input_manifest),
            "--output-manifest",
            str(args.output_manifest),
            "--seed",
            str(args.seed),
            "--gallery-count",
            str(args.gallery_count),
            "--train-count",
            str(args.train_count),
            "--val-count",
            str(args.val_count),
            "--test-count",
            str(args.test_count),
        ]
    )
    return 0


def handle_train(args: argparse.Namespace) -> int:
    config = load_config(args.config, args.override)
    train_from_config(config, args.device, dry_run=args.dry_run)
    return 0


def handle_evaluate(args: argparse.Namespace) -> int:
    evaluate_checkpoint(args.checkpoint, args.subset, args.device, args.output)
    return 0


def handle_export_gallery(args: argparse.Namespace) -> int:
    export_gallery(args.checkpoint, args.output, args.device)
    return 0


def handle_dump_star_crops(args: argparse.Namespace) -> int:
    config = load_config(args.config, args.override)
    dump_star_crops(config, args.subset, args.output_dir, args.limit)
    return 0


def handle_predict_image(args: argparse.Namespace) -> int:
    predict_image(args.checkpoint, args.image, args.device, args.topk, args.pad_ratio, args.pad_mode, args.dump_dir)
    return 0


def handle_prepare_yolo(args: argparse.Namespace) -> int:
    config = load_yolo_config(args.config, args.override)
    prepare_yolo_dataset(config)
    return 0


def handle_train_yolo(args: argparse.Namespace) -> int:
    config = load_yolo_config(args.config, args.override)
    train_yolo_from_config(config, prepare=not args.skip_prepare)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
