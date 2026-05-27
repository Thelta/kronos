from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .character_select import annotate_vis_image, extract_students
from .config import CHARACTER_SELECT_CONFIG
from .ocr_engine import OCRBackendConfig, create_ocr_engine
from .raid import extract_raid_fields
from .result import extract_result_fields
from .schemas import OCRDump
from .video import analyze_video

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SceneDetectionResult:
    scene: str | None
    matched_keywords: dict[str, list[dict[str, object]]]
    matched_scenes: list[str]
    failure_reason: str | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kronos-analyzer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dump_parser = subparsers.add_parser("dump-ocr", help="Run RapidOCR and write dumps plus visualization.")
    dump_parser.add_argument("--input", required=True, help="Image file or directory of images.")
    dump_parser.add_argument("--output", default="outputs/ocr", help="Output directory.")
    dump_parser.add_argument(
        "--glob",
        default="*.png",
        help="Glob used when --input points to a directory.",
    )
    dump_parser.add_argument(
        "--identify-scene",
        default="ocr",
        choices=("ocr", "auto", "character_select", "raid", "result"),
        help="Run OCR only, auto-identify the scene, or force a scene-specific pipeline.",
    )
    dump_parser.set_defaults(handler=handle_dump_ocr)

    video_parser = subparsers.add_parser("analyze-video", help="Analyze one video and write session outputs.")
    video_parser.add_argument("--input", required=True, help="Video file to analyze.")
    video_parser.add_argument("--output", default="outputs/video", help="Output directory.")
    video_parser.set_defaults(handler=handle_analyze_video)
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


def handle_dump_ocr(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images(input_path, args.glob)
    engine = create_ocr_engine(OCRBackendConfig())

    for image_path in image_paths:
        image_array = load_image_array(image_path)
        run = engine.run(image_array, image_path.name)
        dump = run.dump

        json_path = output_dir / f"{image_path.stem}.json"
        txt_path = output_dir / f"{image_path.stem}.txt"
        vis_path = output_dir / f"{image_path.stem}.vis.png"

        json_path.write_text(json.dumps(asdict(dump), indent=2, ensure_ascii=False), encoding="utf-8")
        txt_path.write_text(render_text_dump(dump), encoding="utf-8")
        run.write_visualization(vis_path)

        scene = resolve_scene_mode(args.identify_scene, dump)
        if args.identify_scene != "ocr":
            scene_payload: dict[str, Any] = {"requested": args.identify_scene, "resolved": scene}
            if scene == "character_select":
                students, star_results = extract_students(
                    dump=dump,
                    image_array=image_array,
                    image_stem=image_path.stem,
                    output_dir=output_dir,
                    engine=engine,
                    config=CHARACTER_SELECT_CONFIG,
                )
                scene_payload["students"] = [asdict(item) for item in students]
                scene_payload["star_values"] = [asdict(item) for item in star_results]
                if star_results:
                    annotate_vis_image(vis_path, star_results)
            elif scene == "raid":
                scene_payload["raid"] = asdict(extract_raid_fields(dump))
            elif scene == "result":
                scene_payload["result"] = asdict(extract_result_fields(dump))
            scene_path = output_dir / f"{image_path.stem}.scene.json"
            scene_path.write_text(json.dumps(scene_payload, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"{image_path.name} -> {json_path.name}, {txt_path.name}, {vis_path.name}")

    return 0


def handle_analyze_video(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"Video input not found: {input_path}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = create_ocr_engine(OCRBackendConfig())
    result = analyze_video(video_path=input_path, engine=engine)

    session_path = output_dir / f"{input_path.stem}.session.json"
    events_path = output_dir / f"{input_path.stem}.events.json"

    session_payload = asdict(result.session) if result.session is not None else None
    events_payload = [asdict(event) for event in result.events]

    session_path.write_text(json.dumps(session_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    events_path.write_text(json.dumps(events_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(
        "Video analysis completed for %s: session=%s, retained_events=%d",
        input_path.name,
        "yes" if result.session is not None else "no",
        len(result.events),
    )
    print(f"{input_path.name} -> {session_path.name}, {events_path.name}")
    return 0


def resolve_scene_mode(mode: str, dump: OCRDump) -> str | None:
    if mode == "ocr":
        return None
    if mode != "auto":
        return mode
    return detect_scene(dump)


def detect_scene(dump: OCRDump) -> str | None:
    return detect_scene_details(dump).scene


def detect_scene_details(dump: OCRDump) -> SceneDetectionResult:
    keywords = {
        "character_select": ["総力戦編成", "開始スキル", "部隊2", "プリセット"],
        "raid": ["BATTLE BOSS", "COST", "AUTO", "PAUSE", "区ギブアップ"],
        "result": ["Battle Complete", "Ranking Point", "DEFEAT", "戦闘時間"]
    }
    
    from rapidfuzz import fuzz
    lines = " ".join(dump.combined_text.splitlines())
    logger.info(f"dump {lines}")
    
    if dump.line_count == 0:
        return SceneDetectionResult(scene=None, matched_keywords=[], matched_scenes=[], failure_reason="")
    
    for scene, scene_keywords in keywords.items():
        count = 0
        for keyword in scene_keywords:
            if max([fuzz.ratio(keyword, line.text) for line in dump.lines]) > 80:
                count = count + 1
                logger.info(f"{keyword} found.")
            
        if count > 0:
            return SceneDetectionResult(scene=scene, matched_keywords=[], matched_scenes=[], failure_reason="")
        
    return SceneDetectionResult(scene=None, matched_keywords=[], matched_scenes=[], failure_reason="")
        
        


def format_scene_detection_reason(result: SceneDetectionResult) -> str:
    if result.scene is not None:
        return f"selected={result.scene}"
    return f"unresolved reason={result.failure_reason}"


def summarize_combined_text(text: str, limit: int = 120) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[: limit - 3]}..."


def collect_images(input_path: Path, pattern: str) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(path for path in input_path.glob(pattern) if path.is_file())
    raise FileNotFoundError(f"Input not found: {input_path}")


def load_image_array(image_path: Path) -> Any:
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    return image


def render_text_dump(dump: OCRDump) -> str:
    output_lines: list[str] = [f"image: {dump.image}", f"line_count: {dump.line_count}", ""]
    for index, line in enumerate(dump.lines, start=1):
        output_lines.append(f"[{index}] score={line.score}")
        output_lines.append(line.text)
        output_lines.append(json.dumps(line.box, ensure_ascii=False))
        output_lines.append("")
    return "\n".join(output_lines).strip() + "\n"


def configure_logging() -> None:
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
