from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import CharacterSelectBoundaryConfig, CharacterSelectConfig
from .ocr_engine import OCREngine, OCRModelPreset
from .schemas import OCRDump, OCRLine
from .student_names import load_student_names, resolve_name

logger = logging.getLogger(__name__)


@dataclass
class StarValueResult:
    lv_text: str
    lv_box: list[list[float]]
    region: list[float]
    normal_value: str
    normal_score: float | None
    crop_value: str
    crop_score: float | None
    chosen_value: str
    chosen_source: str
    star_color: str
    crop_file: str | None = None


@dataclass
class StudentResult:
    slot_index: int
    name: str
    level: str
    star_yellow: str | None
    star_blue: str | None


@dataclass
class NameMatch:
    text: str
    primary_line: OCRLine
    secondary_line: OCRLine | None = None


@dataclass(frozen=True)
class Slot:
    slot_index: int
    role: str
    level_line: OCRLine
    name_region: tuple[float, float, float, float]


def extract_star_value(
    *,
    lines: list[OCRLine],
    level_line: OCRLine,
    image_array: Any,
    image_stem: str,
    output_dir: Path | None,
    engine: OCREngine,
    config: CharacterSelectConfig,
    model_preset: OCRModelPreset | None = None,
) -> StarValueResult | None:
    region = box_to_region(
        level_line.box,
        above_height_multiplier=config.star_above_height_multiplier,
        width_multiplier=config.star_width_multiplier,
        bottom_padding_multiplier=config.star_bottom_padding_multiplier,
        left_trim_multiplier=config.star_left_trim_multiplier,
        right_trim_multiplier=config.star_right_trim_multiplier,
    )
    if region is None:
        return None

    normal_candidate = find_digit_candidate_above_lv(
        lines,
        level_line,
        above_height_multiplier=config.star_above_height_multiplier,
        width_multiplier=config.star_width_multiplier,
        bottom_padding_multiplier=config.star_bottom_padding_multiplier,
        left_trim_multiplier=config.star_left_trim_multiplier,
        right_trim_multiplier=config.star_right_trim_multiplier,
    )
    color_crop = crop_above_box(
        image_array,
        level_line.box,
        above_height_multiplier=config.star_above_height_multiplier,
        width_multiplier=config.star_width_multiplier,
        bottom_padding_multiplier=config.star_bottom_padding_multiplier,
        left_trim_multiplier=config.star_left_trim_multiplier,
        right_trim_multiplier=config.star_right_trim_multiplier,
    )
    star_color = detect_star_color(color_crop)

    crop_path_name: str | None = None
    crop_line: OCRLine | None = None
    if normal_candidate is None:
        crop = color_crop
        crop_lines = engine.recognize_crops([crop], profile="digits", model_preset=model_preset)
        crop_dump = OCRDump(
            image=f"{image_stem}.slot_lv.png",
            line_count=len(crop_lines),
            combined_text="\n".join(line.text for line in crop_lines if line.text),
            lines=crop_lines,
        )
        if output_dir is not None:
            crop_path = output_dir / f"{image_stem}.slot_lv.png"
            cv2.imwrite(str(crop_path), crop)
            crop_path_name = crop_path.name
            crop_json_path = output_dir / f"{image_stem}.slot_lv.json"
            crop_txt_path = output_dir / f"{image_stem}.slot_lv.txt"
            crop_json_path.write_text(
                json.dumps(asdict(crop_dump), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            crop_txt_path.write_text(render_text_dump(crop_dump), encoding="utf-8")
        crop_line = crop_dump.lines[0] if crop_dump.lines else None

    raw_value = normal_candidate.text if normal_candidate is not None else (crop_line.text if crop_line is not None else "")
    chosen_source = "normal" if normal_candidate is not None else "specialized"
    chosen_value = _validate_star_value(normalize_to_digits(raw_value), star_color)
    if not chosen_value:
        return None
    return StarValueResult(
        lv_text=level_line.text,
        lv_box=level_line.box,
        region=[float(region[0]), float(region[1]), float(region[2]), float(region[3])],
        normal_value=normal_candidate.text if normal_candidate is not None else "",
        normal_score=normal_candidate.score if normal_candidate is not None else None,
        crop_value=crop_line.text if crop_line is not None else "",
        crop_score=crop_line.score if crop_line is not None else None,
        chosen_value=chosen_value,
        chosen_source=chosen_source,
        star_color=star_color,
        crop_file=crop_path_name,
    )


def extract_students(
    *,
    dump: OCRDump,
    image_array: Any,
    image_stem: str,
    output_dir: Path | None,
    engine: OCREngine,
    config: CharacterSelectConfig,
    model_preset: OCRModelPreset | None = None,
) -> tuple[list[StudentResult], list[StarValueResult]]:
    lines = dump.lines

    top_boundary, bottom_boundary = find_boundary_anchors(lines, config.boundary)
    if top_boundary is not None and bottom_boundary is not None:
        lines = filter_lines_by_boundary(lines, top_boundary, bottom_boundary)
        logger.info(
            "Character select boundary for %s: top=%r bottom=%r filtered=%d/%d",
            image_stem,
            top_boundary.text,
            bottom_boundary.text,
            len(lines),
            len(dump.lines),
        )
        has_striker = any(
            normalize_match_text(line.text) and "striker" in normalize_match_text(line.text)
            for line in lines
            if line.text
        )
        if not has_striker:
            logger.info("Character select parse skipped for %s: STRIKER not found in boundary", image_stem)
            return [], []
    else:
        logger.info("Character select parse skipped for %s: boundary anchors not found", image_stem)
        return [], []

    slots = build_slots_from_lv_lines_no_anchor(lines, config)
    logger.info("Character select slots for %s: %d", image_stem, len(slots))

    known_names = load_student_names()

    students: list[StudentResult] = []
    star_results: list[StarValueResult] = []
    for slot in slots:
        name_match = find_name_candidate_in_region(lines, slot.name_region, config.lv_pattern)
        raw_name = name_match.text if name_match is not None else ""
        resolved = resolve_name(raw_name, known_names) if raw_name else ""

        star_result = extract_star_value(
            lines=lines,
            level_line=slot.level_line,
            image_array=image_array,
            image_stem=f"{image_stem}.slot_{slot.slot_index}",
            output_dir=output_dir,
            engine=engine,
            config=config,
            model_preset=model_preset,
        )
        if star_result is not None:
            star_results.append(star_result)
        star_value = star_result.chosen_value if star_result is not None else ""
        star_color = star_result.star_color if star_result is not None else ""
        students.append(
            StudentResult(
                slot_index=slot.slot_index,
                name=resolved,
                level=extract_level_value(slot.level_line.text),
                star_yellow=star_value if star_color == "yellow" else None,
                star_blue=star_value if star_color == "blue" else None,
            )
        )
    return students, star_results


def find_boundary_anchors(
    lines: list[OCRLine],
    config: CharacterSelectBoundaryConfig,
) -> tuple[OCRLine | None, OCRLine | None]:
    """Find the top (部隊4) and bottom (出撃) boundary anchors.

    Returns (top_anchor, bottom_anchor) where each may be None if not found.
    Uses normalized containment check rather than fuzzy matching for speed.
    """
    norm_top = normalize_match_text(config.top_keyword)
    norm_bottom = normalize_match_text(config.bottom_keyword)

    top_anchor: OCRLine | None = None
    bottom_anchor: OCRLine | None = None

    for line in lines:
        if not line.text or not line.box:
            continue
        norm = normalize_match_text(line.text)
        if not norm:
            continue
        if norm_top and norm_top in norm:
            top_anchor = line
        if norm_bottom and norm_bottom in norm:
            bottom_anchor = line

    return top_anchor, bottom_anchor


def filter_lines_by_boundary(
    lines: list[OCRLine],
    top_anchor: OCRLine,
    bottom_anchor: OCRLine,
) -> list[OCRLine]:
    """Keep only lines whose vertical center is between the bottom edge of
    top_anchor and the top edge of bottom_anchor."""
    _, _, _, top_boundary = bounds_from_box(top_anchor.box)
    bottom_boundary = min(p[1] for p in bottom_anchor.box)

    filtered: list[OCRLine] = []
    for line in lines:
        if not line.box:
            continue
        _, line_top, _, line_bottom = bounds_from_box(line.box)
        center_y = (line_top + line_bottom) / 2.0
        if top_boundary <= center_y <= bottom_boundary:
            filtered.append(line)
    return filtered


def normalize_match_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"[\W_]+", "", normalized, flags=re.UNICODE)


def build_slots_from_lv_lines_no_anchor(
    lines: list[OCRLine],
    config: CharacterSelectConfig,
) -> list[Slot]:
    """Build slots from Lv lines without a STRIKER anchor.

    Rows are sorted top-to-bottom: the first row is striker, the rest are special.
    """
    pattern = re.compile(config.lv_pattern, re.IGNORECASE)
    lv_lines = [line for line in lines if line.box and pattern.search(line.text)]
    if not lv_lines:
        return []

    lv_heights = []
    for line in lv_lines:
        _, top, _, bottom = bounds_from_box(line.box)
        lv_heights.append(max(1.0, bottom - top))
    avg_lv_height = sum(lv_heights) / len(lv_heights)

    gap_threshold = avg_lv_height * config.row_gap_multiplier
    lv_with_y = []
    for line in lv_lines:
        _, top, _, bottom = bounds_from_box(line.box)
        center_y = (top + bottom) / 2.0
        lv_with_y.append((center_y, line))
    lv_with_y.sort(key=lambda item: item[0])

    rows: list[list[OCRLine]] = []
    current_row: list[OCRLine] = [lv_with_y[0][1]]
    current_y = lv_with_y[0][0]
    for center_y, line in lv_with_y[1:]:
        if abs(center_y - current_y) > gap_threshold:
            rows.append(current_row)
            current_row = [line]
            current_y = center_y
        else:
            current_row.append(line)
            current_y = (current_y * (len(current_row) - 1) + center_y) / len(current_row)
    rows.append(current_row)

    def row_center_y(row: list[OCRLine]) -> float:
        ys = []
        for line in row:
            _, top, _, bottom = bounds_from_box(line.box)
            ys.append((top + bottom) / 2.0)
        return sum(ys) / len(ys)

    rows.sort(key=row_center_y)

    def line_center_x(line: OCRLine) -> float:
        left, _, right, _ = bounds_from_box(line.box)
        return (left + right) / 2.0

    for row in rows:
        row.sort(key=line_center_x)

    slots: list[Slot] = []
    slot_index = 1
    for row_idx, row in enumerate(rows):
        role = "striker" if row_idx == 0 else "special"
        for i, lv_line in enumerate(row):
            lv_left, lv_top, lv_right, lv_bottom = bounds_from_box(lv_line.box)
            lv_height = max(1.0, lv_bottom - lv_top)

            name_left = lv_right
            if i + 1 < len(row):
                next_left, _, _, _ = bounds_from_box(row[i + 1].box)
                name_right = next_left
            else:
                if len(row) >= 2:
                    name_gaps = []
                    for j in range(len(row) - 1):
                        _, _, j_right, _ = bounds_from_box(row[j].box)
                        j_next_left, _, _, _ = bounds_from_box(row[j + 1].box)
                        name_gaps.append(j_next_left - j_right)
                    avg_name_gap = sum(name_gaps) / len(name_gaps)
                else:
                    avg_name_gap = lv_height * 8.0
                name_right = lv_right + avg_name_gap

            name_top = lv_top - (lv_height * config.name_above_multiplier)
            name_bottom = lv_bottom

            slots.append(
                Slot(
                    slot_index=slot_index,
                    role=role,
                    level_line=lv_line,
                    name_region=(name_left, name_top, name_right, name_bottom),
                )
            )
            slot_index += 1

    return slots


def find_name_candidate_in_region(
    lines: list[OCRLine],
    region: tuple[float, float, float, float],
    lv_pattern: str,
) -> NameMatch | None:
    region_left, region_top, region_right, region_bottom = region
    candidates = [
        line
        for line in lines
        if line.box
        and box_center_in_region(line.box, region)
        and _is_text_candidate(line, lv_pattern, min_length=1)
    ]
    if not candidates:
        return None

    # Group lines into rows by y overlap, then sort within each row by x
    rows: list[list[OCRLine]] = []
    for line in candidates:
        line_top = min(p[1] for p in line.box)
        line_bottom = max(p[1] for p in line.box)
        line_center_y = (line_top + line_bottom) / 2.0
        placed = False
        for row in rows:
            row_top = min(min(p[1] for p in r.box) for r in row)
            row_bottom = max(max(p[1] for p in r.box) for r in row)
            if row_top <= line_center_y <= row_bottom:
                row.append(line)
                placed = True
                break
        if not placed:
            rows.append([line])

    # Sort rows by y, lines within each row by x
    rows.sort(key=lambda row: min(min(p[1] for p in line.box) for line in row))
    for row in rows:
        row.sort(key=lambda line: min(p[0] for p in line.box))

    ordered = [line for row in rows for line in row]
    primary = ordered[0]
    name_text = "".join(line.text.strip() for line in ordered)
    name_text = name_text.replace("(", "（").replace(")", "）")
    secondary = ordered[1] if len(ordered) > 1 else None
    return NameMatch(text=name_text, primary_line=primary, secondary_line=secondary)


def _is_text_candidate(line: OCRLine, lv_pattern: str, min_length: int = 2) -> bool:
    if not line.text or not line.box:
        return False
    stripped = line.text.strip()
    if len(stripped) < min_length:
        return False
    if re.search(lv_pattern, stripped, re.IGNORECASE):
        return False
    if normalize_to_digits(stripped) == stripped and stripped:
        return False
    return True


def box_overlap_area(box: list[list[float]], region: tuple[float, float, float, float]) -> float:
    box_left, box_top, box_right, box_bottom = bounds_from_box(box)
    region_left, region_top, region_right, region_bottom = region
    overlap_width = max(0.0, min(box_right, region_right) - max(box_left, region_left))
    overlap_height = max(0.0, min(box_bottom, region_bottom) - max(box_top, region_top))
    return overlap_width * overlap_height


def annotate_vis_image(vis_path: Path, star_results: list[StarValueResult]) -> None:
    vis_image = cv2.imread(str(vis_path))
    if vis_image is None:
        return
    for star_result in star_results:
        draw_star_result(vis_image, star_result)
    cv2.imwrite(str(vis_path), vis_image)


def find_digit_candidate_above_lv(
    lines: list[OCRLine],
    lv_line: OCRLine,
    *,
    above_height_multiplier: float,
    width_multiplier: float,
    bottom_padding_multiplier: float,
    left_trim_multiplier: float,
    right_trim_multiplier: float,
) -> OCRLine | None:
    region = box_to_region(
        lv_line.box,
        above_height_multiplier=above_height_multiplier,
        width_multiplier=width_multiplier,
        bottom_padding_multiplier=bottom_padding_multiplier,
        left_trim_multiplier=left_trim_multiplier,
        right_trim_multiplier=right_trim_multiplier,
    )
    if region is None:
        return None

    candidates: list[OCRLine] = []
    for line in lines:
        if line is lv_line or not line.box:
            continue
        digits = normalize_to_digits(line.text)
        if not digits or not is_probable_digit_candidate(line.text, digits):
            continue
        if box_intersects_region(line.box, region):
            candidates.append(OCRLine(text=digits, score=line.score, box=line.box))

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: (
            item.score if item.score is not None else -1.0,
            len(item.text),
        ),
        reverse=True,
    )
    return candidates[0]


def box_to_region(
    box: list[list[float]],
    *,
    above_height_multiplier: float,
    width_multiplier: float,
    bottom_padding_multiplier: float,
    left_trim_multiplier: float,
    right_trim_multiplier: float,
) -> tuple[float, float, float, float] | None:
    if not box:
        return None

    xs = [point[0] for point in box]
    ys = [point[1] for point in box]
    left = min(xs)
    right = max(xs)
    top = min(ys)
    bottom = max(ys)
    box_width = max(1.0, right - left)
    box_height = max(1.0, bottom - top)

    region_left = left + (box_width * left_trim_multiplier)
    region_right = left + (box_width * width_multiplier) - (box_width * right_trim_multiplier)
    region_top = top - (box_height * above_height_multiplier)
    region_bottom = top + (box_height * bottom_padding_multiplier)
    return (region_left, region_top, region_right, region_bottom)


def box_center_in_region(box: list[list[float]], region: tuple[float, float, float, float]) -> bool:
    """Check that the box's center point falls within the region."""
    if not box:
        return False
    left, top, right, bottom = region
    xs = [point[0] for point in box]
    ys = [point[1] for point in box]
    center_x = (min(xs) + max(xs)) / 2.0
    center_y = (min(ys) + max(ys)) / 2.0
    return left <= center_x <= right and top <= center_y <= bottom


def box_intersects_region(box: list[list[float]], region: tuple[float, float, float, float]) -> bool:
    if not box:
        return False
    left, top, right, bottom = region
    xs = [point[0] for point in box]
    ys = [point[1] for point in box]
    box_left = min(xs)
    box_right = max(xs)
    box_top = min(ys)
    box_bottom = max(ys)
    return not (
        box_right < left
        or box_left > right
        or box_bottom < top
        or box_top > bottom
    )


def draw_star_result(image: Any, star_result: StarValueResult) -> None:
    draw_polygon(image, star_result.lv_box, (255, 128, 0))
    if star_result.region:
        left, top, right, bottom = star_result.region
        cv2.rectangle(
            image,
            (int(round(left)), int(round(top))),
            (int(round(right)), int(round(bottom))),
            (0, 255, 255),
            2,
        )

    label = (
        f"{star_result.lv_text} N:{star_result.normal_value or '-'} "
        f"C:{star_result.crop_value or '-'} -> {star_result.chosen_value or '?'} "
        f"{star_result.star_color or '?'} "
        f"[{star_result.chosen_source}]"
    )
    anchor_x = int(round(min(point[0] for point in star_result.lv_box))) if star_result.lv_box else 0
    anchor_y = int(round(min(point[1] for point in star_result.lv_box))) - 10 if star_result.lv_box else 20
    anchor_y = max(anchor_y, 20)
    cv2.putText(
        image,
        label,
        (anchor_x, anchor_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )


def draw_polygon(image: Any, box: list[list[float]], color: tuple[int, int, int]) -> None:
    if not box:
        return
    points = np.array([[int(round(x)), int(round(y))] for x, y in box], dtype=np.int32)
    cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)


def crop_above_box(
    image: Any,
    box: list[list[float]],
    *,
    above_height_multiplier: float,
    width_multiplier: float,
    bottom_padding_multiplier: float,
    left_trim_multiplier: float,
    right_trim_multiplier: float,
) -> Any:
    if not box:
        raise ValueError("Cannot crop above an empty OCR box.")

    xs = [point[0] for point in box]
    ys = [point[1] for point in box]
    left = min(xs)
    right = max(xs)
    top = min(ys)
    bottom = max(ys)

    box_width = max(1.0, right - left)
    box_height = max(1.0, bottom - top)

    crop_left = max(0, int(round(left + (box_width * left_trim_multiplier))))
    crop_right = min(
        image.shape[1],
        int(round(left + (box_width * width_multiplier) - (box_width * right_trim_multiplier))),
    )
    crop_top = max(0, int(round(top - (box_height * above_height_multiplier))))
    crop_bottom = min(image.shape[0], int(round(top + (box_height * bottom_padding_multiplier))))

    if crop_right <= crop_left or crop_bottom <= crop_top:
        raise ValueError("Computed an empty crop for Lv.* region.")
    return image[crop_top:crop_bottom, crop_left:crop_right]


def normalize_to_digits(text: str) -> str:
    confusion_map = str.maketrans(
        {
            "O": "0",
            "o": "0",
            "Q": "0",
            "I": "1",
            "l": "1",
            "|": "1",
            "S": "5",
            "s": "5",
            "B": "8",
            "b": "6",
            "Z": "2",
        }
    )
    normalized = text.translate(confusion_map)
    return "".join(character for character in normalized if character.isdigit())


def is_probable_digit_candidate(text: str, digits: str) -> bool:
    stripped = text.strip()
    if not digits or len(digits) > 2:
        return False
    allowed_non_digits = {".", " ", ":"}
    meaningful_characters = [
        character for character in stripped if not character.isspace()
    ]
    if not meaningful_characters:
        return False
    digit_like_count = sum(
        1
        for character in meaningful_characters
        if character.isdigit() or character.upper() in {"O", "Q", "I", "L", "S", "B", "Z", "|"}
    )
    other_count = sum(
        1
        for character in meaningful_characters
        if not (character.isdigit() or character.upper() in {"O", "Q", "I", "L", "S", "B", "Z", "|"} or character in allowed_non_digits)
    )
    return digit_like_count >= 1 and other_count == 0


def ensure_three_channel(image: Any) -> Any:
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 3:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)


def detect_star_color(crop: Any) -> str:
    crop_bgr = ensure_three_channel(crop)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)

    yellow_mask = cv2.inRange(hsv, np.array([18, 80, 30]), np.array([45, 255, 255]))
    blue_mask = cv2.inRange(hsv, np.array([85, 70, 30]), np.array([135, 255, 255]))

    yellow_score = int(cv2.countNonZero(yellow_mask))
    blue_score = int(cv2.countNonZero(blue_mask))
    minimum_score = max(12, int(crop.shape[0] * crop.shape[1] * 0.01))

    if yellow_score < minimum_score and blue_score < minimum_score:
        return ""
    if yellow_score >= blue_score:
        return "yellow"
    return "blue"



def _validate_star_value(value: str, star_color: str) -> str:
    if not value or not value.isdigit():
        return ""
    n = int(value)
    if star_color == "yellow" and 1 <= n <= 5:
        return value
    if star_color == "blue" and 1 <= n <= 4:
        return value
    return ""


def extract_level_value(text: str) -> str:
    match = re.search(r"(\d+)", text)
    return match.group(1) if match else ""


def bounds_from_box(box: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in box]
    ys = [point[1] for point in box]
    return min(xs), min(ys), max(xs), max(ys)


def render_text_dump(dump: OCRDump) -> str:
    output_lines: list[str] = [f"image: {dump.image}", f"line_count: {dump.line_count}", ""]
    for index, line in enumerate(dump.lines, start=1):
        output_lines.append(f"[{index}] score={line.score}")
        output_lines.append(line.text)
        output_lines.append(json.dumps(line.box, ensure_ascii=False))
        output_lines.append("")
    return "\n".join(output_lines).strip() + "\n"
