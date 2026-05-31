from __future__ import annotations

import re
from dataclasses import dataclass

from .schemas import OCRDump, OCRLine

_COST_LABEL_X = 1580.0
_COST_LABEL_Y = 1300.0
_MIN_COST_VALUE = -11
_MAX_COST_VALUE = 11
_MINUS_TRANSLATION = str.maketrans({
    "−": "-",
    "－": "-",
    "ー": "-",
    "―": "-",
    "—": "-",
    "–": "-",
    "ｰ": "-",
})


@dataclass
class RaidResult:
    boss_remaining_hp: int | None
    boss_total_hp: int | None
    timer: str
    cost: int | None = None
    brightness_recovery_triggered: bool = False


def extract_raid_fields(dump: OCRDump) -> RaidResult:
    hp_line = find_boss_hp_line(dump.lines)
    timer_line = find_timer_line(dump.lines)
    cost_label_line = find_cost_label_line(dump.lines)
    cost_value_line = find_cost_value_line(dump.lines, cost_label_line)
    remaining_hp, total_hp = parse_hp_pair(hp_line.text if hp_line is not None else "")
    return RaidResult(
        boss_remaining_hp=remaining_hp,
        boss_total_hp=total_hp,
        timer=extract_timer_value(timer_line.text if timer_line is not None else ""),
        cost=parse_cost_value(cost_value_line.text if cost_value_line is not None else ""),
    )


def find_boss_hp_line(lines: list[OCRLine]) -> OCRLine | None:
    candidates: list[tuple[float, OCRLine]] = []
    for line in lines:
        if "/" not in line.text:
            continue
        if len(re.findall(r"\d", line.text)) < 8:
            continue
        left, top, right, _ = bounds_from_box(line.box)
        center_x = (left + right) / 2.0
        score = abs(center_x - 1280.0) + (top * 0.5)
        candidates.append((score, line))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def find_timer_line(lines: list[OCRLine]) -> OCRLine | None:
    candidates: list[tuple[float, OCRLine]] = []
    for line in lines:
        timer = extract_timer_value(line.text)
        if not timer:
            continue
        left, top, right, _ = bounds_from_box(line.box)
        center_x = (left + right) / 2.0
        score = abs(center_x - 2250.0) + top
        candidates.append((score, line))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def find_cost_label_line(lines: list[OCRLine]) -> OCRLine | None:
    candidates: list[tuple[float, OCRLine]] = []
    for line in lines:
        normalized = re.sub(r"[^A-Z]", "", line.text.upper())
        if normalized != "COST":
            continue
        if not line.box:
            continue
        left, top, right, bottom = bounds_from_box(line.box)
        center_x = (left + right) / 2.0
        center_y = (top + bottom) / 2.0
        score = abs(center_x - _COST_LABEL_X) + abs(center_y - _COST_LABEL_Y)
        candidates.append((score, line))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def find_cost_value_line(lines: list[OCRLine], cost_label_line: OCRLine | None) -> OCRLine | None:
    if cost_label_line is None or not cost_label_line.box:
        return None

    anchor_left, anchor_top, anchor_right, anchor_bottom = bounds_from_box(cost_label_line.box)
    anchor_center_x = (anchor_left + anchor_right) / 2.0
    anchor_center_y = (anchor_top + anchor_bottom) / 2.0
    anchor_width = max(anchor_right - anchor_left, 1.0)

    candidates: list[tuple[float, OCRLine]] = []
    for line in lines:
        if line is cost_label_line or not line.box:
            continue
        if parse_cost_value(line.text) is None:
            continue

        left, top, right, bottom = bounds_from_box(line.box)
        center_x = (left + right) / 2.0
        center_y = (top + bottom) / 2.0
        width = max(right - left, 1.0)
        overlap = max(0.0, min(anchor_right, right) - max(anchor_left, left))
        overlap_ratio = overlap / min(anchor_width, width)
        center_x_delta = abs(center_x - anchor_center_x)
        vertical_gap = top - anchor_bottom

        if center_y <= anchor_center_y:
            continue
        if vertical_gap > 140:
            continue
        if overlap_ratio < 0.35 and center_x_delta > max(anchor_width * 0.75, 40.0):
            continue

        score = max(vertical_gap, 0.0) * 2.0 + center_x_delta - (overlap_ratio * 25.0)
        candidates.append((score, line))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def parse_hp_pair(text: str) -> tuple[int | None, int | None]:
    if not text or "/" not in text:
        return None, None
    left_text, right_text = text.split("/", maxsplit=1)
    return normalize_number(left_text), normalize_number(right_text)


def normalize_number(text: str) -> int | None:
    digits = re.sub(r"\D", "", text)
    return int(digits) if digits else None


def extract_timer_value(text: str) -> str:
    match = re.search(r"\d{2}:\d{2}\.\d{3}", text)
    return match.group(0) if match else ""


def parse_cost_value(text: str) -> int | None:
    normalized = text.translate(_MINUS_TRANSLATION).replace(" ", "").strip()
    if not re.fullmatch(r"-?\d{1,2}", normalized):
        return None
    value = int(normalized)
    if value < _MIN_COST_VALUE or value > _MAX_COST_VALUE:
        return None
    return value


def bounds_from_box(box: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in box]
    ys = [point[1] for point in box]
    return min(xs), min(ys), max(xs), max(ys)
