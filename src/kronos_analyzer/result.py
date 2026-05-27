from __future__ import annotations

import re
from dataclasses import dataclass

from .schemas import OCRDump, OCRLine


@dataclass
class ResultSceneResult:
    ranking_point: int | None
    timer: str


def extract_result_fields(dump: OCRDump) -> ResultSceneResult:
    ranking_label = find_line_containing(dump.lines, "rankingpoint")
    ranking_line = ranking_label if contains_number(ranking_label) else find_numeric_neighbor(dump.lines, ranking_label)
    timer_label = find_line_containing(dump.lines, "戦闘時間")
    timer_line = timer_label if extract_timer_value(timer_label.text if timer_label is not None else "") else find_numeric_neighbor(dump.lines, timer_label)
    return ResultSceneResult(
        ranking_point=normalize_number(ranking_line.text if ranking_line is not None else ""),
        timer=extract_timer_value(timer_line.text if timer_line is not None else (timer_label.text if timer_label is not None else "")),
    )


def find_line_containing(lines: list[OCRLine], needle: str) -> OCRLine | None:
    lowered_needle = needle.lower()
    for line in lines:
        if lowered_needle in line.text.lower():
            return line
    return None


def find_numeric_neighbor(lines: list[OCRLine], anchor: OCRLine | None) -> OCRLine | None:
    if anchor is None or not anchor.box:
        return None
    _, anchor_top, anchor_right, anchor_bottom = bounds_from_box(anchor.box)
    anchor_center_y = (anchor_top + anchor_bottom) / 2.0
    candidates: list[tuple[float, OCRLine]] = []
    for line in lines:
        if line is anchor or not line.box or not contains_number(line):
            continue
        left, top, _, bottom = bounds_from_box(line.box)
        center_y = (top + bottom) / 2.0
        x_gap = left - anchor_right
        if x_gap < -30:
            continue
        if abs(center_y - anchor_center_y) > 80:
            continue
        score = abs(center_y - anchor_center_y) + x_gap
        candidates.append((score, line))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def contains_number(line: OCRLine | None) -> bool:
    return bool(line is not None and re.search(r"\d", line.text))


def normalize_number(text: str) -> int | None:
    digits = re.sub(r"\D", "", text)
    return int(digits) if digits else None


def extract_timer_value(text: str) -> str:
    match = re.search(r"\d{2}:\d{2}\.\d{3}", text)
    return match.group(0) if match else ""


def bounds_from_box(box: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in box]
    ys = [point[1] for point in box]
    return min(xs), min(ys), max(xs), max(ys)
