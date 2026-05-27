from __future__ import annotations

import re
from dataclasses import dataclass

from .schemas import OCRDump, OCRLine


@dataclass
class RaidResult:
    boss_remaining_hp: int | None
    boss_total_hp: int | None
    timer: str


def extract_raid_fields(dump: OCRDump) -> RaidResult:
    hp_line = find_boss_hp_line(dump.lines)
    timer_line = find_timer_line(dump.lines)
    remaining_hp, total_hp = parse_hp_pair(hp_line.text if hp_line is not None else "")
    return RaidResult(
        boss_remaining_hp=remaining_hp,
        boss_total_hp=total_hp,
        timer=extract_timer_value(timer_line.text if timer_line is not None else ""),
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


def bounds_from_box(box: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in box]
    ys = [point[1] for point in box]
    return min(xs), min(ys), max(xs), max(ys)
