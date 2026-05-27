from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .ocr_engine import OCREngine
from .raid import (
    RaidResult,
    bounds_from_box,
    extract_raid_fields,
    extract_timer_value,
    find_boss_hp_line,
    find_timer_line,
    parse_hp_pair,
)
from .schemas import OCRDump
from .session_aggregator import TEAM_RESET_THRESHOLD_MS, parse_raid_timer_ms

logger = logging.getLogger(__name__)

_EMA_ALPHA = 0.2
_MIN_SAMPLES_FOR_REOCR = 3
_CROP_PADDING = 15


@dataclass
class AxisAlignedBox:
    left: float
    top: float
    right: float
    bottom: float


@dataclass
class RaidFieldTracker:
    avg_box: AxisAlignedBox | None = None
    sample_count: int = 0

    def update_box(self, box: AxisAlignedBox) -> None:
        if self.avg_box is None:
            self.avg_box = AxisAlignedBox(
                left=box.left, top=box.top, right=box.right, bottom=box.bottom
            )
        else:
            alpha = _EMA_ALPHA
            inv = 1.0 - alpha
            self.avg_box = AxisAlignedBox(
                left=inv * self.avg_box.left + alpha * box.left,
                top=inv * self.avg_box.top + alpha * box.top,
                right=inv * self.avg_box.right + alpha * box.right,
                bottom=inv * self.avg_box.bottom + alpha * box.bottom,
            )
        self.sample_count += 1


@dataclass
class RaidTracker:
    hp_tracker: RaidFieldTracker = field(default_factory=RaidFieldTracker)
    timer_tracker: RaidFieldTracker = field(default_factory=RaidFieldTracker)
    prev_remaining_hp: int | None = None
    prev_total_hp: int | None = None
    prev_timer_ms: int | None = None

    def extract(
        self, dump: OCRDump, image_array: Any, engine: OCREngine
    ) -> RaidResult:
        result = extract_raid_fields(dump)

        # Check for team change before anomaly detection
        if self._is_team_change(result):
            logger.info("Team change detected, resetting previous values")
            self.prev_remaining_hp = None
            self.prev_total_hp = None
            self.prev_timer_ms = None

        hp_problem = self._detect_hp_problem(result)
        timer_problem = self._detect_timer_problem(result)

        # Attempt re-OCR for HP if needed
        if hp_problem is not None and self.hp_tracker.sample_count >= _MIN_SAMPLES_FOR_REOCR:
            logger.info("HP anomaly detected: %s; attempting re-OCR", hp_problem)
            reocr_hp = self._reocr_hp(image_array, engine)
            if reocr_hp is not None:
                remaining, total = reocr_hp
                result = RaidResult(
                    boss_remaining_hp=remaining,
                    boss_total_hp=total,
                    timer=result.timer,
                )
                hp_problem = self._detect_hp_problem(result)

        # Attempt re-OCR for timer if needed
        if timer_problem is not None and self.timer_tracker.sample_count >= _MIN_SAMPLES_FOR_REOCR:
            logger.info("Timer anomaly detected: %s; attempting re-OCR", timer_problem)
            reocr_timer = self._reocr_timer(image_array, engine)
            if reocr_timer is not None:
                result = RaidResult(
                    boss_remaining_hp=result.boss_remaining_hp,
                    boss_total_hp=result.boss_total_hp,
                    timer=reocr_timer,
                )
                timer_problem = self._detect_timer_problem(result)

        # Update bbox trackers on successful reads
        if hp_problem is None:
            hp_line = find_boss_hp_line(dump.lines)
            if hp_line is not None and hp_line.box:
                left, top, right, bottom = bounds_from_box(hp_line.box)
                self.hp_tracker.update_box(
                    AxisAlignedBox(left=left, top=top, right=right, bottom=bottom)
                )

        if timer_problem is None:
            timer_line = find_timer_line(dump.lines)
            if timer_line is not None and timer_line.box:
                left, top, right, bottom = bounds_from_box(timer_line.box)
                self.timer_tracker.update_box(
                    AxisAlignedBox(left=left, top=top, right=right, bottom=bottom)
                )

        # Update previous values for next-frame comparison
        if result.boss_remaining_hp is not None:
            self.prev_remaining_hp = result.boss_remaining_hp
        if result.boss_total_hp is not None:
            self.prev_total_hp = result.boss_total_hp
        timer_ms = parse_raid_timer_ms(result.timer)
        if timer_ms is not None:
            self.prev_timer_ms = timer_ms

        return result

    def _detect_hp_problem(self, result: RaidResult) -> str | None:
        if result.boss_remaining_hp is None:
            return "remaining_hp_parse_failure"
        if (
            self.prev_total_hp is not None
            and result.boss_total_hp is not None
            and result.boss_total_hp != self.prev_total_hp
        ):
            return "total_hp_changed"
        if (
            self.prev_remaining_hp is not None
            and result.boss_remaining_hp > self.prev_remaining_hp
        ):
            return "remaining_hp_increased"
        return None

    def _detect_timer_problem(self, result: RaidResult) -> str | None:
        if not result.timer:
            return "timer_parse_failure"
        timer_ms = parse_raid_timer_ms(result.timer)
        if timer_ms is None:
            return "timer_parse_failure"
        if self.prev_timer_ms is not None and timer_ms > self.prev_timer_ms:
            return "timer_increased"
        return None

    def _is_team_change(self, result: RaidResult) -> bool:
        if self.prev_timer_ms is None:
            return False
        timer_ms = parse_raid_timer_ms(result.timer)
        if timer_ms is None:
            return False
        return (timer_ms - self.prev_timer_ms) >= TEAM_RESET_THRESHOLD_MS

    def _reocr_hp(
        self, image_array: Any, engine: OCREngine
    ) -> tuple[int | None, int | None] | None:
        box = self.hp_tracker.avg_box
        if box is None:
            return None
        crop = _crop_from_box(image_array, box)
        if crop is None:
            return None
        lines = engine.recognize_crops([crop])
        if not lines:
            return None
        text = lines[0].text
        remaining, total = parse_hp_pair(text)
        if remaining is None:
            return None
        return remaining, total

    def _reocr_timer(self, image_array: Any, engine: OCREngine) -> str | None:
        box = self.timer_tracker.avg_box
        if box is None:
            return None
        crop = _crop_from_box(image_array, box)
        if crop is None:
            return None
        lines = engine.recognize_crops([crop])
        if not lines:
            return None
        timer = extract_timer_value(lines[0].text)
        return timer if timer else None


def _crop_from_box(image_array: Any, box: AxisAlignedBox) -> Any:
    h, w = image_array.shape[:2]
    left = max(0, int(box.left) - _CROP_PADDING)
    top = max(0, int(box.top) - _CROP_PADDING)
    right = min(w, int(box.right) + _CROP_PADDING)
    bottom = min(h, int(box.bottom) + _CROP_PADDING)
    if right <= left or bottom <= top:
        return None
    return image_array[top:bottom, left:right]
