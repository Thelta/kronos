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
    find_cost_label_line,
    find_cost_value_line,
    extract_timer_value,
    find_boss_hp_line,
    find_timer_line,
    parse_cost_value,
    parse_hp_pair,
)
from .schemas import OCRDump
from .session_aggregator import TEAM_RESET_THRESHOLD_MS, parse_raid_timer_ms

logger = logging.getLogger(__name__)

_EMA_ALPHA = 0.2
_MIN_SAMPLES_FOR_REOCR = 3
_CROP_PADDING = 15
BRIGHTNESS_THRESHOLD = 90.0


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
    cost_tracker: RaidFieldTracker = field(default_factory=RaidFieldTracker)
    prev_remaining_hp: int | None = None
    prev_total_hp: int | None = None
    prev_timer_ms: int | None = None
    brightness_was_below_threshold: bool = False

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
        cost_problem = self._detect_cost_problem(result)

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
                    cost=result.cost,
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
                    cost=result.cost,
                )
                timer_problem = self._detect_timer_problem(result)

        if cost_problem is not None and self.cost_tracker.sample_count >= _MIN_SAMPLES_FOR_REOCR:
            logger.info("Cost anomaly detected: %s; attempting re-OCR", cost_problem)
            reocr_cost = self._reocr_cost(image_array, engine)
            if reocr_cost is not None:
                result = RaidResult(
                    boss_remaining_hp=result.boss_remaining_hp,
                    boss_total_hp=result.boss_total_hp,
                    timer=result.timer,
                    cost=reocr_cost,
                )
                cost_problem = self._detect_cost_problem(result)

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

        if cost_problem is None and result.cost is not None:
            cost_label_line = find_cost_label_line(dump.lines)
            cost_line = find_cost_value_line(dump.lines, cost_label_line)
            if cost_line is not None and cost_line.box:
                left, top, right, bottom = bounds_from_box(cost_line.box)
                self.cost_tracker.update_box(
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

        brightness_recovery_triggered = self._update_brightness_state(image_array)
        return RaidResult(
            boss_remaining_hp=result.boss_remaining_hp,
            boss_total_hp=result.boss_total_hp,
            timer=result.timer,
            cost=result.cost,
            brightness_recovery_triggered=brightness_recovery_triggered,
        )

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

    def _detect_cost_problem(self, result: RaidResult) -> str | None:
        if result.cost is None:
            return "cost_parse_failure"
        return None

    def _update_brightness_state(self, image_array: Any) -> bool:
        brightness = _mean_frame_brightness(image_array)
        if brightness < BRIGHTNESS_THRESHOLD:
            self.brightness_was_below_threshold = True
            return False
        if self.brightness_was_below_threshold:
            self.brightness_was_below_threshold = False
            return True
        return False

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

    def _reocr_cost(self, image_array: Any, engine: OCREngine) -> int | None:
        box = self.cost_tracker.avg_box
        if box is None:
            return None
        crop = _crop_from_box(image_array, box)
        if crop is None:
            return None
        lines = engine.recognize_crops([crop], profile="default")
        for line in lines:
            cost = parse_cost_value(line.text)
            if cost is not None:
                return cost
        return None


def _crop_from_box(image_array: Any, box: AxisAlignedBox) -> Any:
    h, w = image_array.shape[:2]
    left = max(0, int(box.left) - _CROP_PADDING)
    top = max(0, int(box.top) - _CROP_PADDING)
    right = min(w, int(box.right) + _CROP_PADDING)
    bottom = min(h, int(box.bottom) + _CROP_PADDING)
    if right <= left or bottom <= top:
        return None
    return image_array[top:bottom, left:right]


def _mean_frame_brightness(image_array: Any) -> float:
    if image_array.ndim == 2:
        return float(np.mean(image_array))
    if image_array.ndim == 3 and image_array.shape[2] >= 3:
        b = image_array[..., 0].astype(np.float32)
        g = image_array[..., 1].astype(np.float32)
        r = image_array[..., 2].astype(np.float32)
        return float(np.mean((0.114 * b) + (0.587 * g) + (0.299 * r)))
    return float(np.mean(image_array))
