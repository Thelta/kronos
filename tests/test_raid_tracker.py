from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from kronos_analyzer.raid import RaidResult
from kronos_analyzer.raid_tracker import (
    AxisAlignedBox,
    RaidFieldTracker,
    RaidTracker,
    _crop_from_box,
)
from kronos_analyzer.schemas import OCRDump, OCRLine


def _make_dump(
    hp_text: str = "50000/100000",
    timer_text: str = "03:00.000",
    hp_box: list[list[float]] | None = None,
    timer_box: list[list[float]] | None = None,
) -> OCRDump:
    if hp_box is None:
        hp_box = [[1200, 50], [1360, 50], [1360, 80], [1200, 80]]
    if timer_box is None:
        timer_box = [[2200, 50], [2300, 50], [2300, 80], [2200, 80]]
    lines = [
        OCRLine(text=hp_text, score=0.95, box=hp_box),
        OCRLine(text=timer_text, score=0.95, box=timer_box),
    ]
    return OCRDump(
        image="test.png",
        line_count=len(lines),
        combined_text="\n".join(l.text for l in lines),
        lines=lines,
    )


def _make_image(w: int = 2560, h: int = 1440) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_engine() -> MagicMock:
    return MagicMock()


class TestColdStart:
    def test_first_frames_no_fallback(self):
        tracker = RaidTracker()
        engine = _make_engine()
        image = _make_image()

        r1 = tracker.extract(_make_dump("50000/100000", "03:00.000"), image, engine)
        assert r1.boss_remaining_hp == 50000
        assert r1.boss_total_hp == 100000
        assert r1.timer == "03:00.000"

        r2 = tracker.extract(_make_dump("45000/100000", "02:55.000"), image, engine)
        assert r2.boss_remaining_hp == 45000
        assert r2.timer == "02:55.000"

        engine.recognize_crops.assert_not_called()


class TestEMAConvergence:
    def test_avg_box_after_5_frames(self):
        ft = RaidFieldTracker()
        boxes = [
            AxisAlignedBox(100, 50, 200, 80),
            AxisAlignedBox(102, 52, 202, 82),
            AxisAlignedBox(98, 48, 198, 78),
            AxisAlignedBox(104, 54, 204, 84),
            AxisAlignedBox(100, 50, 200, 80),
        ]
        for box in boxes:
            ft.update_box(box)

        assert ft.sample_count == 5
        assert ft.avg_box is not None
        # First box sets initial, subsequent apply EMA
        # Values should be near 100, 50, 200, 80 (the mean-ish)
        assert 98 < ft.avg_box.left < 104
        assert 48 < ft.avg_box.top < 54
        assert 198 < ft.avg_box.right < 204
        assert 78 < ft.avg_box.bottom < 84


class TestHPParseFailure:
    def test_reocr_triggered_on_none_hp(self):
        tracker = RaidTracker()
        engine = _make_engine()
        image = _make_image()

        # 3 good frames to build up bbox
        for hp in ["50000/100000", "48000/100000", "46000/100000"]:
            tracker.extract(_make_dump(hp, "03:00.000"), image, engine)

        # Now garbled HP
        engine.recognize_crops.return_value = [
            OCRLine(text="44000/100000", score=0.9, box=[])
        ]
        result = tracker.extract(
            _make_dump("garbled", "02:50.000"), image, engine
        )
        engine.recognize_crops.assert_called_once()
        assert result.boss_remaining_hp == 44000
        assert result.boss_total_hp == 100000


class TestHPIncrease:
    def test_reocr_triggered_on_hp_increase(self):
        tracker = RaidTracker()
        engine = _make_engine()
        image = _make_image()

        for hp in ["50000/100000", "48000/100000", "46000/100000"]:
            tracker.extract(_make_dump(hp, "03:00.000"), image, engine)

        # HP goes up — anomaly
        engine.recognize_crops.return_value = [
            OCRLine(text="44000/100000", score=0.9, box=[])
        ]
        result = tracker.extract(
            _make_dump("60000/100000", "02:50.000"), image, engine
        )
        engine.recognize_crops.assert_called_once()
        assert result.boss_remaining_hp == 44000


class TestTotalHPChange:
    def test_reocr_triggered_on_total_hp_change(self):
        tracker = RaidTracker()
        engine = _make_engine()
        image = _make_image()

        for hp in ["50000/100000", "48000/100000", "46000/100000"]:
            tracker.extract(_make_dump(hp, "03:00.000"), image, engine)

        # Total HP changed
        engine.recognize_crops.return_value = [
            OCRLine(text="44000/100000", score=0.9, box=[])
        ]
        result = tracker.extract(
            _make_dump("44000/200000", "02:50.000"), image, engine
        )
        engine.recognize_crops.assert_called_once()
        assert result.boss_total_hp == 100000


class TestTimerIncrease:
    def test_reocr_triggered_on_timer_increase(self):
        tracker = RaidTracker()
        engine = _make_engine()
        image = _make_image()

        # Use decreasing timers (< 3s gap so no team change)
        for timer in ["03:00.000", "02:59.000", "02:58.000"]:
            tracker.extract(_make_dump("50000/100000", timer), image, engine)

        # Timer goes up by < 3s — anomaly, not team change
        engine.recognize_crops.return_value = [
            OCRLine(text="02:57.000", score=0.9, box=[])
        ]
        result = tracker.extract(
            _make_dump("49000/100000", "02:59.500"), image, engine
        )
        engine.recognize_crops.assert_called()
        assert result.timer == "02:57.000"


class TestReOCRFails:
    def test_original_result_returned_when_reocr_fails(self):
        tracker = RaidTracker()
        engine = _make_engine()
        image = _make_image()

        for hp in ["50000/100000", "48000/100000", "46000/100000"]:
            tracker.extract(_make_dump(hp, "03:00.000"), image, engine)

        # Re-OCR returns garbage too
        engine.recognize_crops.return_value = [
            OCRLine(text="garbled_again", score=0.1, box=[])
        ]
        result = tracker.extract(
            _make_dump("garbled", "02:50.000"), image, engine
        )
        # Original garbled result is returned (remaining_hp is None)
        assert result.boss_remaining_hp is None


class TestTeamChange:
    def test_timer_jump_resets_prev_no_reocr(self):
        tracker = RaidTracker()
        engine = _make_engine()
        image = _make_image()

        for hp, timer in [
            ("50000/100000", "03:00.000"),
            ("48000/100000", "02:55.000"),
            ("46000/100000", "02:50.000"),
        ]:
            tracker.extract(_make_dump(hp, timer), image, engine)

        # Timer jumps up by >= 3s — team change
        result = tracker.extract(
            _make_dump("100000/100000", "03:00.000"), image, engine
        )
        # Should NOT trigger re-OCR because prev values were reset
        engine.recognize_crops.assert_not_called()
        assert result.boss_remaining_hp == 100000
        assert result.timer == "03:00.000"

        # Bbox trackers should still have samples
        assert tracker.hp_tracker.sample_count >= 3
        assert tracker.timer_tracker.sample_count >= 3


class TestCropFromBox:
    def test_crop_with_padding(self):
        image = np.ones((100, 200, 3), dtype=np.uint8)
        box = AxisAlignedBox(left=50, top=20, right=150, bottom=60)
        crop = _crop_from_box(image, box)
        assert crop is not None
        # With 15px padding: top=5, bottom=75, left=35, right=165
        assert crop.shape == (70, 130, 3)

    def test_crop_clamped_to_bounds(self):
        image = np.ones((100, 200, 3), dtype=np.uint8)
        box = AxisAlignedBox(left=5, top=5, right=195, bottom=95)
        crop = _crop_from_box(image, box)
        assert crop is not None
        assert crop.shape == (100, 200, 3)

    def test_invalid_crop_returns_none(self):
        image = np.ones((100, 200, 3), dtype=np.uint8)
        # Box completely outside image bounds (after padding still invalid)
        box = AxisAlignedBox(left=300, top=200, right=250, bottom=180)
        crop = _crop_from_box(image, box)
        assert crop is None
