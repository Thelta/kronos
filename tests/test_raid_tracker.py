from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

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
    cost_text: str | None = "3",
    hp_box: list[list[float]] | None = None,
    timer_box: list[list[float]] | None = None,
    cost_label_box: list[list[float]] | None = None,
    cost_box: list[list[float]] | None = None,
) -> OCRDump:
    if hp_box is None:
        hp_box = [[1200, 50], [1360, 50], [1360, 80], [1200, 80]]
    if timer_box is None:
        timer_box = [[2200, 50], [2300, 50], [2300, 80], [2200, 80]]
    if cost_label_box is None:
        cost_label_box = [[1550, 1280], [1625, 1280], [1625, 1316], [1550, 1316]]
    if cost_box is None:
        cost_box = [[1560, 1314], [1608, 1314], [1608, 1365], [1560, 1365]]
    lines = [
        OCRLine(text=hp_text, score=0.95, box=hp_box),
        OCRLine(text=timer_text, score=0.95, box=timer_box),
    ]
    if cost_text is not None:
        lines.extend([
            OCRLine(text="COST", score=0.99, box=cost_label_box),
            OCRLine(text=cost_text, score=0.99, box=cost_box),
        ])
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
        assert r1.cost == 3

        r2 = tracker.extract(_make_dump("45000/100000", "02:55.000"), image, engine)
        assert r2.boss_remaining_hp == 45000
        assert r2.timer == "02:55.000"
        assert r2.cost == 3

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


class TestCostTracking:
    def test_reocr_triggered_on_invalid_cost(self):
        tracker = RaidTracker()
        engine = _make_engine()
        image = _make_image()

        for cost in ["3", "4", "5"]:
            tracker.extract(_make_dump("50000/100000", "03:00.000", cost), image, engine)

        engine.recognize_crops.return_value = [
            OCRLine(text="-11", score=0.9, box=[])
        ]
        result = tracker.extract(
            _make_dump("49000/100000", "02:59.000", "12"), image, engine
        )

        engine.recognize_crops.assert_called_once()
        assert result.cost == -11

    def test_invalid_cost_does_not_update_tracker_box(self):
        tracker = RaidTracker()
        engine = _make_engine()
        image = _make_image()

        for cost in ["3", "4", "5"]:
            tracker.extract(_make_dump("50000/100000", "03:00.000", cost), image, engine)

        assert tracker.cost_tracker.sample_count == 3
        engine.recognize_crops.return_value = [
            OCRLine(text="garbled", score=0.1, box=[])
        ]
        result = tracker.extract(
            _make_dump("49000/100000", "02:59.000", "garbled"), image, engine
        )

        assert result.cost is None
        assert tracker.cost_tracker.sample_count == 3

    def test_cost_can_jump_without_triggering_anomaly(self):
        tracker = RaidTracker()
        engine = _make_engine()
        image = _make_image()

        result_1 = tracker.extract(_make_dump("50000/100000", "03:00.000", "0"), image, engine)
        result_2 = tracker.extract(_make_dump("49000/100000", "02:59.000", "11"), image, engine)
        result_3 = tracker.extract(_make_dump("48000/100000", "02:58.000", "-11"), image, engine)

        assert result_1.cost == 0
        assert result_2.cost == 11
        assert result_3.cost == -11
        engine.recognize_crops.assert_not_called()


class TestBrightnessRecovery:
    def test_dark_then_bright_triggers_once(self):
        tracker = RaidTracker()
        engine = _make_engine()

        dark = np.full((32, 32, 3), 80, dtype=np.uint8)
        bright = np.full((32, 32, 3), 120, dtype=np.uint8)

        first = tracker.extract(_make_dump(), dark, engine)
        second = tracker.extract(_make_dump(timer_text="02:59.000"), bright, engine)
        third = tracker.extract(_make_dump(timer_text="02:58.000"), bright, engine)

        assert first.brightness_recovery_triggered is False
        assert second.brightness_recovery_triggered is True
        assert third.brightness_recovery_triggered is False

    def test_multiple_dark_frames_arm_single_recovery(self):
        tracker = RaidTracker()
        engine = _make_engine()

        dark = np.full((32, 32, 3), 70, dtype=np.uint8)
        bright = np.full((32, 32, 3), 140, dtype=np.uint8)

        first = tracker.extract(_make_dump(), dark, engine)
        second = tracker.extract(_make_dump(timer_text="02:59.000"), dark, engine)
        third = tracker.extract(_make_dump(timer_text="02:58.000"), bright, engine)

        assert first.brightness_recovery_triggered is False
        assert second.brightness_recovery_triggered is False
        assert third.brightness_recovery_triggered is True

    def test_bright_to_bright_never_triggers(self):
        tracker = RaidTracker()
        engine = _make_engine()
        bright = np.full((32, 32, 3), 120, dtype=np.uint8)

        first = tracker.extract(_make_dump(), bright, engine)
        second = tracker.extract(_make_dump(timer_text="02:59.000"), bright, engine)

        assert first.brightness_recovery_triggered is False
        assert second.brightness_recovery_triggered is False

    def test_dark_to_dark_never_triggers(self):
        tracker = RaidTracker()
        engine = _make_engine()
        dark = np.full((32, 32, 3), 60, dtype=np.uint8)

        first = tracker.extract(_make_dump(), dark, engine)
        second = tracker.extract(_make_dump(timer_text="02:59.000"), dark, engine)

        assert first.brightness_recovery_triggered is False
        assert second.brightness_recovery_triggered is False

    def test_new_recovery_requires_new_dark_period(self):
        tracker = RaidTracker()
        engine = _make_engine()

        dark = np.full((32, 32, 3), 75, dtype=np.uint8)
        bright = np.full((32, 32, 3), 130, dtype=np.uint8)

        first = tracker.extract(_make_dump(), dark, engine)
        second = tracker.extract(_make_dump(timer_text="02:59.000"), bright, engine)
        third = tracker.extract(_make_dump(timer_text="02:58.000"), bright, engine)
        fourth = tracker.extract(_make_dump(timer_text="02:57.000"), dark, engine)
        fifth = tracker.extract(_make_dump(timer_text="02:56.000"), bright, engine)

        assert first.brightness_recovery_triggered is False
        assert second.brightness_recovery_triggered is True
        assert third.brightness_recovery_triggered is False
        assert fourth.brightness_recovery_triggered is False
        assert fifth.brightness_recovery_triggered is True


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
