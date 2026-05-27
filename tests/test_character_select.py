from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kronos_analyzer.character_select import (  # noqa: E402
    Slot,
    build_slots_from_lv_lines_no_anchor,
    extract_students,
)
from kronos_analyzer.config import CHARACTER_SELECT_CONFIG  # noqa: E402
from kronos_analyzer.schemas import OCRDump, OCRLine  # noqa: E402


class FakeEngine:
    def __init__(self, crop_text: str = "5"):
        self.crop_text = crop_text
        self.calls: list[tuple[str, str]] = []

    def recognize_crops(self, crops, profile: str = "default", model_preset=None):
        self.calls.append(("recognize_crops", profile))
        return [OCRLine(text=self.crop_text, score=0.95, box=[])]


def make_line(text: str, left: float, top: float, right: float, bottom: float, score: float = 0.95) -> OCRLine:
    return OCRLine(text=text, score=score, box=[[left, top], [right, top], [right, bottom], [left, bottom]])


class CharacterSelectTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = patch("kronos_analyzer.character_select.load_student_names", return_value=[])
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_extract_students_returns_empty_when_boundary_missing(self) -> None:
        dump = OCRDump(
            image="frame.png",
            line_count=1,
            combined_text="Lv.90",
            lines=[make_line("Lv.90", 35, 70, 52, 78)],
        )

        students, star_results = extract_students(
            dump=dump,
            image_array=np.zeros((100, 100, 3), dtype=np.uint8),
            image_stem="frame",
            output_dir=None,
            engine=FakeEngine(),
            config=CHARACTER_SELECT_CONFIG,
        )

        self.assertEqual(students, [])
        self.assertEqual(star_results, [])

    def test_extract_students_with_boundary(self) -> None:
        """When boundary anchors and STRIKER are present, Lv lines produce slots."""
        dump = OCRDump(
            image="frame.png",
            line_count=5,
            combined_text="部隊4\nSTRIKER\nLv.90\nHoshino\n出撃",
            lines=[
                make_line("部隊4", 10, 400, 100, 450, 0.99),
                make_line("STRIKER", 50, 500, 160, 530, 0.99),
                make_line("Lv.90", 200, 510, 280, 540, 0.94),
                make_line("Hoshino", 285, 500, 380, 530, 0.96),
                make_line("出撃", 300, 600, 380, 650, 0.99),
            ],
        )
        engine = FakeEngine()

        with tempfile.TemporaryDirectory() as temp_dir:
            students, star_results = extract_students(
                dump=dump,
                image_array=np.zeros((1080, 1920, 3), dtype=np.uint8),
                image_stem="frame",
                output_dir=Path(temp_dir),
                engine=engine,
                config=CHARACTER_SELECT_CONFIG,
            )

        self.assertEqual(len(students), 1)
        self.assertIn("Hoshino", students[0].name)
        self.assertEqual(students[0].level, "90")

    def test_build_slots_two_rows(self) -> None:
        """Two rows of Lv lines are separated into striker and special rows by y order."""
        lines = [
            make_line("Lv.90", 200, 750, 280, 780, 0.94),
            make_line("Lv.85", 400, 750, 480, 780, 0.94),
            make_line("Lv.70", 600, 750, 680, 780, 0.94),
            make_line("Lv.60", 300, 900, 380, 930, 0.94),
            make_line("Lv.55", 500, 900, 580, 930, 0.94),
        ]

        slots = build_slots_from_lv_lines_no_anchor(lines, CHARACTER_SELECT_CONFIG)

        self.assertEqual(len(slots), 5)
        striker_slots = [s for s in slots if s.role == "striker"]
        special_slots = [s for s in slots if s.role == "special"]
        self.assertEqual(len(striker_slots), 3)
        self.assertEqual(len(special_slots), 2)
        # Striker slots should be sorted by x
        self.assertLess(striker_slots[0].name_region[0], striker_slots[1].name_region[0])
        self.assertLess(striker_slots[1].name_region[0], striker_slots[2].name_region[0])

    def test_build_slots_returns_empty_when_no_lv_lines(self) -> None:
        lines = [make_line("Hoshino", 200, 750, 280, 780)]

        slots = build_slots_from_lv_lines_no_anchor(lines, CHARACTER_SELECT_CONFIG)

        self.assertEqual(slots, [])


if __name__ == "__main__":
    unittest.main()
