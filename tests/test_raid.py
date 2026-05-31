from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kronos_analyzer.raid import (  # noqa: E402
    extract_raid_fields,
    find_cost_label_line,
    find_cost_value_line,
    parse_cost_value,
)
from kronos_analyzer.schemas import OCRDump, OCRLine  # noqa: E402


def make_line(text: str, box: list[list[float]]) -> OCRLine:
    return OCRLine(text=text, score=0.99, box=box)


def make_dump(lines: list[OCRLine]) -> OCRDump:
    return OCRDump(
        image="raid.png",
        line_count=len(lines),
        combined_text="\n".join(line.text for line in lines),
        lines=lines,
    )


class RaidParsingTests(unittest.TestCase):
    def test_extract_raid_fields_reads_cost_below_cost_label(self) -> None:
        dump = make_dump([
            make_line("68,633,666/70,000,000", [[1161, 106], [1469, 106], [1469, 141], [1161, 141]]),
            make_line("03:24.867", [[2154, 66], [2357, 66], [2357, 115], [2154, 115]]),
            make_line("COST", [[1552, 1279], [1625, 1279], [1625, 1316], [1552, 1316]]),
            make_line("3", [[1563, 1314], [1607, 1314], [1607, 1365], [1563, 1365]]),
        ])

        result = extract_raid_fields(dump)

        self.assertEqual(result.boss_remaining_hp, 68633666)
        self.assertEqual(result.boss_total_hp, 70000000)
        self.assertEqual(result.timer, "03:24.867")
        self.assertEqual(result.cost, 3)

    def test_parse_cost_value_accepts_signed_bounds(self) -> None:
        self.assertEqual(parse_cost_value("-11"), -11)
        self.assertEqual(parse_cost_value("11"), 11)
        self.assertEqual(parse_cost_value("−1"), -1)

    def test_parse_cost_value_rejects_missing_or_out_of_range_values(self) -> None:
        self.assertIsNone(parse_cost_value(""))
        self.assertIsNone(parse_cost_value("12"))
        self.assertIsNone(parse_cost_value("-12"))
        self.assertIsNone(parse_cost_value("A AUTO"))

    def test_extract_raid_fields_returns_none_when_cost_label_or_value_missing(self) -> None:
        without_label = make_dump([
            make_line("68,633,666/70,000,000", [[1161, 106], [1469, 106], [1469, 141], [1161, 141]]),
            make_line("03:24.867", [[2154, 66], [2357, 66], [2357, 115], [2154, 115]]),
            make_line("3", [[1563, 1314], [1607, 1314], [1607, 1365], [1563, 1365]]),
        ])
        invalid_value = make_dump([
            make_line("68,633,666/70,000,000", [[1161, 106], [1469, 106], [1469, 141], [1161, 141]]),
            make_line("03:24.867", [[2154, 66], [2357, 66], [2357, 115], [2154, 115]]),
            make_line("COST", [[1552, 1279], [1625, 1279], [1625, 1316], [1552, 1316]]),
            make_line("12", [[1563, 1314], [1607, 1314], [1607, 1365], [1563, 1365]]),
        ])

        self.assertIsNone(extract_raid_fields(without_label).cost)
        self.assertIsNone(extract_raid_fields(invalid_value).cost)

    def test_find_cost_value_line_prefers_valid_candidate_below_label(self) -> None:
        lines = [
            make_line("COST", [[1552, 1279], [1625, 1279], [1625, 1316], [1552, 1316]]),
            make_line("9", [[1700, 1200], [1740, 1200], [1740, 1240], [1700, 1240]]),
            make_line("3", [[1563, 1314], [1607, 1314], [1607, 1365], [1563, 1365]]),
            make_line("8", [[1710, 1314], [1750, 1314], [1750, 1365], [1710, 1365]]),
        ]

        label = find_cost_label_line(lines)
        value = find_cost_value_line(lines, label)

        self.assertIsNotNone(label)
        self.assertIsNotNone(value)
        self.assertEqual(value.text, "3")


if __name__ == "__main__":
    unittest.main()
