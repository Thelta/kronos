from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OCRLine:
    text: str
    score: float | None
    box: list[list[float]]


@dataclass
class OCRDump:
    image: str
    line_count: int
    combined_text: str
    lines: list[OCRLine]
