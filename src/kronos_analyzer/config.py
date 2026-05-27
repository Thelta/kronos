from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CharacterSelectBoundaryConfig:
    top_keyword: str = "部隊4"
    bottom_keyword: str = "出撃"
    min_match_score: float = 80.0


@dataclass(frozen=True)
class CharacterSelectConfig:
    lv_pattern: str = r"^L[Vv][\.\s]?\d+"
    boundary: CharacterSelectBoundaryConfig = CharacterSelectBoundaryConfig()
    name_above_multiplier: float = 0.7
    row_gap_multiplier: float = 2.0
    star_above_height_multiplier: float = 1.05
    star_width_multiplier: float = 0.7
    star_bottom_padding_multiplier: float = 0.0
    star_left_trim_multiplier: float = 0.12
    star_right_trim_multiplier: float = 0.08


CHARACTER_SELECT_CONFIG = CharacterSelectConfig()
