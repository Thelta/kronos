from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from unittest.mock import patch

from kronos_analyzer.character_select import StudentResult  # noqa: E402
from kronos_analyzer.raid import RaidResult  # noqa: E402
from kronos_analyzer.result import ResultSceneResult  # noqa: E402
from kronos_analyzer.session_aggregator import (  # noqa: E402
    SceneObservation,
    _aggregate_students,
    _resolve_name,
    aggregate_session,
    parse_raid_timer_ms,
)


def make_student(name: str, *, slot_index: int = 1, level: str = "90", star_yellow: str | None = "5", star_blue: str | None = None) -> StudentResult:
    return StudentResult(
        slot_index=slot_index,
        name=name,
        level=level,
        star_yellow=star_yellow,
        star_blue=star_blue,
    )


def make_character_select(frame_index: int, video_time_ms: int, name: str = "Hoshino") -> SceneObservation:
    return SceneObservation(
        frame_index=frame_index,
        video_time_ms=video_time_ms,
        scene="character_select",
        students=[make_student(name)],
    )


def make_raid(
    frame_index: int,
    video_time_ms: int,
    timer: str,
    remaining_hp: int | None,
    total_hp: int | None = 1000,
    cost: int | None = None,
) -> SceneObservation:
    return SceneObservation(
        frame_index=frame_index,
        video_time_ms=video_time_ms,
        scene="raid",
        raid=RaidResult(
            boss_remaining_hp=remaining_hp,
            boss_total_hp=total_hp,
            timer=timer,
            cost=cost,
        ),
    )


def make_result(frame_index: int, video_time_ms: int, ranking_point: int = 12345, timer: str = "00:10.000") -> SceneObservation:
    return SceneObservation(
        frame_index=frame_index,
        video_time_ms=video_time_ms,
        scene="result",
        result=ResultSceneResult(
            ranking_point=ranking_point,
            timer=timer,
        ),
    )


class SessionAggregatorTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = patch("kronos_analyzer.session_aggregator._load_student_names", return_value=[])
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_no_raid_observations_returns_none(self) -> None:
        summary = aggregate_session(
            [
                make_character_select(1, 1000),
                make_result(2, 2000),
            ]
        )

        self.assertIsNone(summary)

    def test_character_select_raid_result_yields_complete_single_segment_session(self) -> None:
        summary = aggregate_session(
            [
                make_character_select(1, 1000),
                make_raid(2, 2000, "01:00.000", 1000),
                make_raid(3, 3000, "00:58.000", 900),
                make_result(4, 4000),
            ]
        )

        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary.resolved_phases, ["character_select", "raid", "result"])
        self.assertEqual(summary.team_segments[0].students[0].name, "Hoshino")
        self.assertEqual(len(summary.team_segments), 1)
        self.assertEqual(summary.team_segments[0].segment_label, "team_1")
        self.assertEqual(summary.ranking_point, 12345)
        self.assertEqual(summary.warnings, [])

    def test_raid_timer_reset_creates_second_anonymous_team_segment(self) -> None:
        summary = aggregate_session(
            [
                make_raid(1, 1000, "00:50.000", 1000),
                make_raid(2, 2000, "00:48.000", 900),
                make_raid(3, 3000, "00:54.000", 600),
                make_raid(4, 4000, "00:52.000", 500),
                make_result(5, 5000),
            ]
        )

        assert summary is not None
        self.assertEqual([segment.segment_label for segment in summary.team_segments], ["team_1", "team_2"])
        self.assertEqual(summary.team_segments[0].transition_from_previous, "initial")
        self.assertEqual(summary.team_segments[1].transition_from_previous, "timer_reset")

    def test_non_raid_gap_before_timer_reset_marks_transition_with_gap(self) -> None:
        summary = aggregate_session(
            [
                make_raid(1, 1000, "00:50.000", 1000),
                make_character_select(2, 1500, "Shiroko"),
                make_raid(3, 2000, "00:54.500", 700),
                make_result(4, 3000),
            ]
        )

        assert summary is not None
        self.assertEqual(len(summary.team_segments), 2)
        self.assertEqual(summary.team_segments[1].transition_from_previous, "timer_reset_with_non_raid_gap")

    def test_small_timer_increase_is_treated_as_noise(self) -> None:
        summary = aggregate_session(
            [
                make_raid(1, 1000, "00:50.000", 1000),
                make_raid(2, 2000, "00:51.500", 950),
                make_result(3, 3000),
            ]
        )

        assert summary is not None
        self.assertEqual(len(summary.team_segments), 1)

    def test_duplicate_timer_keeps_last_valid_sample(self) -> None:
        summary = aggregate_session(
            [
                make_raid(1, 1000, "00:50.000", 1000),
                make_raid(2, 2000, "00:50.000", None, None),
                make_raid(3, 3000, "00:50.000", 980),
                make_result(4, 4000),
            ]
        )

        assert summary is not None
        samples = summary.team_segments[0].raid_timeline
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].frame_index, 3)
        self.assertEqual(samples[0].boss_remaining_hp, 980)

    def test_duplicate_timer_prefers_sample_with_cost_on_quality_tie(self) -> None:
        summary = aggregate_session(
            [
                make_raid(1, 1000, "00:50.000", 1000, 1000, None),
                make_raid(2, 2000, "00:50.000", 1000, 1000, 3),
                make_result(3, 3000),
            ]
        )

        assert summary is not None
        samples = summary.team_segments[0].raid_timeline
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].frame_index, 2)
        self.assertEqual(samples[0].cost, 3)

    def test_raid_timeline_carries_cost(self) -> None:
        summary = aggregate_session(
            [
                make_raid(1, 1000, "00:50.000", 1000, 1000, -11),
                make_result(2, 2000),
            ]
        )

        assert summary is not None
        self.assertEqual(summary.team_segments[0].raid_timeline[0].cost, -11)

    def test_segment_damage_resets_while_session_damage_stays_cumulative(self) -> None:
        summary = aggregate_session(
            [
                make_raid(1, 1000, "00:50.000", 1000),
                make_raid(2, 2000, "00:48.000", 900),
                make_raid(3, 3000, "00:54.000", 600),
                make_raid(4, 4000, "00:52.000", 500),
                make_result(5, 5000),
            ]
        )

        assert summary is not None
        team_1_samples = summary.team_segments[0].raid_timeline
        team_2_samples = summary.team_segments[1].raid_timeline

        self.assertEqual(team_1_samples[0].segment_damage, 0)
        self.assertEqual(team_1_samples[1].segment_damage, 100)
        self.assertEqual(team_2_samples[0].session_damage, 400)
        self.assertEqual(team_2_samples[0].segment_damage, 0)
        self.assertEqual(team_2_samples[1].segment_damage, 100)

    def test_missing_result_finalizes_from_raid_tail_with_warnings(self) -> None:
        summary = aggregate_session(
            [
                make_raid(1, 1000, "00:50.000", 1000),
                make_raid(2, 2000, "00:48.000", 900),
            ]
        )

        assert summary is not None
        self.assertEqual(summary.ended_at_frame_index, 2)
        self.assertIn("missing_result_scene", summary.warnings)
        self.assertIn("raid_not_finished", summary.warnings)

    def test_missing_character_select_warns_and_keeps_students_none(self) -> None:
        summary = aggregate_session(
            [
                make_raid(1, 1000, "00:50.000", 1000),
                make_result(2, 2000),
            ]
        )

        assert summary is not None
        self.assertIsNone(summary.team_segments[0].students)
        self.assertIn("missing_character_select", summary.warnings)

    def test_conflicting_total_hp_adds_warning(self) -> None:
        summary = aggregate_session(
            [
                make_raid(1, 1000, "00:50.000", 1000, 1000),
                make_raid(2, 2000, "00:48.000", 900, 1200),
                make_result(3, 3000),
            ]
        )

        assert summary is not None
        self.assertIn("inconsistent_boss_total_hp", summary.warnings)
        self.assertEqual(summary.team_segments[0].raid_timeline[1].boss_total_hp_canonical, 1000)

    def test_raid_after_result_creates_additional_segment(self) -> None:
        summary = aggregate_session(
            [
                make_raid(1, 1000, "00:50.000", 1000),
                make_result(2, 2000),
                make_raid(3, 3000, "00:50.000", 1000),
            ]
        )

        assert summary is not None
        self.assertNotIn("unsupported_additional_attempts", summary.warnings)
        self.assertEqual(len(summary.team_segments), 2)
        self.assertEqual(summary.ended_at_frame_index, 2)

    def test_two_cs_groups_produce_two_segments_with_different_rosters(self) -> None:
        summary = aggregate_session(
            [
                make_character_select(1, 1000, "Hoshino"),
                make_character_select(2, 1500, "Hoshino"),
                make_raid(3, 2000, "00:50.000", 1000),
                make_raid(4, 3000, "00:48.000", 900),
                make_result(5, 4000),
                make_character_select(6, 5000, "Shiroko"),
                make_character_select(7, 5500, "Shiroko"),
                make_raid(8, 6000, "00:50.000", 600),
                make_raid(9, 7000, "00:48.000", 500),
                make_result(10, 8000),
            ]
        )

        assert summary is not None
        self.assertEqual(len(summary.team_segments), 2)
        self.assertIsNotNone(summary.team_segments[0].students)
        self.assertEqual(summary.team_segments[0].students[0].name, "Hoshino")
        self.assertIsNotNone(summary.team_segments[1].students)
        self.assertEqual(summary.team_segments[1].students[0].name, "Shiroko")
        self.assertNotIn("missing_character_select", summary.warnings)

    def test_parse_raid_timer_ms_parses_expected_format(self) -> None:
        self.assertEqual(parse_raid_timer_ms("01:23.456"), 83_456)
        self.assertIsNone(parse_raid_timer_ms("bad"))


class AggregateStudentsTests(unittest.TestCase):
    def _patch_names(self, names: list[str]):
        return patch("kronos_analyzer.session_aggregator._load_student_names", return_value=names)

    def test_majority_vote_picks_most_common_name(self) -> None:
        observations = [
            SceneObservation(frame_index=1, video_time_ms=100, scene="character_select", students=[
                make_student("Hoshino", slot_index=1),
            ]),
            SceneObservation(frame_index=2, video_time_ms=200, scene="character_select", students=[
                make_student("Hoshino", slot_index=1),
            ]),
            SceneObservation(frame_index=3, video_time_ms=300, scene="character_select", students=[
                make_student("Hoshimo", slot_index=1),
            ]),
        ]
        with self._patch_names(["Hoshino"]):
            result = _aggregate_students(observations)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "Hoshino")

    def test_fuzzy_resolves_noisy_name_to_canonical(self) -> None:
        observations = [
            SceneObservation(frame_index=1, video_time_ms=100, scene="character_select", students=[
                make_student("Hoshimo", slot_index=1),
            ]),
        ]
        with self._patch_names(["Hoshino", "Shiroko"]):
            result = _aggregate_students(observations)

        self.assertEqual(result[0].name, "Hoshino")

    def test_multiple_slots_aggregated_independently(self) -> None:
        observations = [
            SceneObservation(frame_index=1, video_time_ms=100, scene="character_select", students=[
                make_student("Hoshino", slot_index=1),
                make_student("Shiroko", slot_index=2),
            ]),
            SceneObservation(frame_index=2, video_time_ms=200, scene="character_select", students=[
                make_student("Hoshino", slot_index=1),
                make_student("Shiroko", slot_index=2),
            ]),
        ]
        with self._patch_names(["Hoshino", "Shiroko"]):
            result = _aggregate_students(observations)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].name, "Hoshino")
        self.assertEqual(result[1].name, "Shiroko")

    def test_star_yellow_and_blue_majority_vote(self) -> None:
        observations = [
            SceneObservation(frame_index=1, video_time_ms=100, scene="character_select", students=[
                make_student("A", slot_index=1, star_yellow="5", star_blue=None),
            ]),
            SceneObservation(frame_index=2, video_time_ms=200, scene="character_select", students=[
                make_student("A", slot_index=1, star_yellow="5", star_blue=None),
            ]),
            SceneObservation(frame_index=3, video_time_ms=300, scene="character_select", students=[
                make_student("A", slot_index=1, star_yellow=None, star_blue="3"),
            ]),
        ]
        with self._patch_names([]):
            result = _aggregate_students(observations)

        self.assertEqual(result[0].star_yellow, "5")
        self.assertEqual(result[0].star_blue, "3")

    def test_empty_observations_returns_empty(self) -> None:
        with self._patch_names([]):
            result = _aggregate_students([])
        self.assertEqual(result, [])

    def test_level_majority_vote(self) -> None:
        observations = [
            SceneObservation(frame_index=1, video_time_ms=100, scene="character_select", students=[
                make_student("A", slot_index=1, level="90"),
            ]),
            SceneObservation(frame_index=2, video_time_ms=200, scene="character_select", students=[
                make_student("A", slot_index=1, level="90"),
            ]),
            SceneObservation(frame_index=3, video_time_ms=300, scene="character_select", students=[
                make_student("A", slot_index=1, level="80"),
            ]),
        ]
        with self._patch_names([]):
            result = _aggregate_students(observations)

        self.assertEqual(result[0].level, "90")


class ResolveNameTests(unittest.TestCase):
    def test_exact_match(self) -> None:
        self.assertEqual(_resolve_name("Hoshino", ["Hoshino", "Shiroko"]), "Hoshino")

    def test_fuzzy_match_above_threshold(self) -> None:
        self.assertEqual(_resolve_name("Hoshimo", ["Hoshino", "Shiroko"]), "Hoshino")

    def test_no_match_below_threshold(self) -> None:
        self.assertEqual(_resolve_name("XYZABC", ["Hoshino", "Shiroko"]), "XYZABC")

    def test_empty_known_names_returns_raw(self) -> None:
        self.assertEqual(_resolve_name("Hoshino", []), "Hoshino")

    def test_empty_raw_name_returns_empty(self) -> None:
        self.assertEqual(_resolve_name("", ["Hoshino"]), "")


if __name__ == "__main__":
    unittest.main()
