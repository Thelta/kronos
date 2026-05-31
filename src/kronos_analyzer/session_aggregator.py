from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import logging
from typing import Literal

from .character_select import StudentResult
from .student_names import load_student_names, resolve_name
from .raid import RaidResult
from .result import ResultSceneResult

SceneName = Literal["character_select", "raid", "result"]
TransitionKind = Literal["initial", "timer_reset", "timer_reset_with_non_raid_gap"]

TEAM_RESET_THRESHOLD_MS = 3000
_FUZZY_MATCH_THRESHOLD = 70
logger = logging.getLogger(__name__)

_cached_student_names: list[str] | None = None


@dataclass(frozen=True)
class SceneObservation:
    frame_index: int
    video_time_ms: int
    scene: SceneName
    students: list[StudentResult] | None = None
    raid: RaidResult | None = None
    result: ResultSceneResult | None = None


@dataclass
class RaidTimelineSample:
    frame_index: int
    video_time_ms: int
    raid_timer_text: str
    raid_timer_ms: int | None
    raid_elapsed_ms: int | None
    boss_remaining_hp: int | None
    boss_total_hp_observed: int | None
    boss_total_hp_canonical: int | None
    cost: int | None
    session_damage: int | None
    segment_damage: int | None


@dataclass
class TeamSegmentSummary:
    segment_index: int
    segment_label: str
    identity: None
    identity_source: None
    started_at_frame_index: int
    started_at_video_time_ms: int
    ended_at_frame_index: int
    ended_at_video_time_ms: int
    raid_start_timer_text: str | None
    raid_end_timer_text: str | None
    raid_timeline: list[RaidTimelineSample]
    best_segment_damage: int | None
    final_segment_damage: int | None
    transition_from_previous: TransitionKind
    students: list[StudentResult] | None = None


@dataclass
class SessionSummary:
    started_at_frame_index: int
    started_at_video_time_ms: int
    ended_at_frame_index: int
    ended_at_video_time_ms: int
    resolved_phases: list[SceneName]
    team_segments: list[TeamSegmentSummary]
    session_best_damage: int | None
    session_final_damage: int | None
    ranking_point: int | None
    result_timer_text: str | None
    result_video_time_ms: int | None
    warnings: list[str]


def _load_student_names() -> list[str]:
    global _cached_student_names
    if _cached_student_names is not None:
        return _cached_student_names
    try:
        with urllib.request.urlopen("https://schaledb.com/data/jp/students.json") as resp:
            data = json.loads(resp.read().decode("utf-8"))
        names = [entry["Name"] for entry in data.values() if "Name" in entry]
        _cached_student_names = names
        logger.info("Loaded %d student names from schaledb", len(names))
    except Exception:
        logger.warning("Failed to fetch student names from schaledb, falling back to raw names")
        _cached_student_names = []
    return _cached_student_names


def _resolve_name(raw_name: str, known_names: list[str]) -> str:
    if not raw_name or not known_names:
        return raw_name
    result = rfprocess.extractOne(raw_name, known_names)
    if result is None:
        return raw_name
    match_name, score, _ = result
    if score >= _FUZZY_MATCH_THRESHOLD:
        return match_name
    return raw_name


def _aggregate_students(observations: list[SceneObservation]) -> list[StudentResult]:
    cs_observations = [obs for obs in observations if obs.scene == "character_select" and obs.students]
    if not cs_observations:
        return []

    known_names = _load_student_names()

    # Collect per-slot data across all frames
    slot_data: dict[int, list[StudentResult]] = {}
    for obs in cs_observations:
        for student in obs.students:
            resolved = StudentResult(
                slot_index=student.slot_index,
                name=_resolve_name(student.name, known_names),
                level=student.level,
                star_yellow=student.star_yellow,
                star_blue=student.star_blue,
            )
            slot_data.setdefault(student.slot_index, []).append(resolved)

    # Majority vote per slot
    aggregated: list[StudentResult] = []
    for slot_index in sorted(slot_data):
        entries = slot_data[slot_index]

        name_counts: Counter[str] = Counter()
        level_counts: Counter[str] = Counter()
        yellow_counts: Counter[str] = Counter()
        blue_counts: Counter[str] = Counter()

        for entry in entries:
            if entry.name:
                name_counts[entry.name] += 1
            if entry.level:
                level_counts[entry.level] += 1
            if entry.star_yellow:
                yellow_counts[entry.star_yellow] += 1
            if entry.star_blue:
                blue_counts[entry.star_blue] += 1

        best_name = name_counts.most_common(1)[0][0] if name_counts else ""
        best_level = level_counts.most_common(1)[0][0] if level_counts else ""
        best_yellow = yellow_counts.most_common(1)[0][0] if yellow_counts else None
        best_blue = blue_counts.most_common(1)[0][0] if blue_counts else None

        aggregated.append(StudentResult(
            slot_index=slot_index,
            name=best_name,
            level=best_level,
            star_yellow=best_yellow,
            star_blue=best_blue,
        ))

    return aggregated


def aggregate_session(observations: list[SceneObservation]) -> SessionSummary | None:
    ordered = sorted(observations, key=lambda item: item.frame_index)

    raid_started = False
    result_observation: SceneObservation | None = None
    pending_new_cycle = False
    pending_non_raid_gap = False
    warnings: list[str] = []
    canonical_total_hp: int | None = None
    first_raid_video_time_ms: int | None = None

    team_segments: list[TeamSegmentSummary] = []
    current_segment: TeamSegmentSummary | None = None
    first_character_select_before_raid: SceneObservation | None = None
    first_raid_observation: SceneObservation | None = None

    # Per-segment character_select grouping
    pending_cs_group: list[SceneObservation] = []
    closed_cs_group: list[SceneObservation] | None = None

    for observation in ordered:
        if observation.scene == "character_select":
            pending_cs_group.append(observation)
            if not raid_started:
                if first_character_select_before_raid is None:
                    first_character_select_before_raid = observation
            elif raid_started:
                pending_non_raid_gap = True
            continue

        # Non-character_select scene: close the pending cs group if any
        if pending_cs_group:
            closed_cs_group = pending_cs_group
            pending_cs_group = []

        if observation.scene == "result":
            if raid_started:
                result_observation = observation
                pending_new_cycle = True
            continue

        if observation.scene != "raid":
            if raid_started:
                pending_non_raid_gap = True
            continue

        if observation.raid is None:
            continue

        if not raid_started:
            raid_started = True
            first_raid_observation = observation
            first_raid_video_time_ms = observation.video_time_ms
            current_segment = _start_segment(
                segment_index=1,
                observation=observation,
                transition="initial",
            )
            _attach_students(current_segment, closed_cs_group)
            closed_cs_group = None
            team_segments.append(current_segment)

        new_sample = _build_raid_sample(
            observation=observation,
            first_raid_video_time_ms=first_raid_video_time_ms,
        )

        if canonical_total_hp is None and new_sample.boss_total_hp_observed is not None:
            canonical_total_hp = new_sample.boss_total_hp_observed

        if (
            canonical_total_hp is not None
            and new_sample.boss_total_hp_observed is not None
            and new_sample.boss_total_hp_observed != canonical_total_hp
        ):
            _add_warning(warnings, "inconsistent_boss_total_hp")

        if pending_new_cycle or _should_start_new_segment(current_segment, new_sample):
            previous_segment = current_segment
            if pending_new_cycle:
                transition: TransitionKind = "timer_reset_with_non_raid_gap"
            elif pending_non_raid_gap:
                transition = "timer_reset_with_non_raid_gap"
            else:
                transition = "timer_reset"
            current_segment = _start_segment(
                segment_index=len(team_segments) + 1,
                observation=observation,
                transition=transition,
            )
            _attach_students(current_segment, closed_cs_group)
            closed_cs_group = None
            team_segments.append(current_segment)
            logger.info(
                "Detected team reset: new_segment=%s at frame=%d time_ms=%d previous_timer=%s new_timer=%s transition=%s",
                current_segment.segment_label,
                observation.frame_index,
                observation.video_time_ms,
                previous_segment.raid_timeline[-1].raid_timer_text if previous_segment is not None and previous_segment.raid_timeline else "",
                new_sample.raid_timer_text,
                current_segment.transition_from_previous,
            )
            pending_new_cycle = False

        pending_non_raid_gap = False
        _append_or_replace_sample(current_segment, new_sample)

    if not team_segments:
        return None

    has_any_students = any(seg.students is not None for seg in team_segments)
    if not has_any_students:
        _add_warning(warnings, "missing_character_select")

    if result_observation is None:
        _add_warning(warnings, "missing_result_scene")
        _add_warning(warnings, "raid_not_finished")
        last_sample = team_segments[-1].raid_timeline[-1]
        logger.info(
            "Finalizing session from raid tail at frame=%d time_ms=%d due to missing result scene",
            last_sample.frame_index,
            last_sample.video_time_ms,
        )
    else:
        logger.info(
            "Finalizing session from result scene at frame=%d time_ms=%d",
            result_observation.frame_index,
            result_observation.video_time_ms,
        )

    _finalize_segments(team_segments, canonical_total_hp)

    started_at = first_character_select_before_raid or first_raid_observation
    if started_at is None:
        return None

    if result_observation is not None:
        ended_at_frame_index = result_observation.frame_index
        ended_at_video_time_ms = result_observation.video_time_ms
    else:
        last_sample = team_segments[-1].raid_timeline[-1]
        ended_at_frame_index = last_sample.frame_index
        ended_at_video_time_ms = last_sample.video_time_ms

    resolved_phases: list[SceneName] = []
    if first_character_select_before_raid is not None:
        resolved_phases.append("character_select")
    resolved_phases.append("raid")
    if result_observation is not None:
        resolved_phases.append("result")

    return SessionSummary(
        started_at_frame_index=started_at.frame_index,
        started_at_video_time_ms=started_at.video_time_ms,
        ended_at_frame_index=ended_at_frame_index,
        ended_at_video_time_ms=ended_at_video_time_ms,
        resolved_phases=resolved_phases,
        team_segments=team_segments,
        session_best_damage=_max_or_none(
            sample.session_damage
            for segment in team_segments
            for sample in segment.raid_timeline
        ),
        session_final_damage=_last_non_none(
            sample.session_damage
            for segment in team_segments
            for sample in segment.raid_timeline
        ),
        ranking_point=result_observation.result.ranking_point if result_observation is not None and result_observation.result is not None else None,
        result_timer_text=result_observation.result.timer if result_observation is not None and result_observation.result is not None else None,
        result_video_time_ms=result_observation.video_time_ms if result_observation is not None else None,
        warnings=warnings,
    )


def parse_raid_timer_ms(text: str) -> int | None:
    if not text or ":" not in text or "." not in text:
        return None
    try:
        minutes_text, rest = text.split(":", maxsplit=1)
        seconds_text, millis_text = rest.split(".", maxsplit=1)
        minutes = int(minutes_text)
        seconds = int(seconds_text)
        millis = int(millis_text)
    except ValueError:
        return None
    if seconds < 0 or seconds >= 60 or millis < 0 or millis >= 1000:
        return None
    return (minutes * 60_000) + (seconds * 1000) + millis


def _attach_students(
    segment: TeamSegmentSummary,
    cs_group: list[SceneObservation] | None,
) -> None:
    if not cs_group:
        return
    students = _aggregate_students(cs_group)
    segment.students = students if students else None


def _start_segment(
    *,
    segment_index: int,
    observation: SceneObservation,
    transition: TransitionKind,
) -> TeamSegmentSummary:
    return TeamSegmentSummary(
        segment_index=segment_index,
        segment_label=f"team_{segment_index}",
        identity=None,
        identity_source=None,
        started_at_frame_index=observation.frame_index,
        started_at_video_time_ms=observation.video_time_ms,
        ended_at_frame_index=observation.frame_index,
        ended_at_video_time_ms=observation.video_time_ms,
        raid_start_timer_text=None,
        raid_end_timer_text=None,
        raid_timeline=[],
        best_segment_damage=None,
        final_segment_damage=None,
        transition_from_previous=transition,
    )


def _build_raid_sample(
    *,
    observation: SceneObservation,
    first_raid_video_time_ms: int | None,
) -> RaidTimelineSample:
    raid = observation.raid
    timer_text = raid.timer if raid is not None else ""
    timer_ms = parse_raid_timer_ms(timer_text)
    raid_elapsed_ms = None
    if first_raid_video_time_ms is not None:
        raid_elapsed_ms = observation.video_time_ms - first_raid_video_time_ms
    return RaidTimelineSample(
        frame_index=observation.frame_index,
        video_time_ms=observation.video_time_ms,
        raid_timer_text=timer_text,
        raid_timer_ms=timer_ms,
        raid_elapsed_ms=raid_elapsed_ms,
        boss_remaining_hp=raid.boss_remaining_hp if raid is not None else None,
        boss_total_hp_observed=raid.boss_total_hp if raid is not None else None,
        boss_total_hp_canonical=None,
        cost=raid.cost if raid is not None else None,
        session_damage=None,
        segment_damage=None,
    )


def _should_start_new_segment(
    segment: TeamSegmentSummary | None,
    new_sample: RaidTimelineSample,
) -> bool:
    if segment is None or not segment.raid_timeline:
        return False
    previous_sample = segment.raid_timeline[-1]
    if previous_sample.raid_timer_ms is None or new_sample.raid_timer_ms is None:
        return False
    return (new_sample.raid_timer_ms - previous_sample.raid_timer_ms) >= TEAM_RESET_THRESHOLD_MS


def _append_or_replace_sample(
    segment: TeamSegmentSummary,
    new_sample: RaidTimelineSample,
) -> None:
    if segment.raid_timeline and segment.raid_timeline[-1].raid_timer_text == new_sample.raid_timer_text:
        previous_sample = segment.raid_timeline[-1]
        if _should_replace_duplicate(previous_sample, new_sample):
            segment.raid_timeline[-1] = new_sample
            segment.ended_at_frame_index = new_sample.frame_index
            segment.ended_at_video_time_ms = new_sample.video_time_ms
        return

    segment.raid_timeline.append(new_sample)
    if segment.raid_start_timer_text is None:
        segment.raid_start_timer_text = new_sample.raid_timer_text or None
    segment.raid_end_timer_text = new_sample.raid_timer_text or segment.raid_end_timer_text
    segment.ended_at_frame_index = new_sample.frame_index
    segment.ended_at_video_time_ms = new_sample.video_time_ms


def _should_replace_duplicate(previous_sample: RaidTimelineSample, new_sample: RaidTimelineSample) -> bool:
    previous_score = _sample_quality_score(previous_sample)
    new_score = _sample_quality_score(new_sample)
    return new_score >= previous_score


def _sample_quality_score(sample: RaidTimelineSample) -> tuple[int, int, int, int]:
    has_timer = int(sample.raid_timer_ms is not None)
    has_any_hp = int(sample.boss_remaining_hp is not None or sample.boss_total_hp_observed is not None)
    has_both_hp = int(sample.boss_remaining_hp is not None and sample.boss_total_hp_observed is not None)
    has_cost = int(sample.cost is not None)
    return has_timer, has_any_hp, has_both_hp, has_cost


def _finalize_segments(
    team_segments: list[TeamSegmentSummary],
    canonical_total_hp: int | None,
) -> None:
    for segment in team_segments:
        for sample in segment.raid_timeline:
            sample.boss_total_hp_canonical = canonical_total_hp
            if canonical_total_hp is not None and sample.boss_remaining_hp is not None:
                sample.session_damage = canonical_total_hp - sample.boss_remaining_hp

        segment_baseline_damage: int | None = None
        for sample in segment.raid_timeline:
            if sample.session_damage is None:
                continue
            if segment_baseline_damage is None:
                segment_baseline_damage = sample.session_damage
            sample.segment_damage = sample.session_damage - segment_baseline_damage

        segment.best_segment_damage = _max_or_none(sample.segment_damage for sample in segment.raid_timeline)
        segment.final_segment_damage = _last_non_none(sample.segment_damage for sample in segment.raid_timeline)
        if segment.raid_start_timer_text is None and segment.raid_timeline:
            segment.raid_start_timer_text = segment.raid_timeline[0].raid_timer_text or None
        if segment.raid_timeline:
            segment.raid_end_timer_text = segment.raid_timeline[-1].raid_timer_text or segment.raid_end_timer_text


def _add_warning(warnings: list[str], warning: str) -> None:
    if warning not in warnings:
        warnings.append(warning)


def _last_non_none(values) -> int | None:
    last_value: int | None = None
    for value in values:
        if value is not None:
            last_value = value
    return last_value


def _max_or_none(values) -> int | None:
    filtered = [value for value in values if value is not None]
    return max(filtered) if filtered else None
