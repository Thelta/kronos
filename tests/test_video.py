from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kronos_analyzer.character_select import StudentResult  # noqa: E402
from kronos_analyzer.cli import detect_scene_details, format_scene_detection_reason, handle_analyze_video  # noqa: E402
from kronos_analyzer.ocr_engine import OCRRun  # noqa: E402
from kronos_analyzer.raid import RaidResult  # noqa: E402
from kronos_analyzer.result import ResultSceneResult  # noqa: E402
from kronos_analyzer.schemas import OCRDump, OCRLine  # noqa: E402
from kronos_analyzer.session_aggregator import SceneObservation, SessionSummary, TeamSegmentSummary  # noqa: E402
from kronos_analyzer.video import (  # noqa: E402
    ANALYSIS_FRAME_INTERVAL_MS,
    PyAVVideoSource,
    VideoAnalysisEvent,
    VideoAnalysisResult,
    VideoFrame,
    analyze_video,
    build_video_analysis_event,
    build_scene_observation,
)


class FakeFrame:
    def __init__(self, *, pts, time_base, array):
        self.pts = pts
        self.time_base = time_base
        self._array = array

    def to_ndarray(self, format: str):
        if format != "bgr24":
            raise AssertionError(f"Unexpected format: {format}")
        return self._array


class FakeContainer:
    def __init__(self, *, frames, average_rate=30, has_video_stream=True):
        self._frames = frames
        video_streams = [SimpleNamespace(average_rate=average_rate)] if has_video_stream else []
        self.streams = SimpleNamespace(video=video_streams)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def decode(self, video: int):
        if video != 0:
            raise AssertionError(f"Unexpected video stream index: {video}")
        yield from self._frames


class FakeEngine:
    def __init__(self, dumps: list[OCRDump] | None = None):
        self._dumps = dumps or []
        self.calls: list[tuple[str, object]] = []

    def run(self, image_array, image_name: str, *, model_preset=None) -> OCRRun:
        self.calls.append(("run", image_name))
        dump = self._dumps.pop(0)
        return OCRRun(dump=dump, _write_visualization=lambda output_path: None)

    def detect_text(self, image_array, *, model_preset=None):
        self.calls.append(("detect_text", None))
        return []

    def recognize_crops(self, crops, profile: str = "default", *, model_preset=None) -> list[OCRLine]:
        self.calls.append(("recognize_crops", profile))
        return [OCRLine(text="5", score=0.95, box=[])]


class FailingEngine(FakeEngine):
    def run(self, image_array, image_name: str, *, model_preset=None) -> OCRRun:
        raise ValueError("boom")


class FakeVideoSource:
    def __init__(self, frames: list[VideoFrame]):
        self._frames = frames

    def iter_frames(self, video_path: Path):
        yield from self._frames


def make_dump(text: str) -> OCRDump:
    return OCRDump(
        image="frame.png",
        line_count=1,
        combined_text=text,
        lines=[OCRLine(text=text, score=0.9, box=[[1.0, 2.0], [3.0, 4.0]])],
    )


class VideoPipelineTests(unittest.TestCase):
    def test_pyav_iter_frames_uses_pts_and_time_base(self) -> None:
        frames = [
            FakeFrame(pts=0, time_base=Fraction(1, 1000), array=np.zeros((2, 2, 3), dtype=np.uint8)),
            FakeFrame(pts=40, time_base=Fraction(1, 1000), array=np.ones((2, 2, 3), dtype=np.uint8)),
        ]
        source = PyAVVideoSource(opener=lambda path: FakeContainer(frames=frames))

        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "sample.mp4"
            video_path.write_bytes(b"video")
            decoded = list(source.iter_frames(video_path))

        self.assertEqual([frame.frame_index for frame in decoded], [0, 1])
        self.assertEqual([frame.video_time_ms for frame in decoded], [0, 40])
        self.assertEqual(decoded[0].image_name, "sample.frame_000000.png")

    def test_pyav_iter_frames_falls_back_when_later_pts_missing(self) -> None:
        frames = [
            FakeFrame(pts=0, time_base=Fraction(1, 1000), array=np.zeros((2, 2, 3), dtype=np.uint8)),
            FakeFrame(pts=None, time_base=Fraction(1, 1000), array=np.ones((2, 2, 3), dtype=np.uint8)),
        ]
        source = PyAVVideoSource(opener=lambda path: FakeContainer(frames=frames, average_rate=25))

        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "sample.mp4"
            video_path.write_bytes(b"video")
            decoded = list(source.iter_frames(video_path))

        self.assertEqual([frame.video_time_ms for frame in decoded], [0, 40])

    def test_pyav_iter_frames_raises_without_recoverable_pts_fallback(self) -> None:
        frames = [
            FakeFrame(pts=0, time_base=Fraction(1, 1000), array=np.zeros((2, 2, 3), dtype=np.uint8)),
            FakeFrame(pts=None, time_base=Fraction(1, 1000), array=np.ones((2, 2, 3), dtype=np.uint8)),
        ]
        source = PyAVVideoSource(opener=lambda path: FakeContainer(frames=frames, average_rate=None))

        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "sample.mp4"
            video_path.write_bytes(b"video")
            with self.assertRaises(ValueError):
                list(source.iter_frames(video_path))

    def test_pyav_iter_frames_raises_when_no_video_stream_exists(self) -> None:
        source = PyAVVideoSource(opener=lambda path: FakeContainer(frames=[], has_video_stream=False))

        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "sample.mp4"
            video_path.write_bytes(b"video")
            with self.assertRaises(ValueError):
                list(source.iter_frames(video_path))

    def test_build_scene_observation_for_supported_scenes(self) -> None:
        image = np.zeros((40, 40, 3), dtype=np.uint8)
        engine = FakeEngine()

        with patch(
            "kronos_analyzer.video.extract_students",
            return_value=([StudentResult(slot_index=1, name="Hoshino", level="90", star_yellow="5", star_blue=None)], []),
        ):
            character_dump = make_dump("総力戦編成 striker special 開始スキル 部隊情報")
            character_observation = build_scene_observation(
                frame_index=0,
                video_time_ms=0,
                dump=character_dump,
                image_array=image,
                engine=engine,
                image_stem="character",
                detected_scene="character_select",
            )
            self.assertEqual(character_observation.scene, "character_select")
            self.assertEqual(character_observation.students[0].name, "Hoshino")

        raid_dump = OCRDump(
            image="frame.png",
            line_count=4,
            combined_text="03:24.867\n68,633,666/70,000,000\nCOST\n-11",
            lines=[
                OCRLine(text="03:24.867", score=0.99, box=[[2154.0, 66.0], [2357.0, 66.0], [2357.0, 115.0], [2154.0, 115.0]]),
                OCRLine(text="68,633,666/70,000,000", score=0.99, box=[[1161.0, 106.0], [1469.0, 106.0], [1469.0, 141.0], [1161.0, 141.0]]),
                OCRLine(text="COST", score=0.99, box=[[1552.0, 1279.0], [1625.0, 1279.0], [1625.0, 1316.0], [1552.0, 1316.0]]),
                OCRLine(text="-11", score=0.99, box=[[1555.0, 1314.0], [1620.0, 1314.0], [1620.0, 1365.0], [1555.0, 1365.0]]),
            ],
        )
        raid_observation = build_scene_observation(
            frame_index=1,
            video_time_ms=1000,
            dump=raid_dump,
            image_array=image,
            engine=engine,
            image_stem="raid",
            detected_scene="raid",
        )
        self.assertEqual(raid_observation.scene, "raid")
        self.assertIsNotNone(raid_observation.raid)
        self.assertEqual(raid_observation.raid.cost, -11)
        self.assertFalse(raid_observation.raid.brightness_recovery_triggered)

        result_dump = make_dump("battle complete rankingpoint 戦闘時間 00:10.000")
        result_observation = build_scene_observation(
            frame_index=2,
            video_time_ms=2000,
            dump=result_dump,
            image_array=image,
            engine=engine,
            image_stem="result",
            detected_scene="result",
        )
        self.assertEqual(result_observation.scene, "result")
        self.assertIsNotNone(result_observation.result)

    def test_build_scene_observation_returns_none_for_unresolved_frame(self) -> None:
        observation = build_scene_observation(
            frame_index=1,
            video_time_ms=1000,
            dump=make_dump("unrelated text"),
            image_array=np.zeros((20, 20, 3), dtype=np.uint8),
            engine=FakeEngine(),
            image_stem="none",
        )

        self.assertIsNone(observation)

    def test_detect_scene_details_reports_matched_keywords(self) -> None:
        detection = detect_scene_details(make_dump("battle boss total damage pause"))

        self.assertIsNone(detection.scene)
        self.assertEqual(detection.failure_reason, "disabled")
        self.assertEqual(detection.matched_keywords, {})
        self.assertIn("reason=disabled", format_scene_detection_reason(detection))

    def test_analyze_video_logs_scene_change_reason(self) -> None:
        frames = [
            VideoFrame(
                frame_index=0,
                video_time_ms=0,
                image_name="raid.frame_000000.png",
                image_array=np.zeros((4, 4, 3), dtype=np.uint8),
            ),
            VideoFrame(
                frame_index=1,
                video_time_ms=ANALYSIS_FRAME_INTERVAL_MS,
                image_name="raid.frame_000001.png",
                image_array=np.zeros((4, 4, 3), dtype=np.uint8),
            ),
        ]
        engine = FakeEngine(
            dumps=[
                make_dump("battle boss total damage"),
                make_dump("unrelated text"),
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "raid.webm"
            video_path.write_bytes(b"video")
            with self.assertLogs("kronos_analyzer.video", level="INFO") as captured:
                with patch(
                    "kronos_analyzer.cli.detect_scene_details",
                    side_effect=[
                        SimpleNamespace(scene="raid", matched_keywords={}, matched_scenes=["raid"], failure_reason=None),
                        SimpleNamespace(scene=None, matched_keywords={}, matched_scenes=[], failure_reason="disabled"),
                        SimpleNamespace(scene=None, matched_keywords={}, matched_scenes=[], failure_reason="disabled"),
                    ],
                ):
                    analyze_video(
                        video_path=video_path,
                        engine=engine,
                        video_source=FakeVideoSource(frames),
                    )

        joined = "\n".join(captured.output)
        self.assertIn("Scene changed at frame=0", joined)
        self.assertIn("from=none to=raid", joined)
        self.assertIn("from=raid to=none", joined)
        self.assertIn("reason=disabled", joined)
        self.assertIn("text_excerpt=", joined)

    def test_analyze_video_wraps_frame_context_on_processing_failure(self) -> None:
        frames = [
            VideoFrame(
                frame_index=419,
                video_time_ms=16760,
                image_name="raid.frame_000419.png",
                image_array=np.zeros((4, 4, 3), dtype=np.uint8),
            )
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "raid.webm"
            video_path.write_bytes(b"video")

            with self.assertRaisesRegex(RuntimeError, r"frame=419.*raid\.frame_000419\.png"):
                analyze_video(
                    video_path=video_path,
                    engine=FailingEngine(),
                    video_source=FakeVideoSource(frames),
                )

    def test_analyze_video_samples_at_five_fps_before_ocr(self) -> None:
        frames = [
            VideoFrame(
                frame_index=0,
                video_time_ms=0,
                image_name="raid.frame_000000.png",
                image_array=np.zeros((4, 4, 3), dtype=np.uint8),
            ),
            VideoFrame(
                frame_index=1,
                video_time_ms=100,
                image_name="raid.frame_000001.png",
                image_array=np.zeros((4, 4, 3), dtype=np.uint8),
            ),
            VideoFrame(
                frame_index=2,
                video_time_ms=ANALYSIS_FRAME_INTERVAL_MS,
                image_name="raid.frame_000002.png",
                image_array=np.zeros((4, 4, 3), dtype=np.uint8),
            ),
        ]
        engine = FakeEngine(
            dumps=[
                make_dump("battle boss total damage"),
                make_dump("battle complete rankingpoint 戦闘時間 00:10.000"),
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "raid.webm"
            video_path.write_bytes(b"video")
            with patch(
                "kronos_analyzer.cli.detect_scene_details",
                side_effect=[
                    SimpleNamespace(scene="raid", matched_keywords={}, matched_scenes=["raid"], failure_reason=None),
                    SimpleNamespace(scene="result", matched_keywords={}, matched_scenes=["result"], failure_reason=None),
                ],
            ):
                result = analyze_video(
                    video_path=video_path,
                    engine=engine,
                    video_source=FakeVideoSource(frames),
                )

        self.assertEqual(
            engine.calls,
            [
                ("run", "raid.frame_000000.png"),
                ("run", "raid.frame_000002.png"),
            ],
        )
        self.assertEqual([event.frame_index for event in result.events], [0, 2])

    def test_handle_analyze_video_writes_session_and_events_without_per_frame_artifacts(self) -> None:
        session = SessionSummary(
            started_at_frame_index=0,
            started_at_video_time_ms=0,
            ended_at_frame_index=1,
            ended_at_video_time_ms=1000,
            resolved_phases=["raid", "result"],
            team_segments=[
                TeamSegmentSummary(
                    segment_index=1,
                    segment_label="team_1",
                    identity=None,
                    identity_source=None,
                    started_at_frame_index=0,
                    started_at_video_time_ms=0,
                    ended_at_frame_index=1,
                    ended_at_video_time_ms=1000,
                    raid_start_timer_text="00:50.000",
                    raid_end_timer_text="00:48.000",
                    raid_timeline=[],
                    best_segment_damage=100,
                    final_segment_damage=100,
                    transition_from_previous="initial",
                )
            ],
            session_best_damage=100,
            session_final_damage=100,
            ranking_point=12345,
            result_timer_text="00:10.000",
            result_video_time_ms=1000,
            warnings=[],
        )
        events = [
            VideoAnalysisEvent(
                frame_index=0,
                video_time_ms=0,
                scene="raid",
                scene_payload={"boss_remaining_hp": 1000, "boss_total_hp": 1000, "timer": "00:50.000", "cost": 3},
                combined_text="battle boss",
            )
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "raid.mp4"
            input_path.write_bytes(b"video")
            output_dir = temp_path / "out"
            args = argparse.Namespace(
                input=str(input_path),
                output=str(output_dir),
            )

            with (
                patch("kronos_analyzer.cli.create_ocr_engine", return_value=FakeEngine()),
                patch(
                    "kronos_analyzer.cli.analyze_video",
                    return_value=VideoAnalysisResult(session=session, events=events),
                ),
            ):
                exit_code = handle_analyze_video(args)

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "raid.session.json").exists())
            self.assertTrue((output_dir / "raid.events.json").exists())
            self.assertFalse((output_dir / "raid_01.json").exists())

            session_payload = json.loads((output_dir / "raid.session.json").read_text(encoding="utf-8"))
            events_payload = json.loads((output_dir / "raid.events.json").read_text(encoding="utf-8"))
            self.assertEqual(session_payload["ranking_point"], 12345)
            self.assertEqual(events_payload[0]["scene"], "raid")
            self.assertEqual(events_payload[0]["scene_payload"]["cost"], 3)

    def test_build_video_analysis_event_includes_raid_cost(self) -> None:
        observation = SceneObservation(
            frame_index=1,
            video_time_ms=1000,
            scene="raid",
            raid=RaidResult(
                boss_remaining_hp=1000,
                boss_total_hp=2000,
                timer="00:50.000",
                cost=11,
                brightness_recovery_triggered=True,
            ),
        )

        event = build_video_analysis_event(
            observation=observation,
            combined_text="battle boss",
        )

        self.assertEqual(event.scene_payload["cost"], 11)
        self.assertTrue(event.scene_payload["brightness_recovery_triggered"])

    def test_video_pipeline_emits_brightness_recovery_trigger_on_first_bright_raid_frame(self) -> None:
        frames = [
            VideoFrame(
                frame_index=0,
                video_time_ms=0,
                image_name="raid.frame_000000.png",
                image_array=np.full((8, 8, 3), 60, dtype=np.uint8),
            ),
            VideoFrame(
                frame_index=1,
                video_time_ms=ANALYSIS_FRAME_INTERVAL_MS,
                image_name="raid.frame_000001.png",
                image_array=np.full((8, 8, 3), 120, dtype=np.uint8),
            ),
            VideoFrame(
                frame_index=2,
                video_time_ms=ANALYSIS_FRAME_INTERVAL_MS * 2,
                image_name="raid.frame_000002.png",
                image_array=np.full((8, 8, 3), 130, dtype=np.uint8),
            ),
        ]
        source = SimpleNamespace(iter_frames=lambda video_path: iter(frames))
        engine = FakeEngine(
            dumps=[
                make_dump("battle boss total damage 00:50.000"),
                make_dump("battle boss total damage 00:48.000"),
                make_dump("battle boss total damage 00:46.000"),
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "raid.mp4"
            video_path.write_bytes(b"video")
            with patch(
                "kronos_analyzer.cli.detect_scene_details",
                side_effect=[
                    SimpleNamespace(scene="raid", matched_keywords={}, matched_scenes=["raid"], failure_reason=None),
                    SimpleNamespace(scene="raid", matched_keywords={}, matched_scenes=["raid"], failure_reason=None),
                    SimpleNamespace(scene="raid", matched_keywords={}, matched_scenes=["raid"], failure_reason=None),
                ],
            ):
                result = analyze_video(video_path=video_path, engine=engine, video_source=source)

        self.assertEqual(len(result.events), 3)
        self.assertFalse(result.events[0].scene_payload["brightness_recovery_triggered"])
        self.assertTrue(result.events[1].scene_payload["brightness_recovery_triggered"])
        self.assertFalse(result.events[2].scene_payload["brightness_recovery_triggered"])

    def test_video_pipeline_with_fake_source_feeds_session_aggregator(self) -> None:
        frames = [
            VideoFrame(frame_index=0, video_time_ms=0, image_name="raid.frame_000000.png", image_array=np.zeros((8, 8, 3), dtype=np.uint8)),
            VideoFrame(frame_index=1, video_time_ms=1000, image_name="raid.frame_000001.png", image_array=np.zeros((8, 8, 3), dtype=np.uint8)),
            VideoFrame(frame_index=2, video_time_ms=2000, image_name="raid.frame_000002.png", image_array=np.zeros((8, 8, 3), dtype=np.uint8)),
        ]
        source = SimpleNamespace(iter_frames=lambda video_path: iter(frames))
        engine = FakeEngine(
            dumps=[
                make_dump("battle boss total damage 00:50.000"),
                make_dump("battle boss total damage 00:48.000"),
                make_dump("battle complete rankingpoint 戦闘時間 00:10.000"),
            ]
        )

        from kronos_analyzer.video import analyze_video

        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "raid.mp4"
            video_path.write_bytes(b"video")
            with patch(
                "kronos_analyzer.cli.detect_scene_details",
                side_effect=[
                    SimpleNamespace(scene="raid", matched_keywords={}, matched_scenes=["raid"], failure_reason=None),
                    SimpleNamespace(scene="raid", matched_keywords={}, matched_scenes=["raid"], failure_reason=None),
                    SimpleNamespace(scene="result", matched_keywords={}, matched_scenes=["result"], failure_reason=None),
                ],
            ):
                result = analyze_video(video_path=video_path, engine=engine, video_source=source)

        self.assertIsNotNone(result.session)
        assert result.session is not None
        self.assertEqual(result.session.resolved_phases, ["raid", "result"])
        self.assertGreaterEqual(len(result.events), 2)
        self.assertEqual(engine.calls[0], ("run", "raid.frame_000000.png"))


if __name__ == "__main__":
    unittest.main()
