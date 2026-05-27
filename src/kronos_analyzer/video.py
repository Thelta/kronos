from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import logging
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

from .character_select import extract_students
from .config import CHARACTER_SELECT_CONFIG
from .ocr_engine import OCREngine, OCRModelPreset
from .raid import extract_raid_fields
from .raid_tracker import RaidTracker
from .result import extract_result_fields
from .schemas import OCRDump
from .session_aggregator import SceneObservation, SessionSummary, aggregate_session

SceneName = Literal["character_select", "raid", "result"]
logger = logging.getLogger(__name__)
PROGRESS_LOG_INTERVAL = 100
ANALYSIS_FPS = 5
ANALYSIS_FRAME_INTERVAL_MS = int(round(1000 / ANALYSIS_FPS))


@dataclass(frozen=True)
class VideoFrame:
    frame_index: int
    video_time_ms: int
    image_name: str
    image_array: Any


@dataclass
class VideoAnalysisEvent:
    frame_index: int
    video_time_ms: int
    scene: SceneName | None
    scene_payload: dict[str, Any] | None
    combined_text: str


@dataclass
class VideoAnalysisResult:
    session: SessionSummary | None
    events: list[VideoAnalysisEvent]


class PyAVVideoSource:
    def __init__(self, opener: Callable[[str], Any] | None = None):
        self._opener = opener

    def iter_frames(self, video_path: Path) -> Iterator[VideoFrame]:
        if not video_path.is_file():
            raise FileNotFoundError(f"Video input not found: {video_path}")

        with self._open_container(video_path) as container:
            streams = getattr(container, "streams", None)
            video_streams = list(getattr(streams, "video", [])) if streams is not None else []
            if not video_streams:
                raise ValueError(f"No video stream found in {video_path}")
            stream = video_streams[0]
            frame_duration_ms = _frame_duration_ms_from_rate(getattr(stream, "average_rate", None))
            logger.info(
                "Opened video %s with PyAV: average_rate=%s, fallback_frame_duration_ms=%s",
                video_path,
                getattr(stream, "average_rate", None),
                frame_duration_ms,
            )

            previous_time_ms: int | None = None
            decoded_count = 0
            for frame_index, frame in enumerate(container.decode(video=0)):
                pts = getattr(frame, "pts", None)
                if pts is not None:
                    video_time_ms = _pts_to_ms(pts, getattr(frame, "time_base", None))
                elif frame_index == 0:
                    video_time_ms = 0
                    logger.info("Frame %d in %s has no pts; using 0ms", frame_index, video_path.name)
                else:
                    if frame_duration_ms is None or previous_time_ms is None:
                        raise ValueError(
                            f"Frame timestamp missing without recoverable frame-rate fallback in {video_path}"
                        )
                    video_time_ms = previous_time_ms + frame_duration_ms
                    logger.info(
                        "Frame %d in %s has no pts; using fallback timestamp %dms",
                        frame_index,
                        video_path.name,
                        video_time_ms,
                    )

                previous_time_ms = video_time_ms
                decoded_count += 1
                yield VideoFrame(
                    frame_index=frame_index,
                    video_time_ms=video_time_ms,
                    image_name=f"{video_path.stem}.frame_{frame_index:06d}.png",
                    image_array=frame.to_ndarray(format="bgr24"),
                )
            logger.info("Decoded %d frames from %s", decoded_count, video_path.name)

    def _open_container(self, video_path: Path) -> Any:
        if self._opener is not None:
            return self._opener(str(video_path))

        import av

        return av.open(str(video_path))


def create_video_source() -> PyAVVideoSource:
    return PyAVVideoSource()


def analyze_video(
    *,
    video_path: Path,
    engine: OCREngine,
    video_source: PyAVVideoSource | None = None,
) -> VideoAnalysisResult:
    source = video_source or create_video_source()
    observations: list[SceneObservation] = []
    events: list[VideoAnalysisEvent] = []
    raid_tracker = RaidTracker()
    decoded_frame_count = 0
    analyzed_frame_count = 0
    next_sample_time_ms: int | None = None
    previous_detected_scene: SceneName | None = None

    for frame in source.iter_frames(video_path):
        decoded_frame_count += 1
        if next_sample_time_ms is None:
            next_sample_time_ms = frame.video_time_ms
        if frame.video_time_ms < next_sample_time_ms:
            continue
        next_sample_time_ms = frame.video_time_ms + ANALYSIS_FRAME_INTERVAL_MS
        analyzed_frame_count += 1
        if decoded_frame_count == 1 or decoded_frame_count % PROGRESS_LOG_INTERVAL == 0:
            logger.info(
                "Processing sampled frame=%d time_ms=%d analyzed_frames=%d retained_observations=%d",
                frame.frame_index,
                frame.video_time_ms,
                analyzed_frame_count,
                len(observations),
            )
        try:
            model_preset: OCRModelPreset = "server" if previous_detected_scene == "character_select" else "mobile"
            run = engine.run(frame.image_array, frame.image_name, model_preset=model_preset)
            from .cli import detect_scene_details, format_scene_detection_reason, summarize_combined_text

            detection = detect_scene_details(run.dump)
            if detection.scene != previous_detected_scene:
                if detection.scene == "character_select" and model_preset != "server":
                    run = engine.run(frame.image_array, frame.image_name, model_preset="server")
                    detection = detect_scene_details(run.dump)
                logger.info(
                    "Scene changed at frame=%d time_ms=%d from=%s to=%s because %s text_excerpt=%r",
                    frame.frame_index,
                    frame.video_time_ms,
                    previous_detected_scene or "none",
                    detection.scene or "none",
                    format_scene_detection_reason(detection),
                    summarize_combined_text(run.dump.combined_text),
                )
                previous_detected_scene = detection.scene
            observation = build_scene_observation(
                frame_index=frame.frame_index,
                video_time_ms=frame.video_time_ms,
                dump=run.dump,
                image_array=frame.image_array,
                engine=engine,
                image_stem=Path(frame.image_name).stem,
                detected_scene=detection.scene,
                raid_tracker=raid_tracker,
                model_preset=model_preset if detection.scene == "character_select" else None,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Video analysis failed at frame={frame.frame_index} "
                f"time_ms={frame.video_time_ms} image={frame.image_name}"
            ) from exc
        if observation is None:
            continue
        observations.append(observation)
        events.append(build_video_analysis_event(observation=observation, combined_text=run.dump.combined_text))
        logger.info(
            "Scene hit at frame=%d time_ms=%d scene=%s",
            observation.frame_index,
            observation.video_time_ms,
            observation.scene,
        )

    session = aggregate_session(observations)
    logger.info(
        "Video %s analyzed: decoded_frames=%d, analyzed_frames=%d, retained_observations=%d, session_built=%s",
        video_path.name,
        decoded_frame_count,
        analyzed_frame_count,
        len(observations),
        "yes" if session is not None else "no",
    )
    return VideoAnalysisResult(
        session=session,
        events=events,
    )


def build_scene_observation(
    *,
    frame_index: int,
    video_time_ms: int,
    dump: OCRDump,
    image_array: Any,
    engine: OCREngine,
    image_stem: str,
    detected_scene: SceneName | None = None,
    raid_tracker: RaidTracker | None = None,
    model_preset: OCRModelPreset | None = None,
) -> SceneObservation | None:
    from .cli import detect_scene

    scene = detected_scene if detected_scene is not None else detect_scene(dump)
    if scene is None:
        return None

    if scene == "character_select":
        students, _ = extract_students(
            dump=dump,
            image_array=image_array,
            image_stem=image_stem,
            output_dir=None,
            engine=engine,
            config=CHARACTER_SELECT_CONFIG,
            model_preset=model_preset,
        )
        return SceneObservation(
            frame_index=frame_index,
            video_time_ms=video_time_ms,
            scene="character_select",
            students=students,
        )

    if scene == "raid":
        raid = (
            raid_tracker.extract(dump, image_array, engine)
            if raid_tracker is not None
            else extract_raid_fields(dump)
        )
        return SceneObservation(
            frame_index=frame_index,
            video_time_ms=video_time_ms,
            scene="raid",
            raid=raid,
        )

    if scene == "result":
        return SceneObservation(
            frame_index=frame_index,
            video_time_ms=video_time_ms,
            scene="result",
            result=extract_result_fields(dump),
        )

    return None


def build_video_analysis_event(
    *,
    observation: SceneObservation,
    combined_text: str,
) -> VideoAnalysisEvent:
    if observation.scene == "character_select":
        payload: dict[str, Any] | None = {
            "students": [asdict(student) for student in observation.students or []],
        }
    elif observation.scene == "raid":
        payload = asdict(observation.raid) if observation.raid is not None else None
    elif observation.scene == "result":
        payload = asdict(observation.result) if observation.result is not None else None
    else:
        payload = None

    return VideoAnalysisEvent(
        frame_index=observation.frame_index,
        video_time_ms=observation.video_time_ms,
        scene=observation.scene,
        scene_payload=payload,
        combined_text=combined_text,
    )


def _pts_to_ms(pts: int, time_base: Fraction | float | None) -> int:
    if time_base is None:
        raise ValueError("Missing frame time_base for timestamped video analysis")
    return int(round(float(pts * time_base) * 1000.0))


def _frame_duration_ms_from_rate(rate: Fraction | float | None) -> int | None:
    if rate is None:
        return None
    rate_value = float(rate)
    if rate_value <= 0:
        return None
    return int(round(1000.0 / rate_value))
