from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kronos_analyzer.character_select import extract_students  # noqa: E402
from kronos_analyzer.cli import handle_dump_ocr  # noqa: E402
from kronos_analyzer.config import CHARACTER_SELECT_CONFIG  # noqa: E402
from kronos_analyzer.ocr_engine import (  # noqa: E402
    OnDemandOCREngine,
    OCRBackendConfig,
    OCRRun,
    RapidOCREngine,
    _preload_onnxruntime_gpu_dlls,
    _resolve_cuda_enabled,
    build_rapidocr_params,
    create_ocr_engine,
    resolve_ocr_backend_config,
)
from kronos_analyzer.schemas import OCRDump, OCRLine  # noqa: E402


class FakeEngine:
    def __init__(self, run: OCRRun | None = None, crop_lines: list[OCRLine] | None = None):
        self._run = run
        self._crop_lines = crop_lines or []
        self.calls: list[tuple[str, object]] = []

    def run(self, image_array, image_name: str, *, model_preset=None) -> OCRRun:
        self.calls.append(("run", image_name))
        if self._run is None:
            raise AssertionError("run() was not configured for this fake engine")
        return self._run

    def detect_text(self, image_array, *, model_preset=None):
        self.calls.append(("detect_text", None))
        return []

    def recognize_crops(self, crops, profile: str = "default", *, model_preset=None) -> list[OCRLine]:
        self.calls.append(("recognize_crops", profile))
        return list(self._crop_lines)


class OCREngineTests(unittest.TestCase):
    def test_build_rapidocr_params_defaults_use_configured_weights(self) -> None:
        with patch("kronos_analyzer.ocr_engine._get_onnxruntime_available_providers", return_value=["CUDAExecutionProvider", "CPUExecutionProvider"]):
            params = build_rapidocr_params(OCRBackendConfig())

        self.assertEqual(params["Det.model_type"].value, "server")
        self.assertEqual(params["Cls.model_type"].value, "mobile")
        self.assertEqual(params["Rec.model_type"].value, "mobile")
        self.assertTrue(params["EngineConfig.onnxruntime.use_cuda"])
        self.assertEqual(params["EngineConfig.onnxruntime.cuda_ep_cfg.device_id"], 0)

    def test_resolve_ocr_backend_config_applies_mobile_preset(self) -> None:
        resolved = resolve_ocr_backend_config(OCRBackendConfig(), model_preset="mobile")

        self.assertEqual(resolved.det_model_type, "mobile")
        self.assertEqual(resolved.cls_model_type, "mobile")
        self.assertEqual(resolved.rec_model_type, "mobile")

    def test_resolve_ocr_backend_config_applies_server_preset(self) -> None:
        resolved = resolve_ocr_backend_config(OCRBackendConfig(), model_preset="server")

        self.assertEqual(resolved.det_model_type, "server")
        self.assertEqual(resolved.cls_model_type, "server")
        self.assertEqual(resolved.rec_model_type, "server")

    def test_create_ocr_engine_resolves_model_preset_before_building_backend(self) -> None:
        created = create_ocr_engine(model_preset="mobile")

        self.assertIsInstance(created, OnDemandOCREngine)

    def test_on_demand_engine_switches_presets_per_call(self) -> None:
        built_configs: list[OCRBackendConfig] = []
        mobile_run = OCRRun(
            dump=OCRDump(image="mobile.png", line_count=0, combined_text="", lines=[]),
            _write_visualization=lambda output_path: None,
        )
        server_run = OCRRun(
            dump=OCRDump(image="server.png", line_count=0, combined_text="", lines=[]),
            _write_visualization=lambda output_path: None,
        )

        class BuiltEngine:
            def __init__(self, config: OCRBackendConfig):
                self.config = config

            def run(self, image_array, image_name: str) -> OCRRun:
                return mobile_run if self.config.det_model_type == "mobile" else server_run

            def detect_text(self, image_array):
                return []

            def recognize_crops(self, crops, profile: str = "default") -> list[OCRLine]:
                return []

        engine = OnDemandOCREngine(engine_builder=lambda config: built_configs.append(config) or BuiltEngine(config))

        first = engine.run(object(), "frame_a.png", model_preset="mobile")
        second = engine.run(object(), "frame_b.png", model_preset="server")

        self.assertIs(first, mobile_run)
        self.assertIs(second, server_run)
        self.assertEqual([config.det_model_type for config in built_configs], ["mobile", "server"])
        self.assertEqual([config.cls_model_type for config in built_configs], ["mobile", "server"])
        self.assertEqual([config.rec_model_type for config in built_configs], ["mobile", "server"])

    def test_on_demand_engine_reuses_active_preset_until_it_changes(self) -> None:
        built_configs: list[OCRBackendConfig] = []

        class BuiltEngine:
            def __init__(self, config: OCRBackendConfig):
                self.config = config

            def run(self, image_array, image_name: str) -> OCRRun:
                return OCRRun(
                    dump=OCRDump(image=image_name, line_count=0, combined_text="", lines=[]),
                    _write_visualization=lambda output_path: None,
                )

            def detect_text(self, image_array):
                return []

            def recognize_crops(self, crops, profile: str = "default") -> list[OCRLine]:
                return []

        engine = OnDemandOCREngine(engine_builder=lambda config: built_configs.append(config) or BuiltEngine(config))

        engine.run(object(), "frame_a.png", model_preset="mobile")
        engine.run(object(), "frame_b.png", model_preset="mobile")
        engine.run(object(), "frame_c.png", model_preset="server")

        self.assertEqual([config.det_model_type for config in built_configs], ["mobile", "server"])

    def test_build_rapidocr_params_honors_non_default_values(self) -> None:
        with patch("kronos_analyzer.ocr_engine._get_onnxruntime_available_providers", return_value=["CUDAExecutionProvider", "CPUExecutionProvider"]):
            params = build_rapidocr_params(
                OCRBackendConfig(
                    cuda_device_id=2,
                    det_lang="en",
                    rec_lang="japan",
                    det_ocr_version="ppocrv4",
                    det_model_type="mobile",
                    cls_model_type="server",
                    rec_model_type="mobile",
                )
            )

        self.assertEqual(params["Det.lang_type"].value, "en")
        self.assertEqual(params["Rec.lang_type"].value, "japan")
        self.assertEqual(params["Det.ocr_version"].value, "PP-OCRv4")
        self.assertEqual(params["Det.model_type"].value, "mobile")
        self.assertEqual(params["Cls.model_type"].value, "server")
        self.assertEqual(params["Rec.model_type"].value, "mobile")
        self.assertTrue(params["EngineConfig.onnxruntime.use_cuda"])
        self.assertEqual(params["EngineConfig.onnxruntime.cuda_ep_cfg.device_id"], 2)

    def test_gpu_request_falls_back_to_cpu_when_cuda_provider_is_unavailable(self) -> None:
        with (
            patch("kronos_analyzer.ocr_engine._preload_onnxruntime_gpu_dlls"),
            patch("kronos_analyzer.ocr_engine._get_onnxruntime_available_providers", return_value=["CPUExecutionProvider"]),
        ):
            self.assertFalse(_resolve_cuda_enabled(OCRBackendConfig(use_gpu=True)))

    def test_cpu_config_disables_cuda_even_if_provider_exists(self) -> None:
        with patch("kronos_analyzer.ocr_engine._get_onnxruntime_available_providers", return_value=["CUDAExecutionProvider", "CPUExecutionProvider"]):
            self.assertFalse(_resolve_cuda_enabled(OCRBackendConfig(use_gpu=False)))

    def test_preload_onnxruntime_gpu_dlls_uses_packaged_search_path(self) -> None:
        fake_ort = SimpleNamespace(preload_dlls=lambda **kwargs: kwargs)
        recorded: list[dict[str, object]] = []
        fake_ort.preload_dlls = lambda **kwargs: recorded.append(kwargs)

        with patch.dict(sys.modules, {"onnxruntime": fake_ort}):
            _preload_onnxruntime_gpu_dlls()

        self.assertEqual(
            recorded,
            [{"cuda": True, "cudnn": True, "msvc": True, "directory": ""}],
        )

    def test_recognize_crops_routes_profiles(self) -> None:
        engine = RapidOCREngine.__new__(RapidOCREngine)
        engine.config = OCRBackendConfig()
        default_result = [OCRLine(text="alpha", score=0.8, box=[])]
        digit_result = OCRLine(text="5", score=0.9, box=[])

        with (
            patch.object(engine, "_recognize_default_crops", return_value=default_result) as default_mock,
            patch.object(engine, "_recognize_digits_crop", return_value=digit_result) as digits_mock,
        ):
            self.assertEqual(engine.recognize_crops([object()], profile="default"), default_result)
            self.assertEqual(engine.recognize_crops([object(), object()], profile="digits"), [digit_result, digit_result])

        default_mock.assert_called_once()
        self.assertEqual(digits_mock.call_count, 2)

    def test_ocr_run_writes_visualization_without_exposing_backend_types(self) -> None:
        dump = OCRDump(image="frame.png", line_count=1, combined_text="alpha", lines=[OCRLine(text="alpha", score=0.9, box=[])])
        written_paths: list[Path] = []
        run = OCRRun(dump=dump, _write_visualization=written_paths.append)

        run.write_visualization(Path("sample.vis.png"))

        self.assertIs(run.dump, dump)
        self.assertEqual(written_paths, [Path("sample.vis.png")])

    def test_character_select_digit_fallback_uses_engine_profile(self) -> None:
        dump = OCRDump(
            image="frame.png",
            line_count=3,
            combined_text="STRIKER\nHoshino\nLv.90",
            lines=[
                OCRLine(
                    text="STRIKER",
                    score=0.99,
                    box=[[10.0, 72.0], [32.0, 72.0], [32.0, 80.0], [10.0, 80.0]],
                ),
                OCRLine(
                    text="Hoshino",
                    score=0.95,
                    box=[[35.0, 64.0], [62.0, 64.0], [62.0, 72.0], [35.0, 72.0]],
                ),
                OCRLine(
                    text="Lv.90",
                    score=0.9,
                    box=[[35.0, 70.0], [52.0, 70.0], [52.0, 78.0], [35.0, 78.0]],
                )
            ],
        )
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        engine = FakeEngine(crop_lines=[OCRLine(text="5", score=0.95, box=[])])

        with tempfile.TemporaryDirectory() as temp_dir:
            students, _ = extract_students(
                dump=dump,
                image_array=image,
                image_stem="frame",
                output_dir=Path(temp_dir),
                engine=engine,
                config=CHARACTER_SELECT_CONFIG,
            )

        self.assertEqual(engine.calls, [("recognize_crops", "digits")])
        self.assertIsNone(students[0].star_yellow)
        self.assertIsNone(students[0].star_blue)

    def test_cli_dump_ocr_uses_engine_abstraction_and_preserves_output_shape(self) -> None:
        dump = OCRDump(
            image="raid_01.png",
            line_count=1,
            combined_text="battle boss",
            lines=[OCRLine(text="battle boss", score=0.88, box=[[1.0, 2.0], [3.0, 4.0]])],
        )

        def write_visualization(output_path: Path) -> None:
            output_path.write_bytes(b"vis")

        engine = FakeEngine(run=OCRRun(dump=dump, _write_visualization=write_visualization))

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "raid_01.png"
            input_path.touch()
            output_dir = temp_path / "ocr"
            args = argparse.Namespace(
                input=str(input_path),
                output=str(output_dir),
                glob="*.png",
                identify_scene="ocr",
            )

            with (
                patch("kronos_analyzer.cli.create_ocr_engine", return_value=engine),
                patch("kronos_analyzer.cli.load_image_array", return_value=np.zeros((8, 8, 3), dtype=np.uint8)),
            ):
                exit_code = handle_dump_ocr(args)

            self.assertEqual(exit_code, 0)
            self.assertEqual(engine.calls, [("run", "raid_01.png")])

            written_json = json.loads((output_dir / "raid_01.json").read_text(encoding="utf-8"))
            written_txt = (output_dir / "raid_01.txt").read_text(encoding="utf-8")

            self.assertEqual(written_json["image"], "raid_01.png")
            self.assertEqual(written_json["combined_text"], "battle boss")
            self.assertIn("image: raid_01.png", written_txt)
            self.assertEqual((output_dir / "raid_01.vis.png").read_bytes(), b"vis")

    def test_rapidocr_import_is_isolated_to_engine_module(self) -> None:
        module_dir = ROOT / "src" / "kronos_analyzer"
        importers: list[str] = []
        for path in sorted(module_dir.glob("*.py")):
            text = path.read_text(encoding="utf-8")
            if "from rapidocr" in text or "import rapidocr" in text:
                importers.append(path.name)

        self.assertEqual(importers, ["ocr_engine.py"])


if __name__ == "__main__":
    unittest.main()
