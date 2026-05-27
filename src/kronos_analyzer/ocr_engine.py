from __future__ import annotations

from dataclasses import dataclass, field, replace
import gc
import logging
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

from .schemas import OCRDump, OCRLine

RecognitionProfile = Literal["default", "digits"]
OCRModelPreset = Literal["mobile", "server"]
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OCRBackendConfig:
    backend: Literal["rapidocr"] = "rapidocr"
    log_level: str = "INFO"
    use_gpu: bool = True
    cuda_device_id: int = 0
    det_lang: str = "ch"
    cls_lang: str = "ch"
    rec_lang: str = "ch"
    det_ocr_version: str = "ppocrv5"
    cls_ocr_version: str = "ppocrv5"
    rec_ocr_version: str = "ppocrv5"
    det_model_type: str = "mobile"
    cls_model_type: str = "mobile"
    rec_model_type: str = "mobile"

    def with_model_types(
        self,
        *,
        det_model_type: str | None = None,
        cls_model_type: str | None = None,
        rec_model_type: str | None = None,
    ) -> OCRBackendConfig:
        return replace(
            self,
            det_model_type=self.det_model_type if det_model_type is None else det_model_type,
            cls_model_type=self.cls_model_type if cls_model_type is None else cls_model_type,
            rec_model_type=self.rec_model_type if rec_model_type is None else rec_model_type,
        )


@dataclass
class OCRRun:
    dump: OCRDump
    _write_visualization: Callable[[Path], None] = field(repr=False)

    def write_visualization(self, output_path: Path) -> None:
        self._write_visualization(output_path)


class OCREngine(Protocol):
    def run(
        self,
        image_array: Any,
        image_name: str,
        *,
        model_preset: OCRModelPreset | None = None,
    ) -> OCRRun:
        ...

    def detect_text(
        self,
        image_array: Any,
        *,
        model_preset: OCRModelPreset | None = None,
    ) -> list[list[list[float]]]:
        ...

    def recognize_crops(
        self,
        crops: list[Any],
        profile: RecognitionProfile = "default",
        *,
        model_preset: OCRModelPreset | None = None,
    ) -> list[OCRLine]:
        ...


def resolve_ocr_backend_config(
    config: OCRBackendConfig | None = None,
    *,
    model_preset: OCRModelPreset | None = None,
) -> OCRBackendConfig:
    resolved = config or OCRBackendConfig()
    if model_preset is None:
        return resolved
    return resolved.with_model_types(
        det_model_type=model_preset,
        cls_model_type=model_preset,
        rec_model_type=model_preset,
    )


def create_ocr_engine(
    config: OCRBackendConfig | None = None,
    *,
    model_preset: OCRModelPreset | None = None,
) -> OCREngine:
    return OnDemandOCREngine(base_config=config, default_model_preset=model_preset)


def _build_fixed_ocr_engine(config: OCRBackendConfig) -> OCREngine:
    if config.backend != "rapidocr":
        raise ValueError(f"Unsupported OCR backend: {config.backend}")
    return RapidOCREngine(config)


class OnDemandOCREngine:
    def __init__(
        self,
        *,
        base_config: OCRBackendConfig | None = None,
        default_model_preset: OCRModelPreset | None = None,
        engine_builder: Callable[[OCRBackendConfig], OCREngine] | None = None,
    ):
        self._base_config = base_config or OCRBackendConfig()
        self._default_model_preset = default_model_preset
        self._engine_builder = engine_builder or _build_fixed_ocr_engine
        self._active_config: OCRBackendConfig | None = None
        self._active_engine: OCREngine | None = None

    def run(
        self,
        image_array: Any,
        image_name: str,
        *,
        model_preset: OCRModelPreset | None = None,
    ) -> OCRRun:
        return self._get_engine(model_preset=model_preset).run(image_array, image_name)

    def detect_text(
        self,
        image_array: Any,
        *,
        model_preset: OCRModelPreset | None = None,
    ) -> list[list[list[float]]]:
        return self._get_engine(model_preset=model_preset).detect_text(image_array)

    def recognize_crops(
        self,
        crops: list[Any],
        profile: RecognitionProfile = "default",
        *,
        model_preset: OCRModelPreset | None = None,
    ) -> list[OCRLine]:
        return self._get_engine(model_preset=model_preset).recognize_crops(crops, profile=profile)

    def unload(self) -> None:
        self._active_config = None
        self._active_engine = None
        gc.collect()

    def _get_engine(self, *, model_preset: OCRModelPreset | None) -> OCREngine:
        resolved = resolve_ocr_backend_config(
            self._base_config,
            model_preset=self._default_model_preset if model_preset is None else model_preset,
        )
        if self._active_engine is None or self._active_config != resolved:
            self.unload()
            self._active_engine = self._engine_builder(resolved)
            self._active_config = resolved
        return self._active_engine


def build_rapidocr_params(config: OCRBackendConfig) -> dict[str, object]:
    from rapidocr import LangCls, LangDet, LangRec, ModelType, OCRVersion

    use_cuda = _resolve_cuda_enabled(config)
    return {
        "Global.log_level": config.log_level,
        "EngineConfig.onnxruntime.use_cuda": use_cuda,
        "EngineConfig.onnxruntime.cuda_ep_cfg.device_id": config.cuda_device_id,
        "Det.lang_type": _map_enum(config.det_lang, LangDet, "det_lang"),
        "Det.ocr_version": _map_enum(config.det_ocr_version, OCRVersion, "det_ocr_version"),
        "Det.model_type": _map_enum(config.det_model_type, ModelType, "det_model_type"),
        "Cls.lang_type": _map_enum(config.cls_lang, LangCls, "cls_lang"),
        "Cls.ocr_version": _map_enum(config.cls_ocr_version, OCRVersion, "cls_ocr_version"),
        "Cls.model_type": _map_enum(config.cls_model_type, ModelType, "cls_model_type"),
        "Rec.lang_type": _map_enum(config.rec_lang, LangRec, "rec_lang"),
        "Rec.ocr_version": _map_enum(config.rec_ocr_version, OCRVersion, "rec_ocr_version"),
        "Rec.model_type": _map_enum(config.rec_model_type, ModelType, "rec_model_type"),
    }


class RapidOCREngine:
    def __init__(self, config: OCRBackendConfig):
        from rapidocr import RapidOCR

        self.config = config
        self._engine = RapidOCR(params=build_rapidocr_params(config))
        self._log_execution_providers()

    def run(
        self,
        image_array: Any,
        image_name: str,
        *,
        model_preset: OCRModelPreset | None = None,
    ) -> OCRRun:
        _validate_fixed_engine_model_preset(self.config, model_preset)
        result = self._engine(image_array)
        dump = _build_dump(image_name, result)
        return OCRRun(
            dump=dump,
            _write_visualization=lambda output_path: result.vis(str(output_path)),
        )

    def detect_text(
        self,
        image_array: Any,
        *,
        model_preset: OCRModelPreset | None = None,
    ) -> list[list[list[float]]]:
        _validate_fixed_engine_model_preset(self.config, model_preset)
        result = self._engine(image_array, use_cls=False, use_rec=False)
        boxes = _coerce_sequence(getattr(result, "boxes", None))
        return [_normalize_box(box) for box in boxes]

    def recognize_crops(
        self,
        crops: list[Any],
        profile: RecognitionProfile = "default",
        *,
        model_preset: OCRModelPreset | None = None,
    ) -> list[OCRLine]:
        _validate_fixed_engine_model_preset(self.config, model_preset)
        if not crops:
            return []
        if profile == "default":
            return self._recognize_default_crops(crops)
        if profile == "digits":
            return [self._recognize_digits_crop(crop) for crop in crops]
        raise ValueError(f"Unsupported recognition profile: {profile}")

    def _recognize_default_crops(self, crops: list[Any]) -> list[OCRLine]:
        crop_images = [_ensure_three_channel(crop) for crop in crops]
        if self._engine.use_cls:
            crop_images, _ = self._engine.cls_and_rotate(crop_images)
        rec_res = self._engine.recognize_txt(crop_images)

        txts = _coerce_sequence(getattr(rec_res, "txts", None))
        scores = _coerce_sequence(getattr(rec_res, "scores", None))
        lines: list[OCRLine] = []
        for index, text in enumerate(txts):
            score = float(scores[index]) if index < len(scores) and scores[index] is not None else None
            lines.append(OCRLine(text=str(text), score=score, box=[]))
        return lines

    def _recognize_digits_crop(self, crop: Any) -> OCRLine:
        import numpy as np

        crop_bgr = _ensure_three_channel(crop)
        text_rec = self._engine.text_rec

        _, img_h, img_w = text_rec.rec_image_shape[:3]
        max_wh_ratio = max(img_w / img_h, crop_bgr.shape[1] / float(crop_bgr.shape[0]))
        norm_img = text_rec.resize_norm_img(crop_bgr, max_wh_ratio)
        norm_img_batch = np.expand_dims(norm_img, axis=0).astype(np.float32)
        preds = text_rec.session(norm_img_batch)

        text, score = _decode_digits_only_preds(preds, text_rec.postprocess_op)
        return OCRLine(text=text, score=score, box=[])

    def _log_execution_providers(self) -> None:
        component_sessions = {
            "det": getattr(getattr(self._engine, "text_det", None), "session", None),
            "cls": getattr(getattr(self._engine, "text_cls", None), "session", None),
            "rec": getattr(getattr(self._engine, "text_rec", None), "session", None),
        }
        available_providers = _get_onnxruntime_available_providers()
        active_providers = {
            name: session.get_providers()
            for name, session in component_sessions.items()
            if session is not None and hasattr(session, "get_providers")
        }
        logger.info(
            "RapidOCR initialized: use_gpu=%s, cuda_device_id=%d, available_ort_providers=%s, active_component_providers=%s",
            self.config.use_gpu,
            self.config.cuda_device_id,
            available_providers,
            active_providers,
        )


def _map_enum(raw_value: str, enum_type: type[Any], field_name: str) -> Any:
    normalized = raw_value.strip().lower().replace("-", "")
    for item in enum_type:
        item_name = item.name.lower().replace("_", "")
        item_value = str(item.value).lower().replace("-", "")
        if normalized in (item_name, item_value):
            return item
    raise ValueError(f"Unsupported {field_name}: {raw_value}")


def _validate_fixed_engine_model_preset(
    config: OCRBackendConfig,
    model_preset: OCRModelPreset | None,
) -> None:
    if model_preset is None:
        return
    if (
        config.det_model_type == model_preset
        and config.cls_model_type == model_preset
        and config.rec_model_type == model_preset
    ):
        return
    raise ValueError(
        "This OCR engine instance is fixed to a different model preset; "
        "use create_ocr_engine() for on-demand switching."
    )


def _resolve_cuda_enabled(config: OCRBackendConfig) -> bool:
    if not config.use_gpu:
        logger.info("OCR backend configured for CPU execution")
        return False

    _preload_onnxruntime_gpu_dlls()
    available_providers = _get_onnxruntime_available_providers()
    if "CUDAExecutionProvider" in available_providers:
        logger.info(
            "Enabling CUDA for OCR backend on device %d; available ORT providers=%s",
            config.cuda_device_id,
            available_providers,
        )
        return True

    logger.warning(
        "GPU OCR requested but CUDAExecutionProvider is unavailable; falling back to CPU. available ORT providers=%s",
        available_providers,
    )
    return False


def _get_onnxruntime_available_providers() -> list[str]:
    try:
        import onnxruntime as ort
    except ImportError:
        return []
    return list(ort.get_available_providers())


def _preload_onnxruntime_gpu_dlls() -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        return

    preload = getattr(ort, "preload_dlls", None)
    if preload is None:
        logger.info("onnxruntime.preload_dlls is unavailable; skipping packaged GPU DLL preload")
        return

    try:
        preload(cuda=True, cudnn=True, msvc=True, directory="")
        logger.info("Preloaded ONNX Runtime GPU DLLs from packaged dependencies")
    except Exception as exc:
        logger.warning("Failed to preload ONNX Runtime GPU DLLs: %s", exc)


def _build_dump(image_name: str, result: Any) -> OCRDump:
    lines: list[OCRLine] = []
    boxes = _coerce_sequence(getattr(result, "boxes", None))
    txts = _coerce_sequence(getattr(result, "txts", None))
    scores = _coerce_sequence(getattr(result, "scores", None))

    count = max(len(boxes), len(txts), len(scores))
    for index in range(count):
        box = _normalize_box(boxes[index]) if index < len(boxes) else []
        text = str(txts[index]) if index < len(txts) else ""
        score = float(scores[index]) if index < len(scores) and scores[index] is not None else None
        lines.append(OCRLine(text=text, score=score, box=box))

    combined_text = "\n".join(line.text for line in lines if line.text)
    return OCRDump(
        image=image_name,
        line_count=len(lines),
        combined_text=combined_text,
        lines=lines,
    )


def _normalize_box(box: Any) -> list[list[float]]:
    normalized: list[list[float]] = []
    for point in box:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            normalized.append([float(point[0]), float(point[1])])
    return normalized


def _coerce_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _ensure_three_channel(image: Any) -> Any:
    import cv2

    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 3:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)


def _decode_digits_only_preds(preds: Any, decoder: Any) -> tuple[str, float]:
    import numpy as np

    preds_array = preds[0] if isinstance(preds, (list, tuple)) else preds
    allowed_indices = [0] + [
        index for index, character in enumerate(decoder.character) if character.isdigit()
    ]
    restricted_preds = preds_array[:, :, allowed_indices]
    restricted_argmax = restricted_preds.argmax(axis=2)
    preds_idx = np.take(np.array(allowed_indices), restricted_argmax)
    preds_prob = restricted_preds.max(axis=2)

    line_results, _ = decoder.decode(
        preds_idx,
        preds_prob,
        return_word_box=False,
        remove_duplicate=True,
    )
    if not line_results:
        return "", 0.0
    return line_results[0]
