## Kronos Analyzer

Simple RapidOCR pipeline for dumping OCR text and saving visualizations.

### Install

```powershell
uv sync --package kronos-analyzer
```

### Run on a folder

```powershell
uv run --package kronos-analyzer kronos-analyzer dump-ocr --input samples/frames --output outputs/ocr
```

### Run on one image

```powershell
uv run --package kronos-analyzer kronos-analyzer dump-ocr --input samples/frames/raid_01.png --output outputs/ocr
```

### Analyze one video

```powershell
uv run --package kronos-analyzer kronos-analyzer analyze-video --input samples/videos/raid_01.mp4 --output outputs/video
```

### Scene mode

Use `--identify-scene` to choose how much scene-specific processing runs:

```powershell
uv run --package kronos-analyzer kronos-analyzer dump-ocr --input samples/frames --output outputs/ocr --identify-scene ocr
uv run --package kronos-analyzer kronos-analyzer dump-ocr --input samples/frames --output outputs/ocr --identify-scene auto
uv run --package kronos-analyzer kronos-analyzer dump-ocr --input samples/frames --output outputs/ocr --identify-scene character_select
```

Modes:

- `ocr`: OCR only
- `auto`: detect the scene from OCR text and run the matching scene pipeline
- `character_select`, `raid`, `result`: force a scene pipeline

### Output

For each image, the pipeline writes:

- `<name>.json`: boxes, texts, scores, and combined text
- `<name>.txt`: one OCR line per detection
- `<name>.vis.png`: RapidOCR visualization; for `character_select`, this is the original vis plus star search regions and resolved results
- `<name>.scene.json`: scene request, resolved scene, and scene-specific outputs when `--identify-scene` is not `ocr`

For `character_select`, star values are handled inside the main OCR flow:

- first try the normal full-frame OCR result above each `Lv.*`
- if no digit is found there, crop above `Lv.*`
- run the specialized digit-only fallback on that crop

Character-select crop geometry is configured in [src/kronos_analyzer/config.py](/D:/projects/kronos/src/kronos_analyzer/config.py:1), not through CLI flags.

Scene outputs:

- `character_select`: `students` with `name`, `star_level`, and `level`
- `raid`: `boss_remaining_hp`, `boss_total_hp`, and `timer`
- `result`: `ranking_point` and `timer`

For video analysis, the pipeline writes:

- `<name>.session.json`: aggregated session summary
- `<name>.events.json`: compact per-frame scene timeline used for debugging

### Workspace

The repo is now a `uv` workspace:

- root package: `kronos-analyzer`
- shared package: `kronos-shared`
- training package: `kronos-train`

Use `uv sync --package kronos-train` to install the heavy ML environment only when needed. Synthetic card training is documented in [docs/synthetic_arcface_training.md](/D:/projects/kronos/docs/synthetic_arcface_training.md).

YOLO26 merged-detector training is documented in [docs/yolo26_training.md](/D:/projects/kronos/docs/yolo26_training.md).
