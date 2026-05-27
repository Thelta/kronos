# Agents

## Project intent

Build an OCR-first video analyzer that classifies scenes from configurable frame regions, then runs scene-specific extraction logic.

## Current architecture rules

- Keep scene detection data-driven. Keywords and regions belong in config files, not in Python constants spread through business logic.
- Keep OCR backends behind `OCREngine`. RapidOCR is the first backend; custom models will be added later without coupling them to scene processors.
- Use normalized coordinates for frame regions so the same config works across multiple capture resolutions.
- Scene processors may contain extraction heuristics, but they should consume named fields from config instead of hardcoding raw crop coordinates.
- Do not hardcode per-unit card slots for character select. Treat the roster area as a repeated layout that should later be segmented dynamically.

## Working conventions

- Prefer small pure-Python tests for matching, parsing, and config loading.
- Import heavy runtime dependencies such as `cv2` lazily inside execution paths so config and logic tests remain cheap.
- Treat `samples/frames/` as the landing area for calibration frames from the user.

## Next expected step

Once reference frames are available, tune `config/scene_definitions.sample.json` into a real scene definition file by adjusting regions and keywords from observed UI text.
