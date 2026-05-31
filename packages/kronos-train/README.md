# Kronos Train

Training commands for synthetic Blue Archive card recognition.

Typical flow:

1. Generate synthetic card images and a `manifest.jsonl` in `shittim`.
2. Train directly from that manifest; `kronos-train` derives gallery/query subsets internally from `train`/`val`/`test`.
3. Evaluate retrieval and export gallery prototypes.

Skill retrieval notes:

- `evaluate-skill` can use the default synthetic train-derived gallery or a canonical gallery via `--canonical-manifest`.
- `skill_arcface_v1.sample.json` exposes `data.train_sampler`, which can be set to `balanced_identity_mode` to interleave training rows more evenly by identity and `render_mode`.

YOLO26 detector training is also available, but only for the merged workflow from `re123`: synthetic data plus reviewed real data combined into one Ultralytics-style dataset before training.
