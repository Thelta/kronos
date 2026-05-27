# Kronos Train

Training commands for synthetic Blue Archive card recognition.

Typical flow:

1. Generate synthetic card images and a `manifest.jsonl` in `shittim`.
2. Train directly from that manifest; `kronos-train` derives gallery/query subsets internally from `train`/`val`/`test`.
3. Evaluate retrieval and export gallery prototypes.

YOLO26 detector training is also available, but only for the merged workflow from `re123`: synthetic data plus reviewed real data combined into one Ultralytics-style dataset before training.
