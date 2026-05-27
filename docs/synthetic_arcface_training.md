# Synthetic ArcFace Training

`kronos-train` consumes synthetic card datasets generated in `shittim`.

`leader` is ignored by training targets. Only `character_id`, `star_value`, and `assist` are used as labels.

## Workflow

1. Generate synthetic card classification data in `shittim` with a large enough `variants-per-identity`.
2. Train `mobilenetv4-small` with ArcFace identity loss plus `star_value` and `assist` heads.
The trainer consumes the raw `manifest.jsonl` directly and internally derives:

- `gallery` from the cleanest samples in the `train` split
- `train_query` from the remaining `train` split
- `val_query` from `val`
- `test_query` from `test`

```powershell
uv run --package kronos-train kronos-train train `
  --config packages/kronos-train/configs/arcface_v1.sample.json `
  --override data.manifest_path=D:/projects/shittim/outputs/cards/manifest.jsonl `
  --override trainer.output_dir=training_runs/kronos_arcface
```

3. Evaluate retrieval:

```powershell
uv run --package kronos-train kronos-train evaluate `
  --checkpoint training_runs/kronos_arcface/best.pt `
  --subset test_query `
  --output training_runs/kronos_arcface/test_metrics.json
```

4. Export gallery prototypes:

```powershell
uv run --package kronos-train kronos-train export-gallery `
  --checkpoint training_runs/kronos_arcface/best.pt `
  --output training_runs/kronos_arcface/gallery.json
```

## Notes

- V1 is synthetic-only.
- The model input is the full card crop, not a normalized portrait crop.
- Identity is resolved by cosine retrieval against exported gallery prototypes.
- `star_value` and `assist` are predicted by auxiliary heads.
- `leader` is not treated as an auxiliary label.
