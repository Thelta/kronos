# YOLO26 Training

`kronos-train` includes only the merged YOLO26 workflow from `re123`.

This is the training path from [merge_and_train.py](/C:/Users/atilh/Desktop/re123/merge_and_train.py):

- synthetic dataset from `synthetic_dataset_v2/images` and `synthetic_dataset_v2/labels`
- reviewed real images from `labeled_real_data/review`
- real labels in the same review directory, with `_orig` stripped from the image stem
- review filtering from `review_status.json`
- YOLO26n training with `epochs=100`, `batch=2`, `imgsz=1280`, `patience=15`, `device=cuda`

The port does not preserve the separate `train_yolo.py` or `train_on_synthetic.py` workflows.

## Sample Config

Use [packages/kronos-train/configs/yolo26_v1.sample.json](/D:/projects/kronos/packages/kronos-train/configs/yolo26_v1.sample.json:1).

Required dataset inputs:

- `synthetic_images_dir`
- `synthetic_labels_dir`
- `reviewed_real_dir`
- `review_status_file`

## Prepare Dataset

```powershell
uv run --package kronos-train kronos-train prepare-yolo `
  --config packages/kronos-train/configs/yolo26_v1.sample.json
```

## Train YOLO26

```powershell
uv run --package kronos-train kronos-train train-yolo `
  --config packages/kronos-train/configs/yolo26_v1.sample.json
```

If the prepared dataset already exists:

```powershell
uv run --package kronos-train kronos-train train-yolo `
  --config packages/kronos-train/configs/yolo26_v1.sample.json `
  --skip-prepare
```
