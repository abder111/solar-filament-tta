# Model weights (by hdjojo, Apache License 2.0)

| File | Model | Source |
|---|---|---|
| `yolo26x-seg-solar-filament-2048-1109.pt` | YOLO26x-seg, 2048 px | <https://www.kaggle.com/models/hdjojo/yolo26x-seg-solar-filament> |
| `yolov8x-seg-solar-filament-2048-1110.pt` | YOLOv8x-seg, 2048 px | <https://www.kaggle.com/models/hdjojo/yolov8x-seg-solar-filament> |

These two files were trained and published by the Kaggle user hdjojo. They are redistributed here **unmodified**, under
the Apache License 2.0 (full text in `LICENSE-Apache-2.0.txt`). According to the author, they were trained on the 707
competition training images only. They are stored with Git LFS: run `git lfs pull` to download them.

## Training settings stored in the checkpoints

Read with `torch.load(path, weights_only=False)['train_args']` (Ultralytics checkpoints keep their training arguments).

| Setting | YOLO26x-seg | YOLOv8x-seg |
|---|---|---|
| Architecture / starting weights | `yolo26x-seg.pt`, initialised from an earlier checkpoint `best.pt` | `yolov8x-seg.pt`, initialised from `best.pt` |
| Image size | 2048 | 2048 |
| Epochs (early-stopping patience) | 500 (25) | 300 (30) |
| Batch size | 8 | 8 |
| Optimizer | MuSGD, lr 0.001, cosine decay to 1 %, momentum 0.937, weight decay 0.0001, 1 warm-up epoch | same |
| Mixed precision | yes | yes |
| Flips | up-down 0.5, left-right 0.5 | up-down 0.5, left-right 0.5 |
| Geometric | rotation ±10°, shear ±10°, translation 0.1, scale ±10 %, perspective 0.0005 | rotation ±10°, shear ±10°, translation 0.05, scale ±5 %, multi-scale ±25 %, perspective 0.0005 |
| Other | saturation and brightness ±10 %, copy-paste 0.1, no mosaic, no mixup | saturation and brightness ±10 %, copy-paste 0.25, no mosaic, no mixup |
| Mask settings | overlap masks, mask ratio 2 | overlap masks, mask ratio 2 |
| Seed | 3407 | 1110 |

Not stored in the checkpoints: the train/validation split (`data.yaml`), the settings of the earlier stage that produced
`best.pt`, the training date, the Ultralytics version and the validation scores.
