# Earlier work: my own U-Net (not used in the final submission)

`unet_own_training.ipynb` trains and runs a model I built myself before switching to the public YOLO models. Its best
public score was **0.38** (ensemble of the 5 folds). It was run on Google Colab, so the paths at the top point to Google
Drive.

- **Input:** the whole solar disk at 1024 px, after limb-darkening flattening and CLAHE contrast equalisation.
- **Targets:** soft masks, where each pixel's value is the fraction of annotators who marked it, plus a filament-spine map.
- **Model:** U-Net (`segmentation_models_pytorch`) with an ImageNet-pretrained EfficientNet-B4 encoder and two heads
  (filament, spine). AdamW, lr 3e-4, weight decay 1e-4, batch 4, 45 epochs, exponential moving average of the weights.
- **Validation:** 5 folds grouped by observation day, since images of the same day show the same filaments.
- **Instances:** dihedral test-time augmentation, a hysteresis and spine-bridging decoder that turns the probability
  maps into filaments, with all decoder thresholds tuned for Panoptic Quality on out-of-fold images.
