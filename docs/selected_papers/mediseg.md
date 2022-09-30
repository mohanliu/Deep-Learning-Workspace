# Deep Learning for Medical Image Segmentation: Tricks, Challenges and Future Directions
> https://arxiv.org/abs/2209.10307


## Pre-Training model
- PyTorch official from `torchvision`
- ImageNet 1k
- ImageNet 21k
- SimCLR
- MoCo
- ModelGe

## Data Pre-Processing
- Patching
- OverSam
  - random oversampling
  - synthetic minority oversampling (SMOTE)
  - borderline SMOTE
  - adaptive synthetic sampling
- IntesNorm

## Data Augmentation
- geometric transformation-based augmentation
  - pixel-level transform
    - random brightness contrast
    - random gamma
    - CLAHE
      > Contrast limited adaptive histogram equalization
  - spatial-level transform
    - random rotation
    - random scale
    - random flip (horizontal, vertical)
    - random shear
    - random translation
- generative adversarial network (GAN)-based data augmentation

## Model Implementation
- Deep Supervision
  - Training Deeper Convolutional Networks with Deep Supervision
  - https://arxiv.org/abs/1505.02496
- class balance loss
  - Dice 
  - Focal
  - Tversky
  - weighted cross-entropy loss
- online hard example mining (OHEM)
  - https://arxiv.org/abs/1604.03540

## Model Inference
- TTA (Test Time Augmentation)
  - Better aggregation in test-time augmentation
  - https://arxiv.org/abs/2011.11156
- Ensemble
  - voting
  - averaging
  - stacking
  - non-cross-stacking (blending)

## Result Post-Processing
- ABL-CS
  - all-but-largest-component-suppression
  - based on domain knowledge
  - e.g. there should be only one heart of a x-ray image, so if there are small segmentation areas in the obtained mask, we need to remove this small areas
- RSA
  - removal of small area