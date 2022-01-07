# CV Papers

## Transformers on image tasks

- ViT
  - **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**
  - highlights:
    - consider image as non-overlapping 16x16 patches
    - add an class token and use it only for MLP head classifier
    - layer normalization before MSA and MLP
  - link: [https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)
  - Google
- DEiT
  - **Training data-efficient image transformers & distillation through attention**
  - highlights:
    - Knowledge distillation, pretrain ViT with fewer data
    - a distillation token is added to ViT arch and they use the output of this token to check against teacher's softmax outputs
    - The best DeiT model only pre-trained with ImageNet beats the best ViT pre-trained with JFT-300M.
  - link: [https://arxiv.org/abs/2012.12877](https://arxiv.org/abs/2012.12877)
  - Facebook AI
- BEiT:
- iGPT:
  - Image probing
- Perceiver:
  - **Perceiver: General Perception with Iterative Attention**
  - highlights:
    - leverages an asymmetric attention mechanism to iteratively distill inputs into a tight latent bottleneck
    - latent array pass to a sequence of latent transformer
    - latent array will periodically interact with raw image bytes array by *Cross Attention*
  - link: [https://arxiv.org/abs/2103.03206](https://arxiv.org/abs/2103.03206)
  - DeepMind
- FastFormer:
- Swin Transformer v1:
- Swin Transformer v2:
- TNT:
- DETR:
- CLIP
- MaskFeat:
  - **Masked Feature Prediction for Self-Supervised Visual Pre-Training**
  - link: [https://arxiv.org/abs/2112.09133](https://arxiv.org/abs/2112.09133)
  - Facebook AI, JHU

## Fusion of CNN and Transformer

- MobileViT
- ConViT

## Self-supervised Learning and its use cases on Transformer pretraining

- MoCo
- SimCLR
- BYOL
- Swav
- MoCo-v3
- MAE
  - **Masked Autoencoders Are Scalable Vision Learners**
  - highlights:
    - asymmetric encoder-decoder architecture
    - encoder operates only on the visible subset of patches (without mask tokens) —> short sequence and hence less computation
    - decoder is lightweight and reconstructs the input from the latent representation along with mask tokens
    - masking a high proportion of the input image, e.g., 75%, yields a nontrivial and meaningful self-supervisory task
  - link: [https://arxiv.org/abs/2111.06377](https://arxiv.org/abs/2111.06377)
  - Facebook AI Research (FAIR)
- DINO
- SimMIM

## Video Classification

- ViViT:
- VTN
- MViT
  - **Multiscale Vision Transformers**
  - highlights:
    - hierarchically expand the channel capacity while reducing the spatial resolution
    - video and image recognition
    - train *from scratch*
    - multi-head pooling attention (MHPA): a self attention operator that enables flexible resolution modeling in a transformer block allowing Multiscale Transformers to operate at progressively changing spatiotemporal resolution
    - query pooling only at the first layer of each stage to decrease resolution
    - key-value pooling will be employed in all layers
  - link: [https://arxiv.org/abs/2104.11227](https://arxiv.org/abs/2104.11227)
  - Facebook AI, UCB
- Improved MViT
  - **Improved Multiscale Vision Transformers for Classification and Detection**
  - link: [https://arxiv.org/abs/2112.01526v1](https://arxiv.org/abs/2112.01526v1)
  - Facebook AI, UCB
- BEVT
- Video Swin
- SeqFormer

## Multi-label classification

- Asymmetric Focal Loss
- Query2Label

## CNNs on image classification

- ResNet
- Mobilenet
- Mobilenet v2
- Efficient Net
- Noisy Student

## Semi-supervised Learning

- Meta Pesudo Label
- Mean teachers
- MixMatch
- FixMatch
- FlexMatch:
  - link: [https://openreview.net/forum?id=3qMwV98zLIk](https://openreview.net/forum?id=3qMwV98zLIk)
  - NeurIPS 2021

## Semantic Segmenation

- BiSeNet
- PSPNet
- DeepLab v3
- HRNet

## Others

- VQVAE