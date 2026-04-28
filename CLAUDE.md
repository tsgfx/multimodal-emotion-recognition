# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multimodal Emotion Recognition using RAVDESS dataset. Implements audio-only and face-only single-modality models, plus late fusion evaluation combining both modalities. Emotion classes: angry, disgust, fearful, happy, sad (5 classes, Neutral removed).

**Current best: Test Macro F1 = 0.9334** (classwise late fusion with augmented face model)

## Common Commands

### Data Preparation
```bash
# Generate metadata CSV from raw RAVDESS files
python src/prepare_data.py --data_root data/raw/ravdess --output data/processed/metadata.csv

# Extract audio features (log-mel, MFCC)
python src/extract_audio_features.py --metadata data/processed/metadata.csv --workers 8 --skip_existing

# Extract face frames from videos
python src/extract_face_frames.py --metadata data/processed/metadata.csv --workers 8 --skip_existing

# Feature analysis for reports
python src/analyze_features.py --metadata data/processed/metadata.csv
```

### Training
```bash
# Audio model: CNN (default), CRNN, Wav2vec2 fine-tune
# Wav2vec2 fine-tune requires --wav2vec2_pretrained pointing to local model dir
python src/train_audio.py --metadata data/processed/metadata.csv \
  --audio_model wav2vec2_finetune \
  --wav2vec2_pretrained /home/ruichao/Workspace/multimodal-emotion-recognition/wav2vec2-base \
  --epochs 20 --batch_size 16 --num_workers 8 --device cuda \
  --early_stopping_patience 8 --output_tag 5class

# Face model (ResNet18) — recommended: use --face_augment
python src/train_face.py --metadata data/processed/metadata.csv \
  --epochs 20 --batch_size 8 --frames_per_sample 4 \
  --num_workers 8 --device cuda --pretrained \
  --face_augment --early_stopping_patience 8 --output_tag 5class_aug
```

### Evaluation
```bash
# Late fusion evaluation (recommended: classwise_weighted_average)
python src/evaluate.py --metadata data/processed/metadata.csv \
  --batch_size 8 --frames_per_sample 4 --num_workers 8 --device cuda \
  --audio_checkpoint outputs/checkpoints/audio_wav2vec2_finetune.5class.pt \
  --face_checkpoint outputs/checkpoints/face_resnet18.5class_aug.pt \
  --wav2vec2_pretrained /home/ruichao/Workspace/multimodal-emotion-recognition/wav2vec2-base \
  --fusion_strategy classwise_weighted_average \
  --output_tag 5class_aug

# Audio-only evaluation
python src/evaluate_audio.py --metadata data/processed/metadata.csv \
  --audio_checkpoint outputs/checkpoints/audio_wav2vec2_finetune.5class.pt \
  --batch_size 16 --num_workers 8 --device cuda
```

### Output Locations
- Checkpoints: `outputs/checkpoints/audio_wav2vec2_finetune.5class.pt`, `outputs/checkpoints/face_resnet18.5class_aug.pt` (recommended)
- Metrics: `outputs/metrics/fusion_metrics.*.json`, `outputs/metrics/test_fusion_predictions.*.csv`
- Figures: `outputs/figures/`

## Architecture

### Models (src/models.py)
- **AudioCNN**: 3-block CNN on log-mel spectrograms (1→32→64→128 channels)
- **AudioCRNN**: CNN + bidirectional GRU with attention pooling
- **AudioWav2Vec2**: Pretrained Wav2Vec2 base + linear pooler; `trainable=True` for fine-tuning
- **FaceResNet**: ResNet18 backbone, frame-level features averaged temporally

### Late Fusion (src/evaluate.py)
Three fusion strategies (recommended: `classwise_weighted_average`):
1. `weighted_average`: Single alpha weight for all classes (grid search 0.0–1.0)
2. `classwise_weighted_average`: Per-class alpha vector, iterative optimization — **best**
3. `confidence_weighted_average`: Per-sample weighting based on prediction confidence

### Dataset (src/dataset.py)
- `MultimodalEmotionDataset`: Handles audio/face/fusion modes
- Audio: Loads pre-extracted log-mel features from `.npz` files
- Face: Loads pre-extracted face frames as JPG; supports `--face_augment` (hflip, brightness, contrast)
- Supports SpecAugment for audio during training

### Config (src/config.py)
Key constants: `LABELS`, `LABEL_TO_ID`, `FACE_IMAGE_SIZE=224`, `FACE_FRAMES_PER_SAMPLE=8`, `N_MELS=64`, `N_MFCC=20`

### Training Utilities (src/train_utils.py)
- `run_epoch`: Handles forward/backward with mixed precision (AMP)
- `detailed_metrics`: Accuracy, Macro F1, per-class precision/recall/f1, confusion matrix
- `save_checkpoint`/`load_checkpoint`: Model state + metrics + model_config

## Development Notes

- Checkpoints store `model_config` dict to auto-detect CNN vs CRNN vs Wav2Vec2 architecture on load
- `--output_tag` suffix avoids overwriting default metric files in follow-up experiments
- `--class_weight` flag for audio training reweights loss by class frequency
- `--specaugment` enables time/frequency masking augmentation for audio
- `--skip_existing` on feature extraction scripts for resumable pipelines
- `--wav2vec2_pretrained` (default `facebook/wav2vec2-base`) must point to local dir in isolated network environments
- `--face_augment` on train_face enables horizontal flip, brightness and contrast jitter for face frames
- Early stopping: `--early_stopping_patience` (default 8) + `--early_stopping_monitor` (`val_loss` or `val_macro_f1`)
- All models use **5 classes**: angry, disgust, fearful, happy, sad (Neutral removed)