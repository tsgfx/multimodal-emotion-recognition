# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multimodal Emotion Recognition using RAVDESS dataset. Implements audio-only and face-only single-modality models, plus late fusion evaluation combining both modalities. Emotion classes: angry, disgust, fearful, happy, neutral, sad.

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
# Audio model (CNN by default, or CRNN with --audio_model crnn)
python src/train_audio.py --metadata data/processed/metadata.csv --epochs 20 --batch_size 64 --num_workers 8 --device cuda

# Face model (ResNet18)
python src/train_face.py --metadata data/processed/metadata.csv --epochs 15 --batch_size 8 --frames_per_sample 4 --num_workers 8 --device cuda --pretrained
```

### Evaluation
```bash
# Late fusion evaluation (default: weighted_average)
python src/evaluate.py --metadata data/processed/metadata.csv --batch_size 8 --frames_per_sample 4 --num_workers 8 --device cuda

# Audio-only evaluation
python src/evaluate_audio.py --metadata data/processed/metadata.csv --audio_checkpoint outputs/checkpoints/audio_cnn.pt --batch_size 64 --num_workers 8 --device cuda
```

### Output Locations
- Checkpoints: `outputs/checkpoints/audio_cnn.pt`, `outputs/checkpoints/face_resnet18.pt`
- Metrics: `outputs/metrics/fusion_metrics.json`, `outputs/metrics/test_fusion_predictions.csv`
- Figures: `outputs/figures/`

## Architecture

### Models (src/models.py)
- **AudioCNN**: 3-block CNN on log-mel spectrograms (1→32→64→128 channels)
- **AudioCRNN**: CNN + bidirectional GRU with attention pooling
- **FaceResNet**: ResNet18 backbone, frame-level features averaged temporally

### Late Fusion (src/evaluate.py)
Three fusion strategies:
1. `weighted_average`: Single alpha weight for all classes (grid search 0.0–1.0)
2. `classwise_weighted_average`: Per-class alpha vector (iterative optimization)
3. `confidence_weighted_average`: Per-sample weighting based on prediction confidence

### Dataset (src/dataset.py)
- `MultimodalEmotionDataset`: Handles audio/face/fusion modes
- Audio: Loads pre-extracted log-mel features from `.npz` files
- Face: Loads pre-extracted face frames as JPG
- Supports SpecAugment for audio during training

### Config (src/config.py)
Key constants: `LABELS`, `LABEL_TO_ID`, `FACE_IMAGE_SIZE=224`, `FACE_FRAMES_PER_SAMPLE=8`, `N_MELS=64`, `N_MFCC=20`

### Training Utilities (src/train_utils.py)
- `run_epoch`: Handles forward/backward with mixed precision (AMP)
- `detailed_metrics`: Accuracy, Macro F1, per-class precision/recall/f1, confusion matrix
- `save_checkpoint`/`load_checkpoint`: Model state + metrics + model_config

## Development Notes

- Checkpoints store `model_config` dict to auto-detect CNN vs CRNN architecture on load
- `--output_tag` suffix avoids overwriting default metric files in follow-up experiments
- `--class_weight` flag for audio training reweights loss by class frequency
- `--specaugment` enables time/frequency masking augmentation for audio
- `--skip_existing` on feature extraction scripts for resumable pipelines
