# Multimodal Emotion Recognition

基于 RAVDESS 数据集的**五类多模态情绪识别**，使用语音（Wav2Vec2）和面部表情（ResNet18）两种模态，结合晚融合实现高精度情绪分类。

情绪类别：**angry / disgust / fearful / happy / sad**（移除了 Neutral）

当前最佳结果：**Test Macro F1 = 0.9334**（classwise 晚融合）

## 运行环境

推荐在带 NVIDIA GPU 的 Linux 服务器上运行。

```bash
conda create -n mmer python=3.9 -y && conda activate mmer

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# 或 CUDA 12.x
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## 项目文件结构

```
multimodal-emotion-recognition/
├── src/
│   ├── prepare_data.py          # 解析 RAVDESS 文件名，生成元数据
│   ├── extract_audio_features.py # 提取 Log-Mel、MFCC 特征
│   ├── extract_face_frames.py    # 从视频抽帧并裁剪人脸
│   ├── analyze_features.py       # 生成特征分析图
│   ├── train_audio.py            # 训练语音模型（Wav2Vec2 fine-tune）
│   ├── train_face.py             # 训练人脸模型（ResNet18）
│   ├── evaluate.py               # 晚融合评估
│   ├── evaluate_audio.py         # 语音单模态评估
│   ├── models.py                 # 模型定义
│   ├── dataset.py                # 数据集（支持 face_aug, specaugment）
│   ├── train_utils.py            # 训练工具（早停、AMP、指标）
│   ├── config.py                 # 配置（LABELS=5 类）
│   └── utils.py                  # 工具函数
├── data/
│   ├── raw/ravdess/             # 原始 RAVDESS 数据（被 gitignore）
│   └── processed/
│       ├── metadata.csv          # 元数据
│       ├── audio_features/        # 音频特征
│       └── face_frames/          # 人脸帧
├── outputs/
│   ├── checkpoints/              # 模型权重
│   ├── metrics/                 # 指标文件
│   └── figures/                 # 分析图
├── PLAN.md                      # 实验记录（最新结果）
├── CLAUDE.md                     # Claude Code 指南
└── README.md                    # 本文件
```

## 数据准备

下载 RAVDESS 数据后：

```bash
# 生成元数据
python src/prepare_data.py --data_root data/raw/ravdess --output data/processed/metadata.csv

# 提取特征
python src/extract_audio_features.py --metadata data/processed/metadata.csv --workers 8 --skip_existing
python src/extract_face_frames.py --metadata data/processed/metadata.csv --workers 8 --skip_existing

# 特征分析
python src/analyze_features.py --metadata data/processed/metadata.csv
```

## 模型训练

```bash
# 语音模型（Wav2Vec2 fine-tune，需要本地模型路径）
python src/train_audio.py \
  --metadata data/processed/metadata.csv \
  --audio_model wav2vec2_finetune \
  --wav2vec2_pretrained /path/to/wav2vec2-base \
  --epochs 20 --batch_size 16 --num_workers 8 --device cuda \
  --early_stopping_patience 8 --output_tag 5class

# 人脸模型（推荐加 --face_augment）
python src/train_face.py \
  --metadata data/processed/metadata.csv \
  --epochs 20 --batch_size 8 --frames_per_sample 4 \
  --num_workers 8 --device cuda --pretrained \
  --face_augment --early_stopping_patience 8 --output_tag 5class_aug
```

## 晚融合评估

```bash
# 推荐配置：classwise_weighted_average + aug face
python src/evaluate.py \
  --metadata data/processed/metadata.csv \
  --batch_size 8 --frames_per_sample 4 --num_workers 8 --device cuda \
  --audio_checkpoint outputs/checkpoints/audio_wav2vec2_finetune.5class.pt \
  --face_checkpoint outputs/checkpoints/face_resnet18.5class_aug.pt \
  --wav2vec2_pretrained /path/to/wav2vec2-base \
  --fusion_strategy classwise_weighted_average \
  --output_tag 5class_aug
```

支持三种融合策略：`weighted_average`、`classwise_weighted_average`（推荐）、`confidence_weighted_average`。

## 实验结果摘要

| 模型 | Val Macro F1 | Test Macro F1 |
| --- | ---: | ---: |
| 音频 Wav2Vec2 fine-tune | 0.8021 | — |
| 视频 ResNet18（无增强） | 0.7479 | — |
| 视频 ResNet18（+aug） | 0.7768 | — |
| **晚融合 classwise（aug face）** | **0.9075** | **0.9334** |

详细结果见 `PLAN.md`。

## 注意事项

- `neutral` 类因 OOD 问题被移除，配置在 `src/config.py` 的 `LABELS` 中
- Wav2Vec2 需要通过 `hf download facebook/wav2vec2-base --local-dir /path/to/local` 下载到本地
- 早停监控默认 `val_loss`，可改为 `--early_stopping_monitor val_macro_f1`
- `--output_tag` 用于避免覆盖已有实验结果