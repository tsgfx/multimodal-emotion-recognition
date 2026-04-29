# Multimodal Emotion Recognition

基于 RAVDESS 数据集的**五类多模态情绪识别**，融合语音（Wav2Vec2）和面部表情（ResNet18）两种模态的晚融合方法。

**情绪类别**：angry / disgust / fearful / happy / sad（移除了 Neutral）

**最佳结果**：Test Macro F1 = **0.9334**（classwise 晚融合 + face_aug）

---

## 1. 实验结果

### 单模态模型

| 模型 | Val Macro F1 | 备注 |
| --- | ---: | :--- |
| 音频 Wav2Vec2 fine-tune | 0.8021 | 全量微调，epoch=15 |
| 视频 ResNet18（无增强） | 0.7479 | epoch=19 |
| 视频 ResNet18（+ face_aug） | 0.7768 | hflip/brightness/contrast，epoch=16 |

### 晚融合模型

| 融合策略 | Val Macro F1 | Test Macro F1 |
| --- | ---: | ---: |
| weighted_average | 0.8823 | 0.8575 |
| confidence_weighted_average（β=0.25） | 0.8806 | 0.8987 |
| **classwise_weighted_average（aug face）** | **0.9075** | **0.9334** |

### per-class 分类指标（classwise aug face, Test）

| 情绪 | Precision | Recall | F1 | α_audio | 主导模态 |
| --- | ---: | ---: | ---: | ---: | :--- |
| angry | 0.92 | 0.96 | 0.94 | 0.8 | 音频 |
| disgust | 0.92 | 0.96 | 0.94 | 0.5 | 均衡 |
| fearful | 1.00 | 0.88 | 0.93 | 0.2 | 视觉 |
| happy | 0.96 | 0.92 | 0.94 | 0.5 | 均衡 |
| sad | 0.88 | 0.96 | 0.92 | 0.5 | 均衡 |
| **accuracy** | — | — | **0.9333** | — | — |
| **macro avg** | 0.936 | 0.933 | **0.9334** | — | — |

**关键发现**：
- angry 由语音主导（α=0.8），语音韵律对愤怒情绪区分度最高
- fearful 由面部表情主导（α=0.2），面部表情对恐惧情绪区分度最高
- 其余三类两模态均衡贡献
- face_aug 使最终融合 Test F1 从 0.9248 提升到 0.9334

---

## 2. 配置说明

**数据集**：RAVDESS，24 名演员

**划分**（按演员编号避免身份泄漏）：
- train：actors 01–18（每类 144 样本）
- val：actors 19–21（每类 24 样本）
- test：actors 22–24（每类 24 样本）

**Neutral 被移除**：测试集演员 Neutral 类存在严重 OOD 问题（演员 23 的 neutral 听起来像 sad，演员 24 像 angry），移除后模型更稳定。

---

## 3. 数据准备

下载 RAVDESS 数据后，在项目根目录下执行：

```bash
# 生成元数据 CSV
python src/prepare_data.py --data_root data/raw/ravdess --output data/processed/metadata.csv

# 提取音频 Log-Mel 特征（64 mels，3s，16kHz）
python src/extract_audio_features.py --metadata data/processed/metadata.csv --workers 8 --skip_existing

# 从视频抽帧并检测裁剪人脸（224×224，每视频 8 帧）
python src/extract_face_frames.py --metadata data/processed/metadata.csv --workers 8 --skip_existing

# 特征可视化分析
python src/analyze_features.py --metadata data/processed/metadata.csv
```

特征输出目录：
- `data/processed/audio_features/` — `.npz` 文件含 log_mel
- `data/processed/face_frames/<sample_id>/` — 人脸 JPG 帧
- `data/processed/metadata.csv` — 所有样本元数据

---

## 4. 模型训练

### 音频模型（Wav2Vec2 fine-tune）

需要先下载 Wav2Vec2-base 到本地，通过 `--wav2vec2_pretrained` 指定路径：

```bash
python src/train_audio.py \
  --metadata data/processed/metadata.csv \
  --audio_model wav2vec2_finetune \
  --wav2vec2_pretrained /path/to/wav2vec2-base \
  --epochs 20 --batch_size 16 --num_workers 8 --device cuda \
  --early_stopping_patience 8 --output_tag 5class
```

- 学习率：编码器 1e-5，分类头 1e-4
- 优化器：AdamW（weight_decay=1e-4）
- 早停：监控 `val_loss`，可改为 `--early_stopping_monitor val_macro_f1`
- 梯度裁剪：最大范数 1.0

### 人脸模型（ResNet18 + face_aug）

```bash
python src/train_face.py \
  --metadata data/processed/metadata.csv \
  --epochs 20 --batch_size 8 --frames_per_sample 4 \
  --num_workers 8 --device cuda --pretrained \
  --face_augment --early_stopping_patience 8 --output_tag 5class_aug
```

- 帧级特征时序平均池化
- face_augment：horizontal flip、brightness/contrast jitter
- 推荐启用 `--pretrained` 使用 ImageNet 预训练权重

---

## 5. 晚融合评估

### 推荐配置（classwise_weighted_average）

```bash
python src/evaluate.py \
  --metadata data/processed/metadata.csv \
  --batch_size 8 --frames_per_sample 4 --num_workers 8 --device cuda \
  --audio_checkpoint outputs/checkpoints/audio_wav2vec2_finetune.5class.pt \
  --face_checkpoint outputs/checkpoints/face_resnet18.5class_aug.pt \
  --wav2vec2_pretrained /path/to/wav2vec2-base \
  --fusion_strategy classwise_weighted_average \
  --output_tag 5class_aug
```

支持三种融合策略：
- `weighted_average` — 全局单一 α（grid search 0.0–1.0）
- `classwise_weighted_average` — **推荐**，per-class α 向量在验证集上独立优化
- `confidence_weighted_average` — 基于预测置信度的样本级加权（β=0.25）

融合权重（classwise）：
| 情绪 | α_audio | α_face | 主导模态 |
| --- | --- | --- | :--- |
| angry | 0.8 | 0.2 | 音频 |
| fearful | 0.2 | 0.8 | 视觉 |
| 其他 | 0.5 | 0.5 | 均衡 |

---

## 6. 交互式 Demo

启动 Gradio 可视化界面（需音频和视频模型权重）：

```bash
python src/demo.py \
  --audio_checkpoint outputs/checkpoints/audio_wav2vec2_finetune.5class.pt \
  --face_checkpoint outputs/checkpoints/face_resnet18.5class_aug.pt \
  --wav2vec2_pretrained /path/to/wav2vec2-base \
  --share
```

**参数说明：**

| 参数 | 默认值 | 说明 |
| --- | --- | :--- |
| `--audio_checkpoint` | `outputs/checkpoints/audio_wav2vec2_finetune.5class.pt` | 音频模型权重路径 |
| `--face_checkpoint` | `outputs/checkpoints/face_resnet18.5class_aug.pt` | 人脸模型权重路径 |
| `--wav2vec2_pretrained` | `wav2vec2-base` | Wav2Vec2 本地模型目录或 HF id |
| `--device` | `auto` | 设备：`auto` / `cuda` / `cpu` |
| `--port` | `7860` | Gradio 服务端口 |
| `--share` | — | 生成临时公开链接（7 天有效期） |

**样例文件**在 `data/example/`（各情绪的 WAV + MP4），可直接上传测试：

```
data/example/angry.wav   data/example/angry.mp4
data/example/disgust.wav data/example/disgust.mp4
data/example/fearful.wav data/example/fearful.mp4
data/example/happy.wav   data/example/happy.mp4
data/example/sad.wav     data/example/sad.mp4
```

**使用方式**：上传音频（支持 WAV/MP3，支持麦克风录音）和/或视频（支持 MP4/AVI），点击「开始识别」，分别输出音频、视频单模态概率及融合结果。

融合结果说明：
- 仅音频 → 使用 Wav2Vec2 fine-tune 模型预测
- 仅视频 → 使用 ResNet18 人脸模型预测
- 两者都有 → classwise 加权融合（α_audio: angry=0.8, fearful=0.2, 其余=0.5）

---

## 7. 项目结构

```
multimodal-emotion-recognition/
├── data/
│   ├── raw/ravdess/               # 原始 RAVDESS 数据（需下载）
│   ├── processed/
│   │   ├── metadata.csv            # 元数据（含 split/label/路径）
│   │   ├── audio_features/        # 音频 Log-Mel 特征
│   │   └── face_frames/           # 人脸帧
│   └── example/                   # 样例音视频（WAV + MP4）
├── src/
│   ├── prepare_data.py            # 解析 RAVDESS 文件名生成元数据
│   ├── extract_audio_features.py  # 提取 Log-Mel、MFCC 特征
│   ├── extract_face_frames.py     # 视频抽帧 + 人脸检测裁剪
│   ├── analyze_features.py        # 特征分析可视化
│   ├── train_audio.py             # 音频模型训练（Wav2Vec2 fine-tune）
│   ├── train_face.py             # 人脸模型训练（ResNet18 + face_aug）
│   ├── evaluate.py               # 晚融合评估（三种策略）
│   ├── models.py                 # 模型定义（CNN/CRNN/Wav2Vec2/ResNet18）
│   ├── dataset.py                # Dataset（audio/face/fusion 模式）
│   ├── train_utils.py           # 训练工具（AMP/早停/指标）
│   ├── config.py                # 配置（LABELS=5 类常量）
│   ├── utils.py                 # 工具函数
│   └── demo.py                  # Gradio 交互式演示
├── outputs/
│   ├── checkpoints/              # 模型权重
│   │   ├── audio_wav2vec2_finetune.5class.pt
│   │   └── face_resnet18.5class_aug.pt
│   ├── metrics/                  # 评估指标 JSON + CSV
│   │   └── fusion_metrics.5class_aug.json
│   └── figures/                  # 特征分析图
├── CLAUDE.md                     # Claude Code 开发指南
├── README.md                    # 本文件
├── requirements.txt
└── wav2vec2-base/               # 本地 Wav2Vec2 模型（需下载）
```

---

## 8. 模型架构

### 音频分支：Wav2Vec2 Fine-tune

```
Raw audio (16kHz) → Wav2Vec2 encoder → mean pooling → Dropout → Linear classifier
```

使用 `facebook/wav2vec2-base` 全量微调，编码器学习率 1e-5，分类头 1e-4。

### 视觉分支：FaceResNet

```
Face frames → ResNet18 backbone → frame embeddings → temporal mean pooling → classifier
```

ImageNet 预训练 ResNet18，backbone 冻结，只训练分类头。训练时启用 face_augment（hflip、brightness、contrast jitter）。

### 晚融合：Classwise Weighted Average

```
final_prob[c] = α_audio[c] × audio_prob[c] + (1 - α_audio[c]) × face_prob[c]
```

每类权重在验证集上独立优化，允许不同模态对不同情绪发挥不同主导作用。

---

## 9. 关键模型文件

| 文件 | 说明 |
| --- | :--- |
| `outputs/checkpoints/audio_wav2vec2_finetune.5class.pt` | 音频 Wav2Vec2 全量微调模型 |
| `outputs/checkpoints/face_resnet18.5class_aug.pt` | 人脸 ResNet18（+ face_aug）**推荐** |
| `outputs/metrics/fusion_metrics.5class_aug.json` | 晚融合完整指标 |
| `data/example/` | 5 类情绪样例音视频（angry/disgust/fearful/happy/sad） |

---

## 10. 注意事项

- `neutral` 类因 OOD 问题被移除，配置在 `src/config.py` 的 `LABELS` 中
- Wav2Vec2 需下载到本地并通过 `--wav2vec2_pretrained` 指定路径
- 早停默认监控 `val_loss`，可改为 `--early_stopping_monitor val_macro_f1`
- `--output_tag` 用于区分不同实验结果，避免覆盖已有文件
- Gradio Demo 默认端口 7860，`--share` 生成临时公开链接
- 样例文件路径：`data/example/angry.wav`、`data/example/angry.mp4` 等