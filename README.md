# Multimodal Emotion Recognition

本项目实现 RAVDESS 六类多模态情绪识别，使用语音和面部表情两种模态，覆盖数据下载、预处理、特征分析、单模态训练和晚融合评估。

默认情绪类别：

```text
angry / disgust / fearful / happy / neutral / sad
```

## 运行环境

推荐在带 NVIDIA GPU 的 Linux 服务器上运行。

建议环境：

```text
Python 3.9+
CUDA 11.8 或 CUDA 12.x
PyTorch + torchvision
```

创建环境示例：

```bash
conda create -n mmer python=3.9 -y
conda activate mmer
```

安装 GPU 版 PyTorch。以下以 CUDA 11.8 为例：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

如果服务器是 CUDA 12.1，可改用：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

检查 GPU 是否可用：

```bash
python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY
```

## 数据集说明

本项目使用 RAVDESS：

```text
https://zenodo.org/records/1188976
```

需要下载：

```text
Audio_Speech_Actors_01-24.zip
Video_Speech_Actor_01.zip
...
Video_Speech_Actor_24.zip
```

预计磁盘占用：

```text
压缩包：约 13G
解压后 data/raw/ravdess：约 26G
处理中间结果和模型输出：额外数 GB
```

本仓库不提交数据集、处理特征、模型权重和实验输出。这些路径已在 `.gitignore` 中忽略：

```text
data/raw/
data/processed/
outputs/checkpoints/
outputs/figures/
outputs/metrics/
```

## 服务器下载数据集

进入项目根目录：

```bash
cd multimodal-emotion-recognition
mkdir -p data/raw/ravdess
```

下载 speech 音频包：

```bash
curl -L --fail --continue-at - \
  --output data/raw/ravdess/Audio_Speech_Actors_01-24.zip \
  'https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1'
```

下载 24 个 speech 视频包：

```bash
curl --parallel --parallel-max 4 -L --fail --continue-at - \
  --output 'data/raw/ravdess/Video_Speech_Actor_#1.zip' \
  'https://zenodo.org/records/1188976/files/Video_Speech_Actor_[01-24].zip?download=1'
```

如果服务器网络不稳定，可以把 `--parallel-max 4` 调低到 `2`，并重复执行同一命令继续断点续传。

校验音频包 MD5：

```bash
md5sum data/raw/ravdess/Audio_Speech_Actors_01-24.zip
```

期望值：

```text
bc696df654c87fed845eb13823edef8a
```

检查视频 zip 完整性：

```bash
for zip in data/raw/ravdess/Video_Speech_Actor_*.zip; do
  echo "testing $zip"
  unzip -tq "$zip" >/dev/null || exit 1
done
echo "all video zip files passed integrity test"
```

解压：

```bash
unzip -q -n data/raw/ravdess/Audio_Speech_Actors_01-24.zip -d data/raw/ravdess

for zip in data/raw/ravdess/Video_Speech_Actor_*.zip; do
  echo "extracting $zip"
  unzip -q -n "$zip" -d data/raw/ravdess || exit 1
done
```

检查文件数量：

```bash
find data/raw/ravdess -type f -name '*.wav' | wc -l
find data/raw/ravdess -type f -name '*.mp4' | wc -l
du -sh data/raw/ravdess
```

参考结果：

```text
wav 文件数：1440
mp4 文件数：2880
data/raw/ravdess：约 26G
```

## 生成元数据

```bash
python src/prepare_data.py \
  --data_root data/raw/ravdess \
  --output data/processed/metadata.csv
```

参考六类样本分布：

```text
split    angry  disgust  fearful  happy  neutral  sad
train      144      144      144    144       72  144
val         24       24       24     24       12   24
test        24       24       24     24       12   24
```

说明：

- 当前使用 RAVDESS speech 音频 + speech 视频。
- 默认按 actor 划分 train / val / test。
- `neutral` 在 RAVDESS 中样本较少，因此数量低于其他类别。

## 特征提取

提取音频特征：

```bash
python src/extract_audio_features.py \
  --metadata data/processed/metadata.csv \
  --workers 8 \
  --skip_existing
```

抽取人脸帧：

```bash
python src/extract_face_frames.py \
  --metadata data/processed/metadata.csv \
  --workers 8 \
  --skip_existing
```

说明：

- 特征提取阶段主要受视频解码、OpenCV 人脸检测、librosa 音频处理和磁盘 I/O 限制，默认不直接使用 GPU。
- 服务器上建议优先用 `--workers` 做多进程并行；常用取值为 `4`、`8` 或 `$(nproc)` 的一半。
- `--skip_existing` 会跳过已经生成完成的样本，适合中断后续跑。
- GPU 主要用于后续 `train_audio.py`、`train_face.py` 和 `evaluate.py`。

生成报告用特征分析图：

```bash
python src/analyze_features.py --metadata data/processed/metadata.csv
```

输出位置：

```text
data/processed/audio_features/
data/processed/face_frames/
outputs/figures/
```

## GPU 训练

训练语音单模态模型：

```bash
python src/train_audio.py \
  --metadata data/processed/metadata.csv \
  --epochs 20 \
  --batch_size 64 \
  --num_workers 8 \
  --device cuda
```

切换到更强的 `CRNN` 语音骨干：

```bash
python src/train_audio.py \
  --metadata data/processed/metadata.csv \
  --epochs 20 \
  --batch_size 64 \
  --num_workers 8 \
  --device cuda \
  --audio_model crnn \
  --class_weight \
  --output_tag crnn_exp1
```

如果需要先增强语音分支，可在训练时启用类别重加权和 SpecAugment：

```bash
python src/train_audio.py \
  --metadata data/processed/metadata.csv \
  --epochs 20 \
  --batch_size 64 \
  --num_workers 8 \
  --device cuda \
  --class_weight \
  --specaugment
```

`--weighted_sampler` 也可用，但更建议与 `--class_weight` 分开做对照实验，避免同时对少数类做两次放大。

如果需要连续跑多组实验而不覆盖默认文件，可加 `--output_tag`：

```bash
python src/train_audio.py \
  --metadata data/processed/metadata.csv \
  --epochs 30 \
  --batch_size 64 \
  --num_workers 8 \
  --device cuda \
  --class_weight \
  --output_tag class_weight_ep30
```

当前本地 follow-up 实验中，`--class_weight` 是最稳的语音分支配置；继续把训练拉长到 30 epoch 虽然能提升验证集分数，但没有稳定超过当前最优融合结果。

训练视觉单模态模型。GPU 显存较小时先用 `batch_size 4`，显存充足可用 `8` 或 `16`；如果单步仍然慢，先用 `--frames_per_sample 4`：

```bash
python src/train_face.py \
  --metadata data/processed/metadata.csv \
  --epochs 15 \
  --batch_size 8 \
  --frames_per_sample 4 \
  --num_workers 8 \
  --device cuda \
  --pretrained
```

如果服务器无法下载 torchvision 预训练权重，去掉 `--pretrained`：

```bash
python src/train_face.py \
  --metadata data/processed/metadata.csv \
  --epochs 15 \
  --batch_size 8 \
  --frames_per_sample 4 \
  --num_workers 8 \
  --device cuda
```

启动训练时会打印 `torch` 版本、`cuda_available` 和实际 `device`。如果显示 `cuda_available=False` 或 `device=cpu`，说明当前环境安装的是 CPU 版 PyTorch，`nvidia-smi` 看不到训练进程是正常的，需要重新安装 CUDA 版 PyTorch。

输出权重：

```text
outputs/checkpoints/audio_cnn.pt
outputs/checkpoints/face_resnet18.pt
```

## 晚融合评估

```bash
python src/evaluate.py \
  --metadata data/processed/metadata.csv \
  --batch_size 8 \
  --frames_per_sample 4 \
  --num_workers 8 \
  --device cuda
```

评估指定语音 checkpoint 的单模态效果：

```bash
python src/evaluate_audio.py \
  --metadata data/processed/metadata.csv \
  --audio_checkpoint outputs/checkpoints/audio_cnn.class_weight_ep30.pt \
  --batch_size 64 \
  --num_workers 8 \
  --device cuda \
  --output_tag class_weight_ep30
```

如果 checkpoint 是 `CRNN`，评估脚本默认会自动读取保存时的模型配置，不需要手动再传 `--audio_model crnn`。需要覆盖时也可以显式指定：

```bash
python src/evaluate_audio.py \
  --metadata data/processed/metadata.csv \
  --audio_checkpoint outputs/checkpoints/audio_crnn.crnn_exp1.pt \
  --audio_model crnn \
  --batch_size 64 \
  --num_workers 8 \
  --device cuda \
  --output_tag crnn_exp1
```

主要输出：

```text
outputs/metrics/audio_val_metrics.json
outputs/metrics/face_val_metrics.json
outputs/metrics/fusion_metrics.json
outputs/metrics/audio_val_metrics.<tag>.json
outputs/metrics/audio_test_metrics.<tag>.json
outputs/metrics/test_fusion_predictions.csv
outputs/metrics/test_fusion_confusion_matrix.csv
```

必须报告三组实验：

- Audio-only
- Face-only
- Audio + Face late fusion

评价指标：

- Accuracy
- Macro F1-score
- 每类 Precision、Recall、F1-score
- Confusion Matrix

## 推荐完整流程

服务器从零运行时，按以下顺序执行：

```bash
conda create -n mmer python=3.9 -y
conda activate mmer

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

mkdir -p data/raw/ravdess

curl -L --fail --continue-at - \
  --output data/raw/ravdess/Audio_Speech_Actors_01-24.zip \
  'https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1'

curl --parallel --parallel-max 4 -L --fail --continue-at - \
  --output 'data/raw/ravdess/Video_Speech_Actor_#1.zip' \
  'https://zenodo.org/records/1188976/files/Video_Speech_Actor_[01-24].zip?download=1'

unzip -q -n data/raw/ravdess/Audio_Speech_Actors_01-24.zip -d data/raw/ravdess

for zip in data/raw/ravdess/Video_Speech_Actor_*.zip; do
  unzip -q -n "$zip" -d data/raw/ravdess || exit 1
done

python src/prepare_data.py --data_root data/raw/ravdess --output data/processed/metadata.csv
python src/extract_audio_features.py --metadata data/processed/metadata.csv --workers 8 --skip_existing
python src/extract_face_frames.py --metadata data/processed/metadata.csv --workers 8 --skip_existing
python src/analyze_features.py --metadata data/processed/metadata.csv
python src/train_audio.py --metadata data/processed/metadata.csv --epochs 20 --batch_size 64 --num_workers 8 --device cuda
python src/train_face.py --metadata data/processed/metadata.csv --epochs 15 --batch_size 8 --frames_per_sample 4 --num_workers 8 --device cuda --pretrained
python src/evaluate.py --metadata data/processed/metadata.csv --batch_size 8 --frames_per_sample 4 --num_workers 8 --device cuda
```

## 项目文件

核心脚本：

- `src/prepare_data.py`：解析 RAVDESS 文件名，生成六类元数据。
- `src/extract_audio_features.py`：提取 Log-Mel、MFCC 和统计声学特征。
- `src/extract_face_frames.py`：从视频中抽帧并裁剪人脸。
- `src/analyze_features.py`：生成报告用特征分析图。
- `src/train_audio.py`：训练语音单模态模型。
- `src/train_face.py`：训练视觉单模态模型。
- `src/evaluate.py`：执行晚融合评估。

文档：

- `PLAN.md`：项目执行计划。
- `reports/report.md`：课程报告模板。
- `reports/slides_outline.md`：PPT 提纲。
