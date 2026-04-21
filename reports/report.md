# 多模态用户情绪状态识别报告

## 1. 研究背景与问题定义

情绪识别在人机交互、智能客服、在线教育和心理健康辅助分析中具有重要应用价值。单一模态容易受到噪声、遮挡或个体差异影响，因此本项目综合语音和面部表情两种模态，完成用户情绪状态识别。

本项目将任务建模为六分类问题，类别包括：angry、disgust、fearful、happy、neutral、sad。

## 2. 数据集说明

本项目使用 RAVDESS 数据集，筛选包含音频和视频信息的样本。RAVDESS 的文件名包含情绪、强度、语句、重复次数和演员编号等信息，便于解析标签和按演员划分数据。

需补充实验后统计：

| split | angry | disgust | fearful | happy | neutral | sad |
| --- | --- | --- | --- | --- | --- | --- |
| train | TBD | TBD | TBD | TBD | TBD | TBD |
| val | TBD | TBD | TBD | TBD | TBD | TBD |
| test | TBD | TBD | TBD | TBD | TBD | TBD |

## 3. 数据预处理

语音模态：

- 统一采样率为 16 kHz。
- 音频裁剪或补零到固定长度。
- 提取 Log-Mel spectrogram 作为模型输入。
- 提取 MFCC、能量、过零率、谱质心等统计特征用于分析。

视觉模态：

- 从每个视频均匀抽取若干帧。
- 使用 OpenCV Haar cascade 检测人脸。
- 检测失败时使用中心裁剪作为兜底。
- 将人脸图像缩放到 224x224。

## 4. 特征分析

需插入以下图表：

- `outputs/figures/audio_energy_boxplot.png`
- `outputs/figures/mfcc_mean_heatmap.png`
- `outputs/figures/log_mel_examples.png`
- `outputs/figures/face_examples_grid.png`
- `outputs/figures/face_pca.png`

分析要点：

- angry 和 happy 往往能量较高。
- sad 和 neutral 在声学上可能更平稳，容易混淆。
- fearful 和 disgust 可能与 angry 或 sad 在部分声学特征上重叠。
- 面部表情能够为语音不明显的样本提供补充信息。

## 5. 模型设计

语音分支使用 Log-Mel spectrogram + CNN：

```text
Log-Mel spectrogram -> 2D CNN -> Global Average Pooling -> Linear classifier
```

视觉分支使用人脸帧 + ResNet18：

```text
Face frames -> ResNet18 -> frame logits average -> Linear classifier
```

多模态融合采用晚融合：

```text
final_prob = alpha * audio_prob + (1 - alpha) * face_prob
```

其中 alpha 在验证集上搜索，并在测试集上固定使用。

## 6. 实验设置

评价指标：

- Accuracy
- Macro F1-score
- 每类 Precision、Recall、F1-score
- Confusion Matrix

对比实验：

- Audio-only
- Face-only
- Audio + Face late fusion

## 7. 实验结果

需补充训练后的结果：

| 方法 | Accuracy | Macro F1 |
| --- | --- | --- |
| Audio-only | TBD | TBD |
| Face-only | TBD | TBD |
| Late fusion | TBD | TBD |

需插入混淆矩阵：

- `outputs/metrics/test_fusion_confusion_matrix.csv`

## 8. 结果分析

需根据实验结果补充：

- 哪些情绪类别最容易识别。
- 哪些情绪类别最容易混淆。
- 多模态融合相比单模态是否提升。
- 语音和视觉模态各自的优势与不足。

## 9. 总结与展望

本项目完成了语音和面部表情两种模态的情绪识别流程，包括数据预处理、特征分析、单模态建模、多模态晚融合和实验评估。

后续可改进方向：

- 使用更强的预训练语音模型。
- 使用更稳定的人脸检测与表情特征提取方法。
- 尝试中间融合或注意力融合。
- 使用 CREMA-D 作为外部泛化测试。
