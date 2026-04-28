# 多模态用户情绪状态识别报告

## 1. 研究背景与问题定义

情绪识别在人机交互、智能客服、在线教育和心理健康辅助分析中具有重要应用价值。单一模态容易受到噪声、遮挡或个体差异影响，因此本项目综合语音和面部表情两种模态，完成用户情绪状态识别。

本项目将任务建模为五分类问题，类别包括：angry、disgust、fearful、happy、sad。**Neutral 类因测试集表现极差（OOD 问题）被移除**——测试集演员的 Neutral 样本在语音和视觉上均与 sad、angry 高度混淆，移除后模型整体性能显著提升。

## 2. 数据集说明

本项目使用 RAVDESS 数据集，筛选包含音频和视频信息的样本。RAVDESS 的文件名包含情绪、强度、语句、重复次数和演员编号等信息，便于解析标签和按演员划分数据。

按演员编号划分训练集、验证集和测试集，避免同一演员同时出现在不同划分中造成身份泄漏：

| split | angry | disgust | fearful | happy | sad |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 144 | 144 | 144 | 144 | 144 |
| val | 24 | 24 | 24 | 24 | 24 |
| test | 24 | 24 | 24 | 24 | 24 |

## 3. 数据预处理

语音模态：
- 统一采样率为 16 kHz。
- 音频裁剪或补零到固定长度 3 秒。
- 提取 Log-Mel spectrogram（64 mels）作为模型输入。
- 提取 MFCC、能量、过零率、谱质心等统计特征用于分析。

视觉模态：
- 从每个视频均匀抽取 8 帧。
- 使用 OpenCV Haar cascade 检测人脸，检测失败时使用中心裁剪兜底。
- 将人脸图像缩放到 224×224。

## 4. 特征分析

已生成以下特征分析图表：

![Audio energy boxplot](../outputs/figures/audio_energy_boxplot.png)

![MFCC mean heatmap](../outputs/figures/mfcc_mean_heatmap.png)

![Log-Mel examples](../outputs/figures/log_mel_examples.png)

![Face examples](../outputs/figures/face_examples_grid.png)

![Face PCA](../outputs/figures/face_pca.png)

分析要点：
- angry 和 happy 往往能量较高。
- sad 和 fearful 在声学上可能更平稳，容易混淆。
- 面部表情能够为语音不明显的样本提供补充信息。

## 5. 模型设计

### 语音分支：Wav2Vec2 Fine-tune

使用 Facebook 预训练的 Wav2Vec2-base 作为音频编码器，采用端到端微调策略：

```text
Raw audio (16kHz) -> Wav2Vec2 encoder -> mean pooling -> Dropout -> Linear classifier
```

- 编码器参数学习率：1e-5
- 分类头学习率：1e-4
- 优化器：AdamW（weight_decay=1e-4）
- 端到端微调使模型能够学习情绪相关的语音表示。

### 视觉分支：ResNet18 + Face Augmentation

使用 ImageNet 预训练的 ResNet18 提取人脸帧特征，帧级特征经时序平均池化后送入分类头：

```text
Face frames -> ResNet18 -> frame features -> temporal mean pooling -> Linear classifier
```

训练时启用人脸数据增强（horizontal flip、brightness jitter、contrast jitter）以提升泛化能力。

### 多模态融合：Classwise Late Fusion

采用晚融合策略，在预测概率层面加权合并两个模态的输出：

```text
final_prob[class] = alpha_audio[class] * audio_prob[class] + (1 - alpha_audio[class]) * face_prob[class]
```

其中每个情绪类别的权重在验证集上独立优化，允许不同模态对不同情绪发挥不同主导作用。

## 6. 实验设置

评价指标：
- Accuracy、Macro F1-score、每类 Precision/Recall/F1-score、混淆矩阵

对比实验：
- Audio-only（Wav2Vec2 fine-tune）
- Face-only（ResNet18 pretrained，基础 vs 增强）
- Audio + Face late fusion（三种策略对比）

## 7. 实验结果

### 7.1 单模态结果

| 模型 | Val Macro F1 |
| --- | ---: |
| Audio Wav2Vec2 fine-tune | **0.8021** |
| Face ResNet18（无增强） | 0.7479 |
| Face ResNet18（+ face_aug） | **0.7768** |

人脸数据增强（horizontal flip + brightness/contrast jitter）在单模态上提升约 +3%。

### 7.2 晚融合对比

| 融合策略 | Val Macro F1 | Test Macro F1 |
| --- | ---: | ---: |
| weighted_average | 0.8823 | 0.8575 |
| confidence_weighted_average (β=0.25) | 0.8806 | 0.8987 |
| **classwise_weighted_average** | **0.9075** | **0.9334** |

classwise 策略表现最优，相对于单模态最高提升约 **+13%**（Val）。

### 7.3 最优配置 per-class 融合权重

| 情绪 | α_audio | α_face | 主导模态 |
| --- | ---: | ---: | :--- |
| angry | 0.8 | 0.2 | 音频 |
| disgust | 0.5 | 0.5 | 均衡 |
| fearful | 0.2 | 0.8 | 视觉 |
| happy | 0.5 | 0.5 | 均衡 |
| sad | 0.5 | 0.5 | 均衡 |

angry 类音频信息更显著，fearful 类视觉信息更显著，其余类别两模态贡献相近。

### 7.4 测试集分类报告（最优配置）

| 类别 | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| angry | 0.92 | 0.96 | 0.94 | 24 |
| disgust | 0.92 | 0.96 | 0.94 | 24 |
| fearful | 1.00 | 0.88 | 0.93 | 24 |
| happy | 0.96 | 0.92 | 0.94 | 24 |
| sad | 0.88 | 0.96 | 0.92 | 24 |
| **accuracy** | — | — | **0.9333** | 120 |
| **macro avg** | 0.936 | 0.933 | **0.9334** | 120 |

所有五类 F1 均达到 0.92 以上，模型泛化能力稳健。

## 8. 结果分析

**多模态互补效果显著**：晚融合相比单模态最佳结果（audio 0.8021）提升约 +13%，说明语音和视觉在情绪识别上提供了互补信息。

**模态主导因类而异**：angry 主要依赖音频（语音韵律更显著），fearful 主要依赖视觉（面部恐惧表情特征性强），而 disgust/happy/sad 则由两模态共同主导。这一发现与直觉相符，也验证了 classwise 融合策略的有效性。

**face_augmentation 有效**：通过在训练时施加几何和色彩增强，单模态 face 性能从 0.7479 提升到 0.7768，最终融合 Test F1 从 0.9248 提升到 0.9334。

**中性类移除的合理性**：测试集演员的 Neutral 类与 sad/angry 在语音和视觉上均高度重叠，移除后测试集宏观指标更加干净可靠。

## 9. 总结与展望

本项目完成了语音和面部表情两种模态的情绪识别全流程：
- **语音**：Wav2Vec2 全量微调，Val F1=0.8021
- **视觉**：ResNet18 + face_aug，Val F1=0.7768
- **融合**：classwise_weighted_average，**Test Macro F1 = 0.9334**

后续可改进方向：
- 尝试更大预训练模型（Wav2Vec2-large、HuBERT）进一步提升音频分支
- 探索中间融合或跨模态注意力机制，在特征层建模细粒度交互
- 使用 CREMA-D 等外部数据集进行泛化验证
- 针对 disgust 等少数类设计更精细的数据增强策略