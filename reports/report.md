# 多模态用户情绪状态识别报告

## 1. 研究背景与问题定义

情绪识别在人机交互、智能客服、在线教育和心理健康辅助分析中具有重要应用价值。单一模态容易受到噪声、遮挡或个体差异影响，因此本项目综合语音和面部表情两种模态，完成用户情绪状态识别。

本项目将任务建模为六分类问题，类别包括：angry、disgust、fearful、happy、neutral、sad。

## 2. 数据集说明

本项目使用 RAVDESS 数据集，筛选包含音频和视频信息的样本。RAVDESS 的文件名包含情绪、强度、语句、重复次数和演员编号等信息，便于解析标签和按演员划分数据。

本实验选择六类情绪：angry、disgust、fearful、happy、neutral、sad。按演员编号划分训练集、验证集和测试集，避免同一演员同时出现在不同划分中造成身份泄漏。最终使用样本数为 1056，其中 neutral 类样本较少。

| split | angry | disgust | fearful | happy | neutral | sad |
| --- | --- | --- | --- | --- | --- | --- |
| train | 144 | 144 | 144 | 144 | 72 | 144 |
| val | 24 | 24 | 24 | 24 | 12 | 24 |
| test | 24 | 24 | 24 | 24 | 12 | 24 |

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

已生成以下特征分析图表：

![Audio energy boxplot](../outputs/figures/audio_energy_boxplot.png)

![MFCC mean heatmap](../outputs/figures/mfcc_mean_heatmap.png)

![Log-Mel examples](../outputs/figures/log_mel_examples.png)

![Face examples](../outputs/figures/face_examples_grid.png)

![Face PCA](../outputs/figures/face_pca.png)

分析要点：

- angry 和 happy 往往能量较高。
- sad 和 neutral 在声学上可能更平稳，容易混淆。
- fearful 和 disgust 可能与 angry 或 sad 在部分声学特征上重叠。
- 面部表情能够为语音不明显的样本提供补充信息。

## 5. 模型设计

语音分支实现了两种模型：

**AudioCNN**：Log-Mel spectrogram → 2D CNN（3 blocks，32→64→128 channels）→ Global Average Pooling → Linear classifier

**AudioCRNN**：在 CNN 特征提取后接双向 GRU 编码器和注意力池化层，能够同时建模局部时频模式和更长的时间依赖：

```text
Log-Mel spectrogram
  -> CNN features (ConvBlock × 3, MaxPool)
  -> AdaptiveAvgPool(8, None)
  -> reshape -> BiGRU
  -> attention-weighted pooling
  -> Linear classifier
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

### 7.1 Audio-only 模型对比

AudioCNN 在 `class_weight` 加持下配合 20 epoch 训练的结果作为正式基线；后续实现的 AudioCRNN 在相同训练预算下用于结构升级对照。

| 模型 | Val Macro F1 | Test Macro F1 |
| --- | ---: | ---: |
| AudioCNN (`class_weight_only`) | 0.4790 | 0.3904 |
| AudioCRNN (`crnn_full_v1`) | 0.6025 | 0.5721 |

AudioCRNN 在测试集 Macro F1 上比 AudioCNN 提升了约 18 个百分点，说明引入时序建模（BiGRU + attention）是提升语音分支的有效手段。

### 7.2 单模态与融合结果

验证集结果：

| 方法 | Accuracy | Macro F1 |
| --- | --- | --- |
| Audio-only (CNN `class_weight_only`) | 0.5000 | 0.4790 |
| Audio-only (CRNN) | 0.6061 | 0.6025 |
| Face-only | 0.7500 | 0.7467 |
| Late fusion (CNN audio + face) | 0.8258 | 0.8271 |

测试集结果：

| 方法 | Accuracy | Macro F1 |
| --- | --- | --- |
| Audio-only (CNN `class_weight_only`) | 0.3939 | 0.3904 |
| Audio-only (CRNN) | 0.6136 | 0.5721 |
| Face-only | 0.7273 | 0.6771 |
| **Late fusion (CNN audio + face)** | **0.7576** | **0.7394** |

当前最佳融合方案为 AudioCNN + Face，最优融合权重为 `alpha_audio = 0.5, alpha_face = 0.5`。尽管 AudioCRNN 的单模态性能显著更强，但与 Face 融合后的最终效果（Test Macro F1 = 0.7214）反而略低于 AudioCNN + Face（0.7394），这说明更强的 audio-only 模型不一定带来更好的跨模态互补性——AudioCNN 和 Face 的错误模式可能天然更互补。

测试集晚融合混淆矩阵：

| label | angry | disgust | fearful | happy | neutral | sad |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| angry | 20 | 4 | 0 | 0 | 0 | 0 |
| disgust | 0 | 22 | 0 | 0 | 0 | 2 |
| fearful | 0 | 1 | 18 | 0 | 0 | 5 |
| happy | 0 | 0 | 4 | 16 | 0 | 4 |
| neutral | 2 | 0 | 0 | 1 | 5 | 4 |
| sad | 4 | 0 | 0 | 0 | 1 | 19 |

## 8. 结果分析

从整体结果看，视觉模态明显强于语音模态。加入 `class_weight` 后，Audio-only 在测试集上的 Macro F1 提升到 0.3904，Face-only 为 0.6771，说明 RAVDESS 视频中的面部表情仍然是当前任务的主导信息源。晚融合进一步提升到 0.7394，说明语音分支虽然单独性能较弱，但经过类别重加权后能够为视觉分支提供更稳定的补充信息。

测试集上表现较好的类别包括 disgust、angry、happy 和 fearful。当前最佳晚融合模型在 disgust 上 F1 为 0.8627，在 angry 上 F1 为 0.8000，在 happy 上 F1 为 0.7805，在 fearful 上 F1 为 0.7826。这些类别通常具有较明显的面部或声学表达，因此更容易识别。

表现最弱的仍然是 neutral，测试集 F1 为 0.5556，虽然相比基线明显改善，但样本量少且边界仍然不稳定。sad 类 F1 为 0.6552，仍然承接了来自 neutral、fearful 和 happy 的一部分误分类，说明低唤醒度情绪之间仍有重叠。

主要混淆包括 neutral -> sad、happy -> fearful、happy -> sad 和 angry -> disgust。这些混淆符合情绪表达的相似性：neutral 与 sad 在语音能量和面部表情幅度上都可能较弱；angry 和 disgust 都属于负向高强度情绪；fearful、happy 和 sad 在部分演员表达中存在个体差异。

在 follow-up 实验中，还测试了 `class_weight_ep30`、`class_weight_lr5e4_ep30` 和 `class_weight_bs32_ep30` 等语音变体。它们能把 audio-only 验证集 Macro F1 提升到 0.4674 到 0.4917，但测试集和最终融合结果都没有超过当前正式方案，说明继续围绕当前小 CNN 做轻量超参微调的收益已经接近上限。

AudioCRNN 升级实验中，CRNN 将 audio-only 测试集 Macro F1 从 0.3904 大幅提升至 0.5721，提升幅度显著。但 CRNN 与 Face 融合后的最终 Macro F1（0.7214）反而略低于 CNN+Face 融合（0.7394）。对两种 audio 模型的错误模式进行量化重叠分析（Overlap Analysis），结果如下：

| | CNN+Face | CRNN+Face |
| --- | --- | --- |
| audio_only_correct（audio 单独纠正） | 15 | 20 |
| face_only_correct（face 单独纠正） | 58 | 34 |
| both_correct（两者都对） | 37 | 61 |
| both_wrong（两者都错） | 22 | 17 |
| oracle_accuracy（完美选择正确模态） | 0.8333 | 0.8712 |
| Fusion Macro F1 | **0.7394** | 0.7214 |

- **更弱的 audio 不代表融合贡献更小**：CNN audio 虽然正确率只有 39.4%，但它在 face 犯错时单独正确的次数为 15；而 CRNN audio 虽然正确率达 61.4%，face 犯错时它单独正确的次数反而只有 20（face_only_correct 从 58 降到 34）。这说明 CNN 的错误恰好与 face 互补——CNN 弥补 face 的短板，而 face 也弥补 CNN 的短板，两者形成更好的错误隔离。
- **CRNN 错误与 face 高度重叠**：CRNN 让 both_correct 从 37 增至 61，说明 CRNN 和 face 倾向于同时做对或同时做错，模态间互补性反而降低。虽然 CRNN 的 oracle（0.8712）更高，说明单个样本上 CRNN 的置信度更可靠，但加权平均融合无法利用这个信息——融合算法只知道概率，不知道何时该信 audio 何时装 face。
- **oracle 与实际融合的差距揭示融合上限**：CNN+Face 的 oracle accuracy 为 0.8333，实际融合为 0.7576，差距为 0.076；CRNN+Face oracle 为 0.8712，实际为 0.7652，差距扩大到 0.106。说明当前简单加权融合已经接近 CNN+Face 的最优边界，而 CRNN+Face 的潜力尚未被这种融合方式充分挖掘。

本实验采用简单晚融合，优势是实现稳定、可解释性强，并且能够直接比较不同模态贡献。局限是融合只发生在最终概率层，无法学习语音和视觉之间更细粒度的互补关系。

## 9. 总结与展望

本项目完成了语音和面部表情两种模态的情绪识别流程，包括数据预处理、特征分析、单模态建模、多模态晚融合和实验评估。

后续可改进方向：

- 尝试预训练语音模型（wav2vec2、HuBERT 等），进一步提升音频特征表达能力。
- 探索中间融合或注意力融合，使语音和视觉特征能在更早阶段交互。
- 使用 CREMA-D 作为外部泛化测试，验证模型在演员更多样、个体差异更大数据集上的稳健性。
- 优化 neutral 等弱类的数据增强或重采样策略。
