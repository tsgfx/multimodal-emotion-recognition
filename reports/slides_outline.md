# 项目展示 PPT 提纲

## 1. 任务背景

- 多模态情绪识别的应用价值（人机交互、智能客服、心理健康）
- 单模态方法的局限（噪声、遮挡、个体差异）
- 本项目目标：语音 + 面部表情五分类情绪识别

## 2. 数据集与类别

- 数据集：RAVDESS
- 情绪类别：angry、disgust、fearful、happy、sad（移除了 Neutral）
- 模态：音频（Wav2Vec2）、视频人脸帧（ResNet18）
- 数据划分：train=actors 01-18 / val=actors 19-21 / test=actors 22-24（按演员划分避免身份泄漏）
- 移除了 Neutral 类的原因：测试集 OOD 问题，演员间表达差异大

## 3. 数据预处理

- 音频：16 kHz、定长 3s、Log-Mel spectrogram（64 mels）
- 视频：抽 8 帧、OpenCV 人脸检测、224×224 归一化

## 4. 特征分析

- 音频能量箱线图
- MFCC 均值热力图
- Log-Mel 示例图
- 人脸样例网格图
- 人脸 PCA 可视化

## 5. 模型结构

- 语音：Wav2Vec2-base 端到端微调（编码器 lr=1e-5，头 lr=1e-4）
- 视觉：ResNet18 pretrained + temporal mean pooling
- 训练增强：人脸数据增强（hflip、brightness、contrast）
- 晚融合：classwise_weighted_average（每类独立优化权重）

## 6. 实验设计

- Audio-only（Wav2Vec2 fine-tune）
- Face-only（ResNet18，基础 vs 增强）
- Audio + Face late fusion（三策略对比）
- 指标：Accuracy、Macro F1、混淆矩阵、每类 P/R/F1

## 7. 实验结果

### 单模态对比

| 模型 | Val Macro F1 |
| --- | ---: |
| Audio Wav2Vec2 fine-tune | **0.8021** |
| Face ResNet18（无增强） | 0.7479 |
| Face ResNet18（+ face_aug） | **0.7768** |

### 晚融合对比

| 融合策略 | Val Macro F1 | Test Macro F1 |
| --- | ---: | ---: |
| weighted_average | 0.8823 | 0.8575 |
| confidence_weighted_average | 0.8806 | 0.8987 |
| **classwise_weighted_average** | **0.9075** | **0.9334** |

### per-class 最优权重

| 情绪 | α_audio | α_face | 主导模态 |
| --- | ---: | ---: | :--- |
| angry | 0.8 | 0.2 | 音频 |
| fearful | 0.2 | 0.8 | 视觉 |
| disgust/happy/sad | 0.5 | 0.5 | 均衡 |

## 8. 结果分析与总结

- 多模态互补显著：晚融合相对单模态提升约 **+13%**（Val）
- 模态主导因类而异：angry→音频，fearful→视觉，其他均衡
- face_augmentation 有效：单模态 +3%，融合 Test F1 从 0.9248→0.9334
- 所有五类 Test F1 ≥ 0.92，模型稳健

## 9. 后续方向

- 更大预训练模型（Wav2Vec2-large、HuBERT）
- 中间融合 / 跨模态注意力
- CREMA-D 泛化验证
- 针对 disgust 的精细增强策略