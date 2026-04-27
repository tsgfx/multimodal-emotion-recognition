# 项目展示 PPT 提纲

## 1. 任务背景

- 多模态情绪识别的应用价值
- 单模态方法的局限
- 本项目目标：语音 + 面部表情六分类情绪识别

## 2. 数据集与类别

- 数据集：RAVDESS
- 情绪类别：angry、disgust、fearful、happy、neutral、sad
- 模态：音频、视频人脸帧
- 数据划分：train / val / test

## 3. 数据预处理

- 音频：16 kHz、定长、Log-Mel、MFCC
- 视频：抽帧、人脸检测、裁剪、归一化

## 4. 特征分析

- 音频能量箱线图
- MFCC 均值热力图
- Log-Mel 示例图
- 人脸样例网格图
- 人脸 PCA 可视化

## 5. 模型结构

- 语音 CNN 分支（AudioCNN：Log-Mel + 3层CNN）
- 语音 CRNN 分支（AudioCNN + BiGRU + attention，作为 audio-only 对照）
- ResNet18 人脸分支
- 晚融合策略（weighted_average）

## 6. 实验设计

- Audio-only
- Face-only
- Audio + Face late fusion
- 指标：Accuracy、Macro F1、混淆矩阵

## 7. 实验结果

- AudioCNN vs AudioCRNN：CRNN 将 audio-only Test Macro F1 从 0.3904 提升至 0.5721
- 验证集结果：AudioCNN-only Macro F1 0.4790，Face-only Macro F1 0.7467，Late fusion Macro F1 0.8271
- 测试集结果：AudioCNN-only Macro F1 0.3904，CRNN-only Macro F1 0.5721，Face-only Macro F1 0.6771，**Late fusion (CNN+Face) Macro F1 0.7394**（最优）
- CRNN+Face 融合（0.7214）反而略低于 CNN+Face（0.7394）——Overlap 分析揭示原因：CNN 弱但与 face 互补（face_only_correct=58，audio 纠正 15 个 face 错）；CRNN 强但与 face 重叠（face_only_correct=34，audio 纠正仅 20 个 face 错）
- oracle accuracy：CNN+Face=0.8333，CRNN+Face=0.8712；但实际融合差距反而缩小，说明简单加权融合已接近 CNN+Face 最优边界
- 最优融合权重：audio 0.5，face 0.5
- 测试集混淆矩阵：重点展示 neutral -> sad、fearful -> sad、angry -> disgust

## 8. 结果分析与总结

- 容易识别的情绪：disgust、angry、happy
- 容易混淆的情绪：neutral 与 sad，fearful 与 sad，angry 与 disgust
- AudioCRNN 显著提升 audio-only 性能，但与 face 融合后未能超越 CNN+Face
- 可能原因：更强的 audio 模型减少了自身错误，但削弱了与 face 的互补增益
- 多模态融合相比 face-only 进一步提升测试集 Macro F1
- 局限：样本量较小，neutral 类样本少；晚融合无法建模细粒度跨模态交互
