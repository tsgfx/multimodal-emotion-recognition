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

- 语音 CNN 分支
- ResNet18 人脸分支
- 晚融合策略

## 6. 实验设计

- Audio-only
- Face-only
- Audio + Face late fusion
- 指标：Accuracy、Macro F1、混淆矩阵

## 7. 实验结果

- 单模态与多模态结果对比表
- 最优融合权重
- 测试集混淆矩阵

## 8. 结果分析与总结

- 容易识别的情绪
- 容易混淆的情绪
- 多模态融合的有效性
- 局限与未来改进
