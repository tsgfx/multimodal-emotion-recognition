# 多模态情绪识别实验计划与结果

## 1. 实验结果（6 类）

### 1.1 音频单模态模型

| 模型 | Val Macro F1 | Test Macro F1 | 备注 |
| --- | ---: | ---: | :--- |
| CNN (class_weight_only) | 0.4790 | 0.3904 | 正式基线 |
| CRNN (crnn_full_v1) | 0.6025 | **0.5721** | BiGRU+Attention |
| Wav2vec2 frozen | 0.5007 | 0.4658 | Frozen编码器 |
| **Wav2vec2 fine-tune (epoch=13)** | **0.7456** | **0.7525** | 全量微调，**最佳** |

### 1.2 视频单模态模型

| 模型 | Val Macro F1 | Test Macro F1 | 备注 |
| --- | ---: | ---: | :--- |
| ResNet18 (pretrained) | 0.7467 | 0.6771 | Face 是主导模态 |

### 1.3 晚融合模型

| 融合方案 | 音频分支 | α_audio | α_face | Val Macro F1 | Test Macro F1 | 备注 |
| --- | --- | ---: | ---: | ---: | ---: | :--- |
| weighted_average | CNN (class_weight_only) | 0.5 | 0.5 | 0.8271 | 0.7394 | 旧基线 |
| weighted_average | CRNN (crnn_full_v1) | 0.3 | 0.7 | 0.7920 | 0.7214 | |
| weighted_average | **Wav2vec2 fine-tune** | **0.3** | **0.7** | **0.8449** | **0.7888** | **最新最佳** |

---

## 2. 具体问题与解决路径

### 问题：Neutral 类 F1 极低（0.5）

**现象：** Wav2vec2+Face fusion 在 neutral 类上 test F1 仅 0.5，12 个样本中对 4 个。

**根因：** 数据集按演员划分（train: actors 01-18, test: actors 22-24），测试集演员从未在训练时见过。演员 23 的 neutral 听起来像 sad，演员 24 的 neutral 像 angry。这是 **Out-of-Distribution (OOD)** 问题，不是模型能力问题。

**方案 A（进行中）：去除 Neutral 类，5 类分类**

- 音频：Wav2vec2 fine-tune（facebook/wav2vec2-base，全量微调）
- 视觉：ResNet18 pretrained + 人脸帧平均池化
- 融合：weighted_average，α_audio=0.3, α_face=0.7

---

## 3. 后续优化方向

| 优先级 | 内容 | 状态 |
| --- | --- | --- |
| P1 | Wav2vec2 fine-tune | ✅ 已完成 |
| P2 | **5 类分类**（去除 Neutral） | 进行中 |
| P3 | CREMA-D 泛化验证 | 待开始 |

---

## 4. 正式推荐方案

**当前最佳配置（6 类）：**
- 音频：Wav2vec2 fine-tune（facebook/wav2vec2-base，全量微调，epoch=13）
- 视觉：ResNet18 pretrained + 人脸帧平均池化
- 融合：weighted_average，α_audio=0.3, α_face=0.7
- **Test Macro F1 = 0.7888**（actor-hold-out 划分）

**5 类配置（方案 A）：** 待训练验证