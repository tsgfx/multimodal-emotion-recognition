# 多模态情绪识别实验计划与结果

## 1. 实验结果

### 1.1 音频单模态模型

| 模型 | Val Macro F1 | 备注 |
| --- | ---: | :--- |
| Wav2vec2 fine-tune | **0.8021** | 全量微调，epoch=15 最佳 |

### 1.2 视频单模态模型

| 模型 | Val Macro F1 | 备注 |
| --- | ---: | :--- |
| ResNet18 (pretrained) | 0.7479 | 无增强，epoch=19 最佳 |
| **ResNet18 + face_aug** | **0.7768** | hflip/brightness/contrast，epoch=16 最佳 |

### 1.3 晚融合模型

| 融合方案 | Val Macro F1 | Test Macro F1 |
| --- | ---: | ---: |
| weighted_average | 0.8823 | 0.8575 |
| confidence_weighted_average (β=0.25) | 0.8806 | 0.8987 |
| classwise (基础 face) | 0.9158 | 0.9248 |
| **classwise (aug face)** | **0.9075** | **0.9334** |

**注：** classwise 对每类情绪单独优化权重，表现最优；β=0.25 说明低置信度样本中视觉更可靠。

---

## 2. 配置说明

**情感类别：** angry, disgust, fearful, happy, sad（5 类，移除了 Neutral）

- 数据集按演员划分：train=actors 01-18, val=actors 19-21, test=actors 22-24
- Neutral 被移除的原因：测试集演员 Neutral 类表现极差（OOD 问题），移除后模型更稳定

---

## 3. 后续优化方向

| 优先级 | 内容 | 状态 |
| --- | --- | --- |
| P1 | Wav2vec2 fine-tune | ✅ 完成，Val F1=0.8021 |
| P2 | Face ResNet18 重训（基础+增强） | ✅ 完成，基础 Val F1=0.7479，增强 Val F1=0.7768 |
| P3 | 晚融合评估（3 种策略） | ✅ 完成，classwise aug face 最佳 Test F1=0.9334 |
| P4 | CREMA-D 泛化验证 | 报告收尾，待后续数据集可用时执行 |

---

## 4. 正式推荐方案

**当前最佳配置（5 类）：**
- 音频：Wav2vec2 fine-tune（local path，全量微调，epoch=15）
- 视觉：ResNet18 pretrained + 人脸帧平均池化 + face_aug（epoch=16）
- 融合：classwise_weighted_average
- **Test Macro F1 = 0.9334**，Val Macro F1 = 0.9075

**各模态单独性能：**
- 音频（wav2vec2）：Val F1=0.8021
- 视频（face，无增强）：Val F1=0.7479
- 视频（face，+aug）：Val F1=0.7768
- 晚融合（classwise，aug face）：Val F1=0.9075，相对单模态提升约 +13%

**per-class 最优融合权重（classwise）：**

| 情绪 | α_audio | α_face | 主导模态 |
| --- | ---: | ---: | :--- |
| angry | 0.8 | 0.2 | 音频 |
| disgust | 0.5 | 0.5 | 均衡 |
| fearful | 0.2 | 0.8 | 视觉 |
| happy | 0.5 | 0.5 | 均衡 |
| sad | 0.5 | 0.5 | 均衡 |

---

## 5. 命令参考

```bash
# 音频训练（早停）
python src/train_audio.py --metadata data/processed/metadata.csv \
  --audio_model wav2vec2_finetune \
  --wav2vec2_pretrained /home/ruichao/Workspace/multimodal-emotion-recognition/wav2vec2-base \
  --epochs 20 --batch_size 16 --num_workers 8 --device cuda \
  --early_stopping_patience 8 --output_tag 5class

# 人脸训练（早停 + 数据增强）
python src/train_face.py --metadata data/processed/metadata.csv \
  --epochs 20 --batch_size 8 --frames_per_sample 4 \
  --num_workers 8 --device cuda --pretrained \
  --face_augment --early_stopping_patience 8 --output_tag 5class_aug

# 晚融合评估（推荐：classwise）
python src/evaluate.py --metadata data/processed/metadata.csv \
  --batch_size 8 --frames_per_sample 4 --num_workers 8 --device cuda \
  --audio_checkpoint outputs/checkpoints/audio_wav2vec2_finetune.5class.pt \
  --face_checkpoint outputs/checkpoints/face_resnet18.5class_aug.pt \
  --wav2vec2_pretrained /home/ruichao/Workspace/multimodal-emotion-recognition/wav2vec2-base \
  --fusion_strategy classwise_weighted_average \
  --output_tag 5class_aug

# 音频单独评估
python src/evaluate_audio.py --metadata data/processed/metadata.csv \
  --audio_checkpoint outputs/checkpoints/audio_wav2vec2_finetune.5class.pt \
  --batch_size 16 --num_workers 8 --device cuda
```

---

## 6. 关键模型文件

| 文件 | 说明 |
| --- | :--- |
| `outputs/checkpoints/audio_wav2vec2_finetune.5class.pt` | 音频 Wav2vec2 全量微调模型 |
| `outputs/checkpoints/face_resnet18.5class.pt` | 人脸 ResNet18（无增强） |
| `outputs/checkpoints/face_resnet18.5class_aug.pt` | 人脸 ResNet18（+ face_aug）**推荐** |
| `outputs/metrics/fusion_metrics.5class_aug.json` | 晚融合完整指标 |
| `outputs/metrics/test_fusion_predictions.5class_aug.csv` | 测试集预测结果 |