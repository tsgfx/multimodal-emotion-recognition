# 多模态情绪识别作业执行计划

## 1. 项目目标与完成标准

本项目目标是完成一个完整的多模态用户情绪状态识别数据科学项目，使用至少两种模态信息识别用户情绪，并提交课程报告、程序代码、运行文档和项目展示 PPT。

硬性完成标准：

- 情绪类别不少于 4 类。
- 使用至少 2 种模态。
- 完成不同模态的特征分析。
- 说明并实现多模态融合方法。
- 建立完整的多模态情绪识别模型，并给出实验结果。
- 报告包含背景、数据说明、方法设计、实验过程、结果分析和总结。

推荐主线方案：

- 情绪类别：`happy`、`sad`、`angry`、`neutral`、`fearful`、`disgust`。
- 模态组合：语音 + 面部表情。
- 融合方式：晚融合为主，单模态模型作为对比实验。
- 项目形式：Python 深度学习项目，配套实验脚本、结果文件、报告和 PPT。

## 2. 技术路线总览

整体流程：

1. 明确任务定义：将多模态情绪识别建模为 6 类情绪分类任务。
2. 获取并整理数据：优先选择同时包含音频和视频/人脸信息的公开数据集。
3. 数据预处理：从音频中提取声学特征，从视频中抽取人脸图像或帧级视觉特征。
4. 特征分析：分别分析语音特征和面部表情特征与情绪类别之间的关系。
5. 单模态建模：分别训练语音模型和图像/视频模型。
6. 多模态融合：使用晚融合或特征级融合得到最终预测。
7. 结果评估：比较语音单模态、视觉单模态、多模态融合模型。
8. 完成报告、README 和 PPT。

## 3. 数据集选择与数据方案

### 3.1 首选数据集

优先使用 RAVDESS 或 CREMA-D 这类包含语音、面部视频和情绪标签的数据集。选择原因：

- 包含音频和视频，满足至少两种模态要求。
- 标签中包含多种情绪，能够筛选出至少 4 类。
- 规模适合作业实现，训练成本可控。
- 类别标签清晰，便于写报告和做实验分析。

建议优先采用 RAVDESS：

- 模态：音频、视频、人脸表情。
- 可用情绪：happy、sad、angry、neutral、fearful、disgust、surprised、calm。
- 本项目筛选：happy、sad、angry、neutral、fearful、disgust 六类。
- 备选降级方案：如果 6 类训练结果过差或时间不足，可临时降级为 happy、sad、angry、neutral 四类作为稳定对照。

### 3.2 RAVDESS 与 CREMA-D 对比

| 对比项 | RAVDESS | CREMA-D |
| --- | --- | --- |
| 全名 | Ryerson Audio-Visual Database of Emotional Speech and Song | Crowd-sourced Emotional Multimodal Actors Dataset |
| 数据规模 | 约 7,356 个文件 | 约 7,442 个视频片段 |
| 演员数量 | 24 名专业演员 | 91 名演员 |
| 模态 | 音频、视频、音视频；包含 speech 和 song | 音频 + 视频 |
| 情绪类别 | neutral、calm、happy、sad、angry、fearful、disgust、surprised | angry、disgust、fear、happy、neutral、sad |
| 数据特点 | 录制条件更统一，数据更规整 | 演员更多，个体差异更大 |
| 标签处理 | 文件名编码清晰，解析简单 | 文件名也较清晰，但标注体系稍复杂 |
| 实现难度 | 较低 | 中等 |
| 适合用途 | 课程作业、快速跑通完整流程 | 泛化实验、扩展实验、更丰富分析 |

核心区别：

- RAVDESS 更标准化，演员数量较少，录制环境统一，标签和文件命名规则清晰，更适合在课程作业中快速完成数据预处理、模型训练、融合实验和报告撰写。
- CREMA-D 演员数量更多，样本主体差异更丰富，更适合讨论模型泛化能力，但训练难度和结果波动通常更高。

本项目默认选择：

- 主数据集：RAVDESS。
- 默认情绪类别：happy、sad、angry、neutral、fearful、disgust 六类。
- 选择理由：满足 4 类以上情绪和 2 种以上模态要求，同时比 4 类设置更完整；RAVDESS 实现成本较低，便于按时完成完整项目流程。
- 可选扩展：若主流程完成后仍有时间，可使用 CREMA-D 作为补充数据集或泛化测试数据集。

### 3.3 数据目录规划

建议建立如下目录：

```text
data/
  raw/
    ravdess/
  processed/
    audio_features/
    face_frames/
    metadata.csv
outputs/
  checkpoints/
  figures/
  metrics/
reports/
  report.md
  slides_outline.md
src/
  config.py
  prepare_data.py
  extract_audio_features.py
  extract_face_frames.py
  analyze_features.py
  train_audio.py
  train_face.py
  train_fusion.py
  evaluate.py
  dataset.py
  models.py
README.md
requirements.txt
```

### 3.4 元数据设计

生成统一的 `data/processed/metadata.csv`，每一行表示一个样本：

```text
sample_id,audio_path,video_path,face_dir,label,actor_id,split
```

字段说明：

- `sample_id`：样本唯一编号。
- `audio_path`：音频文件路径。
- `video_path`：原始视频路径。
- `face_dir`：抽取后的人脸帧目录。
- `label`：情绪类别。
- `actor_id`：说话人或演员编号。
- `split`：`train`、`val`、`test`。

划分原则：

- 推荐按照说话人划分训练集、验证集和测试集，减少同一说话人同时出现在训练和测试中的数据泄漏。
- 若数据规模太小，可先使用分层随机划分，但需要在报告中说明限制。
- 推荐比例：训练集 70%，验证集 15%，测试集 15%。

## 4. 数据预处理计划

### 4.1 音频预处理

输入：每个样本的音频文件。

处理步骤：

1. 统一采样率，例如 16 kHz。
2. 转为单声道。
3. 对音频长度进行裁剪或补零，例如统一为 3 秒或 4 秒。
4. 提取声学特征：
   - MFCC。
   - Log-Mel spectrogram。
   - 能量、过零率、谱质心等统计特征，作为特征分析使用。
5. 保存为 `.npy` 特征文件。

建议实现：

- 建模输入使用 Log-Mel spectrogram，形状类似 `[1, n_mels, time]`。
- 报告中的特征分析使用 MFCC、能量、音高相关统计特征进行可视化。

### 4.2 视频/图像预处理

输入：每个样本的视频文件。

处理步骤：

1. 从视频中按固定间隔抽帧，例如每秒 2 到 5 帧。
2. 对每帧进行人脸检测。
3. 裁剪人脸区域。
4. 调整图像大小，例如 `224x224`。
5. 归一化像素值。
6. 每个样本保留固定数量的人脸帧，例如 8 或 16 帧，不足则重复采样或补齐。

建议实现：

- 为降低复杂度，视觉分支可采用单帧或多帧平均策略。
- 若时间紧张，先抽取每个视频中质量较好的若干张人脸帧，并使用图像分类模型提取特征。

## 5. 特征分析计划

特征分析需要服务报告，而不只是训练模型。建议输出图表到 `outputs/figures/`。

### 5.1 音频特征分析

分析内容：

- 不同情绪类别的平均音量或能量分布。
- 不同情绪类别的 MFCC 均值差异。
- 不同情绪类别的 Log-Mel spectrogram 示例。
- angry 与 happy 可能具有更高能量，sad 可能语速和能量较低，neutral 通常更平稳，fearful 和 disgust 可能在声学特征上与 angry 或 sad 出现重叠。

建议图表：

- 各类别音频能量箱线图。
- 各类别 MFCC 均值热力图。
- 每类情绪 1 个 Log-Mel spectrogram 示例图。

### 5.2 面部表情特征分析

分析内容：

- 不同情绪类别的人脸样例图。
- 使用预训练 CNN 提取视觉 embedding 后，用 t-SNE 或 PCA 可视化类别分布。
- 分析 happy、angry、sad、neutral、fearful、disgust 在面部区域上的差异。

建议图表：

- 每类情绪的人脸样例网格图。
- 视觉特征 t-SNE 散点图。
- 混淆较多类别的人脸样例对比图。

### 5.3 多模态关系分析

分析内容：

- 音频和视觉模态是否对同一情绪给出一致信息。
- 单模态错误样本中，另一模态是否能纠正错误。
- 解释多模态融合优于单模态的原因。

建议输出：

- 单模态与多模态准确率对比柱状图。
- 若干典型样本的语音预测、视觉预测和融合预测对比表。

## 6. 模型设计计划

### 6.1 语音单模态模型

输入：Log-Mel spectrogram。

推荐结构：

```text
Log-Mel spectrogram
  -> 2D CNN
  -> Global Average Pooling
  -> Fully Connected
  -> Emotion logits
```

可选增强：

- 使用 dropout 防止过拟合。
- 使用 class weight 或 weighted sampler 处理类别不平衡。
- 使用 SpecAugment 做简单音频特征增强。

### 6.2 视觉单模态模型

输入：人脸帧图像。

推荐结构：

```text
Face frames
  -> ResNet18 或 MobileNetV2
  -> Frame-level embeddings
  -> Average pooling across frames
  -> Fully Connected
  -> Emotion logits
```

实现策略：

- 优先使用预训练 ResNet18，减少训练成本。
- 如果每个样本有多帧，对帧级 logits 或 embeddings 求平均。
- 如果时间紧张，可以先用每个视频的中心帧作为视觉输入，再扩展到多帧。

### 6.3 多模态融合模型

主方案：晚融合。

```text
Audio model -> audio logits/probabilities
Face model  -> face logits/probabilities
Fusion      -> weighted average or small MLP
Final       -> emotion prediction
```

推荐先实现两种融合：

1. 概率平均：

```text
final_prob = 0.5 * audio_prob + 0.5 * face_prob
```

2. 加权平均：

```text
final_prob = alpha * audio_prob + (1 - alpha) * face_prob
```

其中 `alpha` 在验证集上搜索，例如 `0.1, 0.2, ..., 0.9`。

可选扩展：中间融合。

```text
audio_embedding + face_embedding
  -> concatenate
  -> MLP classifier
  -> emotion logits
```

建议优先保证晚融合可跑通，再实现中间融合作为加分实验。

## 7. 实验设计与评价指标

### 7.1 对比实验

必须完成：

- 实验 A：语音单模态模型。
- 实验 B：视觉单模态模型。
- 实验 C：多模态晚融合模型。

建议扩展：

- 实验 D：不同融合权重对性能的影响。
- 实验 E：中间融合与晚融合对比。

### 7.2 评价指标

至少报告：

- Accuracy。
- Macro F1-score。
- 每类 Precision、Recall、F1-score。
- Confusion Matrix。

原因：

- Accuracy 直观反映整体分类效果。
- Macro F1 能更公平地评价类别不平衡场景。
- Confusion Matrix 能支持“哪类情绪容易混淆”的结果分析要求。

### 7.3 实验记录格式

将实验结果保存到 `outputs/metrics/experiment_results.csv`：

```text
experiment,modalities,fusion,accuracy,macro_f1,happy_f1,sad_f1,angry_f1,neutral_f1,fearful_f1,disgust_f1,notes
audio_only,audio,none,,,,,,,,,
face_only,face,none,,,,,,,,,
late_fusion,audio+face,weighted_average,,,,,,,,,
```

## 8. 代码实现阶段计划

### 阶段 1：项目骨架

目标：建立可维护的代码结构。

任务：

- 创建 `src/`、`data/`、`outputs/`、`reports/` 目录。
- 编写 `requirements.txt`。
- 编写基础配置文件 `src/config.py`。
- 更新 `README.md`，说明项目目标和运行流程。

验收标准：

- 仓库结构清晰。
- README 能说明如何准备数据、运行预处理、训练和评估。

### 阶段 2：数据准备

目标：让数据以统一格式进入训练流程。

任务：

- 下载或手动放置原始数据到 `data/raw/`。
- 编写 `src/prepare_data.py` 解析原始文件名和标签。
- 生成 `data/processed/metadata.csv`。
- 完成 train/val/test 划分。

验收标准：

- `metadata.csv` 包含样本路径、标签和划分字段。
- 每个类别至少有一定数量样本。
- 训练、验证、测试集标签分布合理。

### 阶段 3：特征提取

目标：完成音频和视觉预处理。

任务：

- 编写 `src/extract_audio_features.py`。
- 编写 `src/extract_face_frames.py`。
- 保存音频特征和人脸帧。
- 对异常样本做日志记录，例如无法检测人脸或音频损坏。

验收标准：

- 大多数样本成功生成音频特征。
- 大多数视频成功抽取人脸帧。
- 失败样本有明确记录，报告中可说明处理方式。

### 阶段 4：特征分析

目标：产出报告中需要的分析图表。

任务：

- 编写 `src/analyze_features.py`。
- 绘制音频能量箱线图。
- 绘制 MFCC 均值热力图。
- 绘制人脸样例网格图。
- 绘制视觉特征 t-SNE 或 PCA 图。

验收标准：

- `outputs/figures/` 下有可直接放入报告的图片。
- 每张图有清晰标题、坐标轴和类别标注。

### 阶段 5：单模态模型训练

目标：获得语音模型和视觉模型基线。

任务：

- 编写 `src/dataset.py` 加载音频特征和人脸帧。
- 编写 `src/models.py` 定义音频 CNN、视觉 CNN 和融合模块。
- 编写 `src/train_audio.py`。
- 编写 `src/train_face.py`。
- 保存最优模型到 `outputs/checkpoints/`。

验收标准：

- 语音单模态模型能完成训练和测试。
- 视觉单模态模型能完成训练和测试。
- 输出 Accuracy、Macro F1 和混淆矩阵。

### 阶段 6：多模态融合实验

目标：完成最终多模态识别模型。

任务：

- 编写 `src/train_fusion.py` 或 `src/evaluate.py` 中的融合逻辑。
- 加载语音模型和视觉模型预测结果。
- 实现概率平均和加权平均。
- 在验证集搜索最优融合权重。
- 在测试集报告最终结果。

验收标准：

- 多模态融合模型完成测试。
- 结果表包含 audio-only、face-only、fusion 三组对比。
- 报告能够解释多模态是否优于单模态，以及原因。

### 阶段 7：结果分析与错误分析

目标：满足作业对结果分析的要求。

任务：

- 生成混淆矩阵。
- 找出容易混淆的情绪对，例如 sad 与 neutral。
- 选取若干错误样本进行案例分析。
- 对比单模态错误与多模态融合结果。

验收标准：

- 报告中有定量指标和定性分析。
- 能说明模型优点、缺点和失败原因。

### 阶段 8：报告、README 和 PPT

目标：完成最终提交材料。

任务：

- 编写课程报告。
- 完善 README。
- 制作 PPT。

报告结构建议：

1. 研究背景与问题定义。
2. 数据集说明。
3. 数据预处理。
4. 特征分析。
5. 模型设计。
6. 实验设置。
7. 实验结果。
8. 结果分析。
9. 总结与展望。

PPT 结构建议：

1. 任务背景。
2. 数据与类别。
3. 多模态方法框架。
4. 特征分析图。
5. 模型结构。
6. 实验结果对比。
7. 混淆矩阵与错误分析。
8. 总结与改进方向。

## 9. 推荐命令设计

后续代码实现完成后，README 中建议提供如下命令：

```bash
pip install -r requirements.txt

python src/prepare_data.py --data_root data/raw/ravdess --output data/processed/metadata.csv

python src/extract_audio_features.py --metadata data/processed/metadata.csv

python src/extract_face_frames.py --metadata data/processed/metadata.csv

python src/analyze_features.py --metadata data/processed/metadata.csv

python src/train_audio.py --metadata data/processed/metadata.csv

python src/train_face.py --metadata data/processed/metadata.csv

python src/evaluate.py --metadata data/processed/metadata.csv --fusion weighted_average
```

## 10. 时间安排

如果按 7 天完成，建议节奏如下：

| 时间 | 目标 | 产出 |
| --- | --- | --- |
| 第 1 天 | 明确方案、准备数据、建立项目结构 | 目录结构、README 初稿、metadata.csv |
| 第 2 天 | 完成音频和视频预处理 | 音频特征、人脸帧 |
| 第 3 天 | 完成特征分析 | 分析图表 |
| 第 4 天 | 训练语音单模态模型 | audio-only 结果 |
| 第 5 天 | 训练视觉单模态模型 | face-only 结果 |
| 第 6 天 | 实现多模态融合和评估 | fusion 结果、混淆矩阵 |
| 第 7 天 | 完成报告、README、PPT | 最终提交材料 |

如果时间只有 3 天，采用压缩方案：

| 时间 | 目标 | 取舍 |
| --- | --- | --- |
| 第 1 天 | 数据整理、音频特征、抽取人脸中心帧 | 先保证数据可训练 |
| 第 2 天 | 训练两个单模态模型和晚融合 | 不做复杂中间融合 |
| 第 3 天 | 生成图表、写报告和 PPT | 聚焦满足硬性要求 |

## 11. 风险与应对

### 风险 1：视频中人脸检测失败

应对：

- 保留失败日志。
- 调整抽帧位置和检测阈值。
- 若多帧失败，使用原始帧中心裁剪作为兜底。

### 风险 2：数据量较小导致过拟合

应对：

- 使用预训练视觉模型。
- 添加 dropout。
- 使用数据增强。
- 报告中说明小数据集限制。

### 风险 3：多模态融合没有明显优于单模态

应对：

- 报告中如实分析原因。
- 尝试验证集加权融合，而不是固定 0.5 平均。
- 对单模态互补样本做案例分析，说明多模态的价值。

### 风险 4：类别不平衡

应对：

- 统计类别分布。
- 使用 Macro F1。
- 使用 class weight 或 weighted sampler。

### 风险 5：训练时间不足

应对：

- 降低图像分辨率。
- 减少帧数。
- 冻结预训练视觉模型大部分层。
- 优先完成晚融合，不追求复杂模型。

## 12. 最终验收清单

数据与任务：

- [ ] 默认完成 6 类情绪：happy、sad、angry、neutral、fearful、disgust。
- [ ] 至少 4 类情绪的课程硬性要求已满足。
- [ ] 至少 2 种模态。
- [ ] 有清晰的数据来源和标签说明。
- [ ] 有训练集、验证集和测试集划分。

代码：

- [ ] 能完成数据预处理。
- [ ] 能完成音频特征提取。
- [ ] 能完成人脸帧提取。
- [ ] 能训练语音单模态模型。
- [ ] 能训练视觉单模态模型。
- [ ] 能完成多模态融合评估。
- [ ] README 写明运行步骤。

实验：

- [ ] 有 audio-only 结果。
- [ ] 有 face-only 结果。
- [ ] 有 multimodal fusion 结果。
- [ ] 有 Accuracy 和 Macro F1。
- [ ] 有混淆矩阵。
- [ ] 有单模态 vs 多模态对比。

报告：

- [ ] 有研究背景。
- [ ] 有数据说明。
- [ ] 有预处理说明。
- [ ] 有特征分析。
- [ ] 有模型设计。
- [ ] 有实验结果。
- [ ] 有结果分析。
- [ ] 有总结与展望。

PPT：

- [ ] 有任务介绍。
- [ ] 有数据和类别说明。
- [ ] 有方法框架图。
- [ ] 有实验结果图表。
- [ ] 有结论和改进方向。

## 13. 推荐最低可交付版本

如果需要优先确保作业可以提交，最低可交付版本如下：

- 数据：RAVDESS 中筛选 happy、sad、angry、neutral、fearful、disgust 六类。
- 模态：音频 + 视频人脸中心帧。
- 音频模型：Log-Mel spectrogram + CNN。
- 视觉模型：人脸图像 + 预训练 ResNet18。
- 融合：验证集搜索权重的晚融合。
- 结果：报告 Accuracy、Macro F1、混淆矩阵。
- 分析：音频特征箱线图、人脸样例图、单模态与融合结果对比。
- 备选：如果 6 类结果过差，可补充 happy、sad、angry、neutral 四类实验作为稳定对照，但主报告仍优先呈现 6 类任务。

该版本能够满足作业要求中的核心检查项，并为后续扩展中间融合、多帧建模和更复杂特征分析留出空间。
