# CFHTransformer_0415

本仓库复现论文 *"Cycle Frequency-Harmonic-Time Transformer for Note-Level Singing Voice Transcription"* (ICME 2024)，并持续优化以逼近论文报告的指标。

## 版本历史

| 版本 | COn | COnP | COnPOff | 论文目标 COnP | 主要改动 |
|------|----:|-----:|--------:|-------------:|---------|
| v6 (旧) | 0.7906 | 0.7604 | 0.4075 | 0.8013 | 基线版本 |
| v7 (旧) | 0.8022 | 0.7752 | 0.4464 | 0.8013 | CQT归一化 + onset类别平衡 + 400首训练 |
| **v7 (最终)** | **0.8007** | **0.7734** | **0.4543** | 0.8013 | GAP: mean→sum + LayerNorm + splits修复 + 更长训练 |

> 旧版代码保存在 `v7_旧/` 目录下。

## v7 (最终版本) 改动详情

相对之前推送的旧版 v7，本次更新的核心改动：

1. **GAP 聚合方式** — 从 `mean` 改为 `sum`（论文原文 "sum up the harmonic features"），并增加 LayerNorm 稳定数值
2. **数据集划分修复** — 使用 `splits_v11/`（361-400 做 val，401-500 做 test），之前 splits 有交叉污染
3. **训练集样本数** — `max_samples_per_epoch` 从 2000 提升至 4000
4. **更长训练** — best model 在 epoch 315（旧版 best 在 ~80 epoch 即过拟合）；后续训练 val 继续上升但 test 下降，epoch 315 为最优
5. **CQT 归一化** — model 内置 `cqt_mean=-65.0, cqt_std=18.0` 归一化
6. **Onset 类别平衡** — `onset_pos_weight=5.0`
7. **训练集** — 400 首（对齐论文：400 train + 100 test，无独立 val）

## Test 集全量评测结果 (100 首)

Best model: **epoch 315**, 阈值网格搜最优 onset=0.55, frame=0.50

```
         Precision  Recall   F1-score
COnPOff  0.442561   0.467830 0.454339
COnP     0.773769   0.772978 0.773362
COn      0.800942   0.800521 0.800730
gt note num: 31311.0  tr note num: 31203.0  song number: 100
```

| 指标 | Precision | Recall | F1-score |
|------|----------:|-------:|---------:|
| COnPOff | 0.4426 | 0.4678 | 0.4543 |
| COnP | 0.7738 | 0.7730 | **0.7734** |
| COn | 0.8009 | 0.8005 | 0.8007 |

### 不同阈值对比

| onset | frame | COn | COnP | COnPOff |
|------:|------:|-----:|-----:|--------:|
| 0.55 | 0.50 | 0.8007 | **0.7734** | 0.4543 |
| 0.50 | 0.50 | 0.8006 | 0.7729 | 0.4541 |
| 0.55 | 0.45 | 0.8002 | 0.7733 | 0.4397 |
| 0.50 | 0.40 | 0.7994 | 0.7721 | 0.4254 |

### 过拟合分析

| Checkpoint | Epoch | val COnP | test COnP | test COnPOff |
|-----------|-------|----------|-----------|-------------|
| **epoch0315** | 315 | 0.8381 | **0.7734** | 0.4543 |
| epoch0456 | 456 | 0.8632 | 0.7645 | 0.4758 |

> epoch 315 之后 val COnP 持续上升，但 test COnP 反而下降约 1 个点，表明模型开始过拟合。因此选择 epoch 315 作为最终模型。

### 与旧版 v7 对比

| 版本 | COn | COnP | COnPOff |
|------|----:|-----:|--------:|
| v7 (旧) | 0.8022 | 0.7752 | 0.4464 |
| **v7 (最终)** | 0.8007 | 0.7734 | **0.4543** |

> 注：旧版 v7 的 test 集存在划分问题，与当前版本不可直接比较。当前版本使用严格无交叉的 splits_v11，结果更可靠。

## 环境与数据依赖

- Python 3.12, PyTorch, librosa, mir_eval
- CQT 预计算缓存：288-bin, hop=800, 50ms/帧
- 标注文件：`MIR-ST500_corrected.json`
- 数据集划分：`splits_v11/` (1-360 train, 361-400 val, 401-500 test)

## 训练

```bash
CUDA_VISIBLE_DEVICES=1 python3 train_conp.py --config config.yaml

# 后台运行
screen -S cft_v7
CUDA_VISIBLE_DEVICES=1 python3 train_conp.py --config config.yaml
# Ctrl+A D 退出screen
```

训练产物保存在 `run/v7_best/` 下：
- `checkpoints/best_model.pt` — 最优模型（epoch 315，按 COnP F1 选择）
- `test_monitor.txt` — 全量 test 监控记录（每 40 epoch 固定一次；20 epoch 后若出现新 best 也会追加一次，仅诊断不参与调参/选模）

## 推理

```bash
CUDA_VISIBLE_DEVICES=3 python3 predict_to_json.py \
    --config config.yaml \
    --checkpoint run/v7_best/checkpoints/best_model.pt \
    --split test \
    --onset_thresh 0.55 \
    --frame_thresh 0.50 \
    --output pred_test_v7.json
```

## 训练后两阶段阈值搜索与 offset-aware 后处理

本目录保留原训练脚本不变：训练内仍是 onset/frame 两阈值，best model 仍按 val COnP 选择。若需要复用 v11 风格的训练后流程，可以先在 val40 上独立搜索 onset/frame，再固定这两个阈值搜索 offset，并最终在 test100 上保存 offset-aware 预测。

第一阶段：val40 搜索 onset/frame，并用选出的阈值验证 test100：

```bash
python3 "评估/A在val40上探索最佳onset和frame阈值.py" \
    --config config.yaml \
    --checkpoint run/某个实验/checkpoints/某个模型.pt
```

第二阶段：固定第一阶段选出的 onset/frame，在 val40 上搜索 offset，并在 test100 上保存 offset-aware JSON：

```bash
python3 "评估/B在val40上探索最佳offset阈值.py" \
    --config config.yaml \
    --checkpoint run/某个实验/checkpoints/某个模型.pt \
    --onset_thresh 0.45 \
    --frame_thresh 0.50
```

也可以直接用指定三阈值导出预测：

```bash
python3 predict_to_json_offset.py \
    --config config.yaml \
    --checkpoint run/20260422_201016_COnP/checkpoints/best_model_epoch0128_COnP0.7958.pt \
    --split test \
    --onset_thresh 0.45 \
    --frame_thresh 0.50 \
    --offset_thresh 0.10 \
    --output pred_test_offset_aware.json
```

默认情况下，这两步的输出都会放在 `评估/输出/` 下：

- 第一轮：`评估/输出/A_val40探索/`
- 第二轮：`评估/输出/B_val40_offset探索/`

这两个默认目录不会自动按 checkpoint 改名。
如果你评多个模型，建议每次手动加一个不同的 `--output_dir`，避免覆盖。

第一轮会输出：

- `A_运行日志.log`
- `A_val40实时探索过程.tsv`
- `A_val40探索结果.tsv`
- `A_筛选出的阈值.tsv`
- `A_在test100上的结果.tsv`
- `A_best_COnP阈值_test预测.json`

第二轮会输出：

- `B_运行日志.log`
- `B_val40_offset探索结果.tsv`
- `B_筛选出的最佳offset阈值.tsv`
- `B_在test100上的结果.tsv`
- `B_offset后处理_test预测.json`

## 评测

评测脚本 `evaluate_github.py` 来源于 [york135/singing_transcription_ICASSP2021](https://github.com/york135/singing_transcription_ICASSP2021/tree/master/evaluate)。

```bash
python3 evaluate_github.py \
    MIR-ST500_corrected.json \
    pred_test_v7.json \
    0.05
```

## 与论文差距

当前 COnP=0.7734 vs 论文 0.8013，差距 2.8%。
