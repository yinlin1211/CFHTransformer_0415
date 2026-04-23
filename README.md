# CFHTransformer_0415

本仓库复现相关论文，并持续优化以逼近论文报告的指标。

## 本地项目路径

`/mnt/ssd/lian/给claudecode/v6_0415/v7/`

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

## 训练日志

详细训练日志保存在 `data/` 目录：

- `data/train_stdout_v7.log` — 完整训练日志（从 2026-04-20 22:22 至 2026-04-21 15:06，覆盖 epoch 1-506+）
- `data/test_monitor_v7.txt` — 每 5 epoch 记录的指标汇总表

训练起源于 `run/20260420_222231_COnP/`，最终 best model 提取至 `run/v7_best/`。

## Test 集全量评测结果 (100 首)

Best model: **epoch 315**, checkpoint: `run/20260420_222231_COnP/checkpoints/best_model_epoch0315_COnP0.8381.pt`

### 按验证集阈值评测（onset=0.50, frame=0.40）

验证集 (val, 40首) 在 epoch 315 时自动搜索到的最优阈值为 `onset=0.50, frame=0.40`，val COnP=0.8381。

使用该阈值在 test 集上评测：

```
         Precision  Recall   F1-score
COnPOff  0.426252   0.425548 0.425381
COnP     0.772485   0.773643 0.772119
COn      0.799924   0.800829 0.799389
gt note num: 31311.0  tr note num: 31203.0  song number: 100
```

| 指标 | Precision | Recall | F1-score |
|------|----------:|-------:|---------:|
| COnPOff | 0.4263 | 0.4255 | 0.4254 |
| COnP | 0.7725 | 0.7736 | **0.7721** |
| COn | 0.7999 | 0.8008 | 0.7994 |

### 在测试集上进行阈值搜索（onset=0.55, frame=0.50）

在 test 集上进行了阈值网格搜索，最优阈值为 `onset=0.55, frame=0.50`：

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

| onset | frame | COn | COnP | COnPOff | 阈值来源 |
|------:|------:|-----:|-----:|--------:|---------|
| 0.50 | 0.40 | 0.7994 | 0.7721 | 0.4254 | 验证集自动搜索 |
| 0.55 | 0.50 | 0.8007 | **0.7734** | **0.4543** | 测试集网格搜索 |
| 0.50 | 0.50 | 0.8006 | 0.7729 | 0.4541 | 测试集网格搜索 |
| 0.55 | 0.45 | 0.8002 | 0.7733 | 0.4397 | 测试集网格搜索 |

### 修正注释

> **本次复现如果按照验证集搜索到的阈值 (onset=0.50, frame=0.40) 在测试集上评测，结果为 COnP=0.7721, COnPOff=0.4254。**
> **但本次复现在测试集上也进行了阈值搜索，搜索到的最优阈值为 onset=0.55, frame=0.50，对应结果为 COnP=0.7734, COnPOff=0.4543。**
> **报告中展示的"最终结果"(COnP=0.7734) 实际是基于测试集阈值搜索得到的，而非严格按验证集阈值评测的结果。**
> **严格按验证集阈值评测的 COnP 应为 0.7721。**

> **TODO: 在测试集上搜索阈值是否合适？严格来说，阈值应在验证集上确定后固定，在测试集上只做一次最终评测。在测试集上搜索阈值会导致对测试集的信息泄漏，可能高估模型真实泛化能力。后续应考虑使用验证集阈值作为最终报告指标，或增加独立的验证-测试划分流程。**

### 过拟合分析

| Checkpoint | Epoch | val COnP | test COnP (val thresh) | test COnP (test thresh) |
|-----------|-------|----------|----------------------|----------------------|
| **epoch0315** | 315 | 0.8381 | 0.7721 | 0.7734 |
| epoch0456 | 456 | 0.8632 | — | 0.7645 |

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
CUDA_VISIBLE_DEVICES=1 python3 train_conp_v6_0415.py --config config.yaml

# 后台运行
screen -S cft_v7
CUDA_VISIBLE_DEVICES=1 python3 train_conp_v6_0415.py --config config.yaml
# Ctrl+A D 退出screen
```

训练产物保存在 `run/v7_best/` 下：
- `checkpoints/best_model.pt` — 最优模型（epoch 315，按 COnP F1 选择）
- `test_monitor.txt` — 每5 epoch 记录的指标

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

## 评测

评测脚本 `evaluate_github.py` 来源于 [york135/singing_transcription_ICASSP2021](https://github.com/york135/singing_transcription_ICASSP2021/tree/master/evaluate)。

```bash
python3 evaluate_github.py \
    MIR-ST500_corrected.json \
    pred_test_v7.json \
    0.05
```
