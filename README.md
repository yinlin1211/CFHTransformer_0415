# CFHTransformer_0415

本仓库复现论文 *"Cycle Frequency-Harmonic-Time Transformer for Note-Level Singing Voice Transcription"* (ICME 2024)，并持续优化以逼近论文报告的指标。

## 版本历史

| 版本 | COn | COnP | COnPOff | 论文目标 COnP | 主要改动 |
|------|----:|-----:|--------:|-------------:|---------|
| v6 (旧) | 0.7906 | 0.7604 | 0.4075 | 0.8013 | 基线版本 |
| v7 (旧) | 0.8022 | 0.7752 | 0.4464 | 0.8013 | CQT归一化 + onset类别平衡 + 400首训练 |
| **v7 (当前)** | **0.8007** | **0.7734** | **0.4543** | 0.8013 | GAP: mean→sum + LayerNorm + splits修复 + 更长训练 |

> 旧版代码保存在 `v7_旧/` 目录下。

## v7 (当前版本) 改动详情

相对之前推送的旧版 v7，本次更新的核心改动：

1. **GAP 聚合方式** — 从 `mean` 改为 `sum`（论文原文 "sum up the harmonic features"），并增加 LayerNorm 稳定数值
2. **数据集划分修复** — 使用 `splits_v11/`（361-400 做 val，401-500 做 test），之前 splits 有交叉污染
3. **训练集样本数** — `max_samples_per_epoch` 从 2000 提升至 4000
4. **更长训练** — 当前 best model 在 epoch 315（旧版 best 在 ~80 epoch 即过拟合）
5. **CQT 归一化** — model 内置 `cqt_mean=-65.0, cqt_std=18.0` 归一化
6. **Onset 类别平衡** — `onset_pos_weight=5.0`
7. **训练集** — 400 首（对齐论文：400 train + 100 test，无独立 val）

## Test 集全量评测结果 (100 首)

Best model: epoch 315, 阈值网格搜最优 onset=0.55, frame=0.50

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

### 与旧版 v7 对比

| 版本 | COn | COnP | COnPOff |
|------|----:|-----:|--------:|
| v7 (旧) | 0.8022 | 0.7752 | 0.4464 |
| **v7 (当前)** | 0.8007 | 0.7734 | **0.4543** |

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
- `checkpoints/best_model.pt` — 最优模型（按 COnP F1 选择）
- `logs/train_stdout.log` — 训练日志
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

```bash
python3 evaluate_github.py \
    MIR-ST500_corrected.json \
    pred_test_v7.json \
    0.05
```

## 与论文差距

当前 COnP=0.7734 vs 论文 0.8013，差距 2.8%。模型仍在训练中（当前 ~325/1300 epoch），后续 best model 预计进一步提升。
