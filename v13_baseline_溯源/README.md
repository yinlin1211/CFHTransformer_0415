# CFT / CFH Transformer MIR-ST500 复现

本仓库用于复现和验证论文 "Cycle Frequency-Harmonic-Time Transformer for Note-Level Singing Voice Transcription" 的 MIR-ST500 声乐音符转写实验。当前目录是一次 baseline 验证版本，以下说明以本目录实际代码为准。

注意：旧入口 `train_conp_v6_0415.py` 当前不存在，训练入口是 `train_conp.py`。

## 当前状态

- 当前训练已在 2026-04-28 手动中断。
- 当前实验目录：`run/20260428_031026_COnP/`
- 最优模型：`run/20260428_031026_COnP/checkpoints/best_model.pt`
- 最优模型来自 epoch 170，val COnP F1 = 0.8052，COn F1 = 0.8195，COnPOff F1 = 0.5004。
- 最优模型记录的阈值：`onset_thresh=0.50`，`frame_thresh=0.40`。
- 最新完整 checkpoint：`run/20260428_031026_COnP/checkpoints/latest.pt`，保存到 epoch 176。
- `test_monitor.txt` 中 epoch 170 的 full test 监控结果：COn F1 = 0.7988，COnP F1 = 0.7693，COnPOff F1 = 0.4424。

`test_monitor.txt` 只是训练过程诊断，不参与选模。最终报告建议重新用 `predict_to_json.py` 导出 JSON，再用 `evaluate_github.py` 评测。

## 环境与数据

已验证环境：

- Python 3.12.3
- PyTorch 2.5.1+cu121
- numpy 1.26.4
- librosa 0.11.0
- mir_eval 0.8.2
- PyYAML 6.0.1

`config.yaml` 当前依赖以下外部数据路径：

- CQT 缓存：`/mnt/ssd/lian/论文复现/CFH-Transformer/cqt_cache_50ms/npy/`
- 标注文件：`/mnt/ssd/lian/论文复现/CFH-Transformer/MIR-ST500_corrected.json`
- split 目录：`/mnt/ssd/lian/给claudecode/v6_0415/splits_v11/`

本目录也保留了 `data/MIR-ST500_corrected.json` 和 `splits_v11/` 副本。如果移动项目，需要同步修改 `config.yaml` 中的绝对路径。

当前 split 的实际内容：

- `train.txt`：1-400，共 400 首
- `val.txt`：361-400，共 40 首
- `test.txt`：401-500，共 100 首

因此当前 `val40` 与训练集后 40 首重叠，主要用于训练过程阈值搜索和监控；严格对外比较时应重点看 `test100`。

## 主要入口

- `train_conp.py`：训练入口，best model 按 val COnP F1 保存。
- `predict_to_json.py`：两阈值推理，只使用 onset/frame 后处理。
- `predict_to_json_offset.py`：三阈值推理，额外使用 offset head。
- `evaluate_github.py`：用原评测格式计算 COn、COnP、COnPOff。
- `model.py`：模型 `CFT_v6` 和损失 `CFTLoss`。
- `dataset.py`：MIR-ST500 CQT 缓存和标签加载。

## 训练

新开训练：

```bash
CUDA_VISIBLE_DEVICES=2 python3 -u train_conp.py \
    --config config.yaml \
    2>&1 | tee train_$(date +%Y%m%d_%H%M%S).log
```

从当前 latest 继续：

```bash
CUDA_VISIBLE_DEVICES=2 python3 -u train_conp.py \
    --config config.yaml \
    --resume run/20260428_031026_COnP/checkpoints/latest.pt
```

训练会自动创建 `run/<timestamp>_COnP/`，主要产物包括：

- `checkpoints/best_model.pt`
- `checkpoints/latest.pt`
- `checkpoints/best_model_epochXXXX_COnP*.pt`
- `checkpoints/checkpoint_epochXXXX.pt`
- `logs/train_stdout.log`
- `test_monitor.txt`

训练脚本每 5 个 epoch 重新搜索一次 onset/frame 阈值，best model 按 val COnP F1 选择。每 40 个 epoch 会做一次 full test 监控，epoch 20 后出现新 best 也会额外监控一次。

## 推理与评测

用当前 best model 导出 test100 预测：

```bash
CUDA_VISIBLE_DEVICES=0 python3 predict_to_json.py \
    --config config.yaml \
    --checkpoint run/20260428_031026_COnP/checkpoints/best_model.pt \
    --split test \
    --onset_thresh 0.50 \
    --frame_thresh 0.40 \
    --output pred_test_epoch0170.json
```

评测：

```bash
python3 evaluate_github.py \
    /mnt/ssd/lian/论文复现/CFH-Transformer/MIR-ST500_corrected.json \
    pred_test_epoch0170.json \
    0.05
```

也可以评估 val40：

```bash
CUDA_VISIBLE_DEVICES=0 python3 predict_to_json.py \
    --config config.yaml \
    --checkpoint run/20260428_031026_COnP/checkpoints/best_model.pt \
    --split val \
    --onset_thresh 0.50 \
    --frame_thresh 0.40 \
    --output pred_val_epoch0170.json
```

## Offset-aware 推理

如果需要直接使用 offset head，可以指定三阈值导出：

```bash
CUDA_VISIBLE_DEVICES=0 python3 predict_to_json_offset.py \
    --config config.yaml \
    --checkpoint run/20260428_031026_COnP/checkpoints/best_model.pt \
    --split test \
    --onset_thresh 0.50 \
    --frame_thresh 0.40 \
    --offset_thresh 0.30 \
    --output pred_test_epoch0170_offset.json
```

这里的 `offset_thresh=0.30` 只是默认示例，不是当前目录重新搜索得到的最优值。

## Val40 阈值搜索和 Test100 报告

阈值选择只看 `val40`，`test100` 不参与搜索和选阈值。脚本输出的 test100 结果仅用于报告当前 val40 阈值下的 test 表现，方便查看以及和后续实验比较稳定性。

针对 epoch 128 checkpoint，在 GPU 2 上先跑 val40 的 onset/frame 二维阈值搜索。脚本结束时会额外用 val40 选出的阈值在 test100 上评一次并保存报告结果：

```bash
CUDA_VISIBLE_DEVICES=2 python3 "评估/A在val40上探索最佳onset和frame阈值.py" \
    --config config.yaml \
    --checkpoint run/20260428_031026_COnP/checkpoints/best_model_epoch0128_COnP0.7952.pt \
    --output_dir "评估/输出/A_epoch0128"
```

第一轮输出重点看：

- `评估/输出/A_epoch0128/A_筛选出的阈值.tsv`
- `评估/输出/A_epoch0128/A_在test100上的结果.tsv`
- `评估/输出/A_epoch0128/A_best_COnP阈值_test预测.json`（脚本已修复；如果 best_COn 与 best_COnP 同阈值，复跑也会生成）

其中 `A_筛选出的阈值.tsv` 是 val40 选阈值结果；`A_在test100上的结果.tsv` 和 `A_best_COnP阈值_test预测.json` 是 test100 报告结果，只用于查看稳定性和后续实验对比，不回流参与下一阶段选阈值。

如果要继续做 offset-aware 三阈值流程，先从 `A_筛选出的阈值.tsv` 里拿到 `best_COnP` 对应的 `onset_thresh` 和 `frame_thresh`，再把下面命令中的 `0.50`、`0.40` 替换成实际选出的值：

```bash
CUDA_VISIBLE_DEVICES=2 python3 "评估/B在val40上探索最佳offset阈值.py" \
    --config config.yaml \
    --checkpoint run/20260428_031026_COnP/checkpoints/best_model_epoch0128_COnP0.7952.pt \
    --onset_thresh 0.50 \
    --frame_thresh 0.40 \
    --output_dir "评估/输出/B_epoch0128"
```

第二轮输出重点看：

- `评估/输出/B_epoch0128/B_筛选出的最佳offset阈值.tsv`
- `评估/输出/B_epoch0128/B_在test100上的结果.tsv`
- `评估/输出/B_epoch0128/B_offset后处理_test预测.json`

其中 `B_筛选出的最佳offset阈值.tsv` 是 val40 选 offset 阈值结果；`B_在test100上的结果.tsv` 和 `B_offset后处理_test预测.json` 是用 val40 选出的三阈值在 test100 上得到的报告结果，只用于查看稳定性和后续实验对比。

### epoch0128 已完成实验结果

使用 checkpoint：

```text
run/20260428_031026_COnP/checkpoints/best_model_epoch0128_COnP0.7952.pt
```

第一轮 A：只在 val40 上搜索 onset/frame，网格为 `0.05` 到 `1.00`，步长 `0.05`，共 `20 x 20 = 400` 组。val40 按 COnP 选择出的阈值是 `onset=0.40, frame=0.45`。

| 阶段 | 选择标准 | onset | frame | offset | COn | COnP | COnPOff |
|------|----------|------:|------:|-------:|----:|-----:|--------:|
| A val40 | best_COnP | 0.40 | 0.45 | - | 0.818046 | 0.802485 | 0.501359 |
| A test100 报告 | val best_COnP 阈值 | 0.40 | 0.45 | - | 0.803632 | 0.775924 | 0.455659 |

补充：A 阶段如果单看 val40 的 COnPOff，最佳是 `onset=0.40, frame=0.80`，val COnPOff 为 `0.588116`，但 val COnP 降到 `0.792388`。本流程继续使用 `best_COnP` 的 `0.40/0.45` 进入第二轮。

第二轮 B：固定第一轮 val40 选出的 `onset=0.40, frame=0.45`，只在 val40 上搜索 offset，网格为 `0.05` 到 `1.00`，步长 `0.05`。val40 按 COnPOff 选择出的 offset 是 `0.15`。

| 阶段 | 选择标准 | onset | frame | offset | COn | COnP | COnPOff |
|------|----------|------:|------:|-------:|----:|-----:|--------:|
| B val40 | best_COnPOff | 0.40 | 0.45 | 0.15 | 0.817418 | 0.802469 | 0.619877 |
| B test100 最终报告 | val 选三阈值 | 0.40 | 0.45 | 0.15 | 0.802668 | 0.775409 | 0.588179 |

最终采用三阈值：

```text
onset_thresh=0.40
frame_thresh=0.45
offset_thresh=0.15
```

最终 test100 预测文件：

```text
评估/输出/B_epoch0128/B_offset后处理_test预测.json
```

该 JSON 覆盖 100 首 test 歌曲，共预测 31,359 个音符。相对 A 阶段只用 onset/frame 的 test100 报告，B 阶段 offset-aware 后处理将 test COnPOff 从 `0.455659` 提升到 `0.588179`，COnP 基本保持在 `0.775` 附近。

## 常用检查命令

查看训练日志：

```bash
tail -f train_tmux_gpu2_train_conp_20260428_031021.log
```

查看 checkpoint 目录：

```bash
ls -lh run/20260428_031026_COnP/checkpoints/
```

查看 GPU：

```bash
nvidia-smi
```

查看脚本参数：

```bash
python3 train_conp.py --help
python3 predict_to_json.py --help
python3 predict_to_json_offset.py --help
```
