"""
第二轮：固定 onset / frame 后，只在 val40 上搜索最佳 offset 阈值。

重要原则：
  - onset / frame 应来自第一轮 val40 搜索结果。
  - offset 阈值选择只使用 val40 的 COnPOff 指标。
  - test100 不参与 offset 阈值搜索、不参与选阈值。
  - 脚本最后输出 test100 结果，仅用于报告当前三阈值下的 test 表现，
    方便查看以及和后续实验比较稳定性。

流程：
1. 先在 val40 上缓存 frame / onset / offset 三张概率图
2. 只搜索 offset_thresh
3. 按 val40 上的 COnPOff 选择最佳 offset
4. 用 val40 选出的三阈值在 test100 上评一次并保存报告结果
5. 保存日志、TSV 指标和最终 test 预测 JSON

使用说明：
  这个脚本评估的是你通过 --checkpoint 指定的模型。
  例如：
    python3 "评估/B在val40上探索最佳offset阈值.py" \
        --config config.yaml \
        --checkpoint run/某个实验/checkpoints/某个模型.pt \
        --onset_thresh 0.45 \
        --frame_thresh 0.50

  输出目录默认固定在：
    评估/输出/B_val40_offset探索/

  如果你想区分不同模型的结果，建议手动传入不同的 --output_dir，例如：
    --output_dir "评估/输出/B_epoch0315"
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

from predict_to_json_offset import frames_to_notes_offset, predict_from_npy
from train_conp import compute_note_f1_single
from model import CFT_v6


def log(message, log_file=None):
    print(message, flush=True)
    if log_file is not None:
        log_file.write(message + "\n")
        log_file.flush()


def notes_to_arrays(notes):
    if not notes:
        return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float)
    return (
        np.array([[float(n[0]), float(n[1])] for n in notes], dtype=float),
        np.array([float(n[2]) for n in notes], dtype=float),
    )


def load_ref_notes(gt_annotations, song_id):
    raw = gt_annotations[song_id]
    notes = [
        [float(n[0]), float(n[1]), float(n[2])]
        for n in raw
        if float(n[1]) - float(n[0]) > 0
    ]
    return notes_to_arrays(notes)


def build_thresholds(start, stop, step):
    return [round(float(x), 2) for x in np.arange(start, stop + step / 2.0, step)]


def write_rows(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def metric_row(offset, metrics, onset, frame):
    return {
        "onset_thresh": f"{onset:.2f}",
        "frame_thresh": f"{frame:.2f}",
        "offset_thresh": f"{offset:.2f}",
        "COn_f1": f"{metrics['COn']:.9f}",
        "COnP_f1": f"{metrics['COnP']:.9f}",
        "COnPOff_f1": f"{metrics['COnPOff']:.9f}",
        "COn_plus_COnP": f"{metrics['COn'] + metrics['COnP']:.9f}",
        "sum_all": f"{metrics['COn'] + metrics['COnP'] + metrics['COnPOff']:.9f}",
    }


def infer_split(model, song_ids, config, gt_annotations, device, split_name, log_file=None):
    npy_dir = Path(config["data"]["cqt_cache_dir"])
    predictions = []
    started = time.time()
    for idx, song_id in enumerate(song_ids):
        npy_path = npy_dir / f"{song_id}.npy"
        if not npy_path.exists():
            log(f"missing npy for {split_name} song {song_id}", log_file)
            continue
        frame_prob, onset_prob, offset_prob = predict_from_npy(model, str(npy_path), config, device)
        ref_intervals, ref_pitches = load_ref_notes(gt_annotations, song_id)
        predictions.append((song_id, frame_prob, onset_prob, offset_prob, ref_intervals, ref_pitches))
        log(f"infer {split_name} [{idx + 1:3d}/{len(song_ids)}] song {song_id}", log_file)
    log(f"infer {split_name} done in {time.time() - started:.1f}s", log_file)
    return predictions


def score_cached_predictions(predictions, config, onset_thresh, frame_thresh, offset_thresh):
    hop_length = config["audio"]["hop_length"]
    sample_rate = config["data"]["sample_rate"]
    con_scores = []
    conp_scores = []
    conpoff_scores = []
    pred_json = {}

    for song_id, frame_prob, onset_prob, offset_prob, ref_intervals, ref_pitches in predictions:
        notes = frames_to_notes_offset(
            frame_prob,
            onset_prob,
            offset_prob,
            hop_length,
            sample_rate,
            onset_thresh=onset_thresh,
            frame_thresh=frame_thresh,
            offset_thresh=offset_thresh,
        )
        pred_json[song_id] = notes
        pred_intervals, pred_pitches = notes_to_arrays(notes)
        con, conp, conpoff = compute_note_f1_single(
            pred_intervals, pred_pitches, ref_intervals, ref_pitches
        )
        if conp is not None:
            con_scores.append(con)
            conp_scores.append(conp)
            conpoff_scores.append(conpoff)

    return {
        "COn": float(np.mean(con_scores)) if con_scores else 0.0,
        "COnP": float(np.mean(conp_scores)) if conp_scores else 0.0,
        "COnPOff": float(np.mean(conpoff_scores)) if conpoff_scores else 0.0,
        "pred_json": pred_json,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT_DIR / "config.yaml"))
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT_DIR / "run/20260422_201016_COnP/checkpoints/best_model.pt"),
    )
    parser.add_argument("--onset_thresh", type=float, required=True)
    parser.add_argument("--frame_thresh", type=float, required=True)
    parser.add_argument("--offset_min", type=float, default=0.05)
    parser.add_argument("--offset_max", type=float, default=1.00)
    parser.add_argument("--offset_step", type=float, default=0.05)
    parser.add_argument(
        "--output_dir",
        default=str(SCRIPT_DIR / "输出/B_val40_offset探索"),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open(config["data"]["label_path"]) as f:
        gt_annotations = json.load(f)

    splits_dir = Path(config["data"]["splits_dir"])
    with open(splits_dir / "val.txt") as f:
        val_song_ids = [line.strip() for line in f if line.strip()]
    with open(splits_dir / "test.txt") as f:
        test_song_ids = [line.strip() for line in f if line.strip()]

    log_path = output_dir / "B_运行日志.log"
    with log_path.open("w") as log_file:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log(f"device {device}", log_file)

        model = CFT_v6(config).to(device)
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        log(
            "checkpoint "
            f"path={args.checkpoint} "
            f"epoch={ckpt.get('epoch')} "
            f"best_conp={ckpt.get('best_conp_f1')} "
            f"best_onset={ckpt.get('best_onset_thresh')} "
            f"best_frame={ckpt.get('best_frame_thresh')}",
            log_file,
        )

        val_predictions = infer_split(model, val_song_ids, config, gt_annotations, device, "val", log_file)
        offset_thresholds = build_thresholds(args.offset_min, args.offset_max, args.offset_step)
        log(f"offset grid {len(offset_thresholds)} values: {offset_thresholds}", log_file)

        val_rows_raw = []
        val_rows = []
        for offset_thresh in offset_thresholds:
            metrics = score_cached_predictions(
                val_predictions,
                config,
                args.onset_thresh,
                args.frame_thresh,
                offset_thresh,
            )
            val_rows_raw.append(
                {
                    "offset": offset_thresh,
                    "COn": metrics["COn"],
                    "COnP": metrics["COnP"],
                    "COnPOff": metrics["COnPOff"],
                }
            )
            val_rows.append(metric_row(offset_thresh, metrics, args.onset_thresh, args.frame_thresh))
            log(
                f"VAL off={offset_thresh:.2f} "
                f"COn={metrics['COn']:.9f} "
                f"COnP={metrics['COnP']:.9f} "
                f"COnPOff={metrics['COnPOff']:.9f}",
                log_file,
            )

        fields = [
            "onset_thresh",
            "frame_thresh",
            "offset_thresh",
            "COn_f1",
            "COnP_f1",
            "COnPOff_f1",
            "COn_plus_COnP",
            "sum_all",
        ]
        write_rows(output_dir / "B_val40_offset探索结果.tsv", fields, val_rows)

        best = max(val_rows_raw, key=lambda r: (r["COnPOff"], r["COnP"], r["COn"]))
        best_metrics = {"COn": best["COn"], "COnP": best["COnP"], "COnPOff": best["COnPOff"]}
        selected_row = {"criterion": "best_val_COnPOff_fixed_best_COnP_onset_frame"}
        selected_row.update(metric_row(best["offset"], best_metrics, args.onset_thresh, args.frame_thresh))
        write_rows(output_dir / "B_筛选出的最佳offset阈值.tsv", ["criterion"] + fields, [selected_row])
        log(
            "SELECTED_BY_VAL "
            f"off={best['offset']:.2f} "
            f"COn={best['COn']:.9f} "
            f"COnP={best['COnP']:.9f} "
            f"COnPOff={best['COnPOff']:.9f}",
            log_file,
        )

        test_predictions = infer_split(model, test_song_ids, config, gt_annotations, device, "test", log_file)
        test_metrics = score_cached_predictions(
            test_predictions,
            config,
            args.onset_thresh,
            args.frame_thresh,
            best["offset"],
        )
        test_row = {"criterion": "test_with_val_selected_offset"}
        test_row.update(metric_row(best["offset"], test_metrics, args.onset_thresh, args.frame_thresh))
        write_rows(output_dir / "B_在test100上的结果.tsv", ["criterion"] + fields, [test_row])

        pred_path = output_dir / "B_offset后处理_test预测.json"
        with pred_path.open("w") as f:
            json.dump(test_metrics["pred_json"], f, indent=2, ensure_ascii=False)

        log(
            "TEST "
            f"on={args.onset_thresh:.2f} "
            f"fr={args.frame_thresh:.2f} "
            f"off={best['offset']:.2f} "
            f"COn={test_metrics['COn']:.9f} "
            f"COnP={test_metrics['COnP']:.9f} "
            f"COnPOff={test_metrics['COnPOff']:.9f}",
            log_file,
        )


if __name__ == "__main__":
    main()
