"""
第一轮：只在 val40 上搜索最佳 onset / frame 阈值。

重要原则：
  - 阈值选择只使用 val40 的指标。
  - test100 不参与阈值搜索、不参与选阈值。
  - 脚本最后输出 test100 结果，仅用于报告当前阈值下的 test 表现，
    方便查看以及和后续实验比较稳定性。

流程：
  - 读取缓存好的 CQT npy
  - 用 predict_from_npy() 做重叠滑窗推理
  - 只用 onset / frame 两个阈值解码，不使用 offset head
  - 在 val40 上搜索 onset / frame
  - 用 val40 选出的若干组阈值，在 test100 上各评一次并保存报告结果

默认输出：
  - A_val40实时探索过程.tsv
  - A_val40探索结果.tsv
  - A_筛选出的阈值.tsv
  - A_在test100上的结果.tsv
  - A_best_COnP阈值_test预测.json
  - A_运行日志.log

使用说明：
  这个脚本评估的是你通过 --checkpoint 指定的模型。
  例如：
    python3 "评估/A在val40上探索最佳onset和frame阈值.py" \
        --config config.yaml \
        --checkpoint run/某个实验/checkpoints/某个模型.pt

  输出目录默认固定在：
    评估/输出/A_val40探索/

  如果你想区分不同模型的结果，建议手动传入不同的 --output_dir，例如：
    --output_dir "评估/输出/A_epoch0315"
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

from predict_to_json import frames_to_notes, predict_from_npy
from train_conp import compute_note_f1_single
from model import CFT_v6


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


def log(message, log_file=None):
    print(message, flush=True)
    if log_file is not None:
        log_file.write(message + "\n")
        log_file.flush()


def append_row(path, fieldnames, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def metric_row(onset, frame, metrics):
    return {
        "onset_thresh": f"{onset:.2f}",
        "frame_thresh": f"{frame:.2f}",
        "COn_f1": f"{metrics['COn']:.9f}",
        "COnP_f1": f"{metrics['COnP']:.9f}",
        "COnPOff_f1": f"{metrics['COnPOff']:.9f}",
        "COn_plus_COnP": f"{metrics['COn'] + metrics['COnP']:.9f}",
        "sum_all": f"{metrics['COn'] + metrics['COnP'] + metrics['COnPOff']:.9f}",
    }


def select_thresholds(rows):
    criteria = [
        ("best_COn", lambda r: (r["COn"], r["COnP"], r["COnPOff"])),
        ("best_COnP", lambda r: (r["COnP"], r["COn"], r["COnPOff"])),
        ("best_COnPOff", lambda r: (r["COnPOff"], r["COnP"], r["COn"])),
        ("best_COn_plus_COnP", lambda r: (r["COn"] + r["COnP"], r["COnPOff"])),
        (
            "best_COn_plus_COnP_plus_COnPOff",
            lambda r: (r["COn"] + r["COnP"] + r["COnPOff"], r["COnP"]),
        ),
    ]
    selected = []
    for criterion, key_fn in criteria:
        row = max(rows, key=key_fn)
        selected.append(
            {
                "criterion": criterion,
                "onset": row["onset"],
                "frame": row["frame"],
                "metrics": {
                    "COn": row["COn"],
                    "COnP": row["COnP"],
                    "COnPOff": row["COnPOff"],
                },
            }
        )
    return selected


def unique_selected(selected):
    seen = set()
    uniq = []
    for item in selected:
        key = (item["onset"], item["frame"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(item)
    return uniq


def infer_split(model, song_ids, config, gt_annotations, device, split_name):
    npy_dir = Path(config["data"]["cqt_cache_dir"])
    preds = []
    started = time.time()
    for idx, song_id in enumerate(song_ids):
        npy_path = npy_dir / f"{song_id}.npy"
        if not npy_path.exists():
            print(f"missing npy for {split_name} song {song_id}", flush=True)
            continue
        pred = predict_from_npy(model, str(npy_path), config, device)
        frame_prob, onset_prob = pred[0], pred[1]
        ref_intervals, ref_pitches = load_ref_notes(gt_annotations, song_id)
        preds.append((song_id, frame_prob, onset_prob, ref_intervals, ref_pitches))
        print(f"infer {split_name} [{idx + 1:3d}/{len(song_ids)}] song {song_id}", flush=True)
    print(f"infer {split_name} done in {time.time() - started:.1f}s", flush=True)
    return preds


def score_cached_predictions(preds, onset_thresh, frame_thresh, config):
    hop_length = config["audio"]["hop_length"]
    sample_rate = config["data"]["sample_rate"]
    con_scores = []
    conp_scores = []
    conpoff_scores = []
    pred_json = {}

    for song_id, frame_prob, onset_prob, ref_intervals, ref_pitches in preds:
        notes = frames_to_notes(
            frame_prob,
            onset_prob,
            hop_length,
            sample_rate,
            onset_thresh=onset_thresh,
            frame_thresh=frame_thresh,
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
    parser.add_argument(
        "--output_dir",
        default=str(SCRIPT_DIR / "输出/A_val40探索"),
    )
    parser.add_argument("--onset_min", type=float, default=0.05)
    parser.add_argument("--onset_max", type=float, default=1.00)
    parser.add_argument("--frame_min", type=float, default=0.05)
    parser.add_argument("--frame_max", type=float, default=1.00)
    parser.add_argument("--step", type=float, default=0.05)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open(config["data"]["label_path"]) as f:
        gt_annotations = json.load(f)

    splits_dir = Path(config["data"]["splits_dir"])
    with open(splits_dir / "val.txt") as f:
        val_song_ids = [line.strip() for line in f if line.strip()]
    with open(splits_dir / "test.txt") as f:
        test_song_ids = [line.strip() for line in f if line.strip()]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "A_运行日志.log"
    live_path = output_dir / "A_val40实时探索过程.tsv"
    live_fields = [
        "index",
        "total",
        "onset_thresh",
        "frame_thresh",
        "COn_f1",
        "COnP_f1",
        "COnPOff_f1",
        "combo_seconds",
        "elapsed_seconds",
    ]
    if live_path.exists():
        live_path.unlink()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with log_path.open("w") as log_file:
        started_all = time.time()
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

        val_preds = infer_split(model, val_song_ids, config, gt_annotations, device, "val")
        onset_thresholds = build_thresholds(args.onset_min, args.onset_max, args.step)
        frame_thresholds = build_thresholds(args.frame_min, args.frame_max, args.step)

        rows_raw = []
        rows = []
        total = len(onset_thresholds) * len(frame_thresholds)
        log(
            f"threshold grid {len(onset_thresholds)} x {len(frame_thresholds)} "
            f"= {total}",
            log_file,
        )
        combo_index = 0
        for onset_thresh in onset_thresholds:
            for frame_thresh in frame_thresholds:
                combo_index += 1
                combo_started = time.time()
                metrics = score_cached_predictions(val_preds, onset_thresh, frame_thresh, config)
                combo_seconds = time.time() - combo_started
                elapsed_seconds = time.time() - started_all
                row_raw = {
                    "onset": onset_thresh,
                    "frame": frame_thresh,
                    "COn": metrics["COn"],
                    "COnP": metrics["COnP"],
                    "COnPOff": metrics["COnPOff"],
                }
                rows_raw.append(row_raw)
                rows.append(metric_row(onset_thresh, frame_thresh, metrics))
                append_row(
                    live_path,
                    live_fields,
                    {
                        "index": combo_index,
                        "total": total,
                        "onset_thresh": f"{onset_thresh:.2f}",
                        "frame_thresh": f"{frame_thresh:.2f}",
                        "COn_f1": f"{metrics['COn']:.9f}",
                        "COnP_f1": f"{metrics['COnP']:.9f}",
                        "COnPOff_f1": f"{metrics['COnPOff']:.9f}",
                        "combo_seconds": f"{combo_seconds:.3f}",
                        "elapsed_seconds": f"{elapsed_seconds:.3f}",
                    },
                )
                log(
                    f"val_threshold [{combo_index:03d}/{total}] "
                    f"on={onset_thresh:.2f} fr={frame_thresh:.2f} "
                    f"COn={metrics['COn']:.6f} "
                    f"COnP={metrics['COnP']:.6f} "
                    f"COnPOff={metrics['COnPOff']:.6f} "
                    f"combo={combo_seconds:.2f}s elapsed={elapsed_seconds:.1f}s",
                    log_file,
                )
            best_so_far = max(rows_raw, key=lambda r: (r["COnP"], r["COn"], r["COnPOff"]))
            log(
                f"searched onset={onset_thresh:.2f}; "
                f"current best onset={best_so_far['onset']:.2f} frame={best_so_far['frame']:.2f} "
                f"val_COnP={best_so_far['COnP']:.6f}",
                log_file,
            )

        fields = [
            "onset_thresh",
            "frame_thresh",
            "COn_f1",
            "COnP_f1",
            "COnPOff_f1",
            "COn_plus_COnP",
            "sum_all",
        ]
        write_rows(output_dir / "A_val40探索结果.tsv", fields, rows)

        selected = select_thresholds(rows_raw)
        selected_rows = []
        for item in selected:
            row = {"criterion": item["criterion"]}
            row.update(metric_row(item["onset"], item["frame"], item["metrics"]))
            selected_rows.append(row)
        write_rows(output_dir / "A_筛选出的阈值.tsv", ["criterion"] + fields, selected_rows)

        test_preds = infer_split(model, test_song_ids, config, gt_annotations, device, "test")
        test_rows = []
        best_conp_item = next(item for item in selected if item["criterion"] == "best_COnP")
        best_conp_key = (best_conp_item["onset"], best_conp_item["frame"])
        for item in unique_selected(selected):
            test_metrics = score_cached_predictions(test_preds, item["onset"], item["frame"], config)
            row = {"criterion": item["criterion"]}
            row.update(metric_row(item["onset"], item["frame"], test_metrics))
            test_rows.append(row)
            if (item["onset"], item["frame"]) == best_conp_key:
                with (output_dir / "A_best_COnP阈值_test预测.json").open("w") as f:
                    json.dump(test_metrics["pred_json"], f, indent=2, ensure_ascii=False)
        write_rows(output_dir / "A_在test100上的结果.tsv", ["criterion"] + fields, test_rows)

        best_conp = max(selected, key=lambda x: x["metrics"]["COnP"])
        log(
            "BEST_VAL_BY_COnP "
            f"onset={best_conp['onset']:.2f} "
            f"frame={best_conp['frame']:.2f} "
            f"val_COn={best_conp['metrics']['COn']:.6f} "
            f"val_COnP={best_conp['metrics']['COnP']:.6f} "
            f"val_COnPOff={best_conp['metrics']['COnPOff']:.6f}",
            log_file,
        )
        for row in test_rows:
            if row["criterion"] == "best_COnP":
                log(
                    "TEST_WITH_BEST_VAL_COnP "
                    f"onset={row['onset_thresh']} frame={row['frame_thresh']} "
                    f"COn={row['COn_f1']} "
                    f"COnP={row['COnP_f1']} "
                    f"COnPOff={row['COnPOff_f1']}",
                    log_file,
                )


if __name__ == "__main__":
    main()
