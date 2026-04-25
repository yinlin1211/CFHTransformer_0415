"""
Val-only onset/frame threshold search, then test-only verification.

This script is the standalone version of the val40 threshold search workflow.
It follows the same inference path as the current repo:
  - read cached CQT npy
  - run overlap inference via predict_from_npy()
  - decode with onset/frame thresholds only (ignore offset head)

Outputs:
  - val_threshold_search.tsv
  - selected_thresholds.tsv
  - test_with_selected_thresholds.tsv
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
from train_conp_v6_0415 import compute_note_f1_single
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
        frame_prob, onset_prob, _ = predict_from_npy(model, str(npy_path), config, device)
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
        default=str(ROOT_DIR / "run/20260422_201016_COnP/checkpoints/best_model_epoch0128_COnP0.7958.pt"),
    )
    parser.add_argument(
        "--output_dir",
        default=str(ROOT_DIR / "run/20260422_201016_COnP/threshold_search_v2_epoch0128"),
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}", flush=True)

    model = CFT_v6(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(
        "checkpoint "
        f"epoch={ckpt.get('epoch')} "
        f"best_conp={ckpt.get('best_conp_f1')} "
        f"best_onset={ckpt.get('best_onset_thresh')} "
        f"best_frame={ckpt.get('best_frame_thresh')}",
        flush=True,
    )

    val_preds = infer_split(model, val_song_ids, config, gt_annotations, device, "val")
    onset_thresholds = build_thresholds(args.onset_min, args.onset_max, args.step)
    frame_thresholds = build_thresholds(args.frame_min, args.frame_max, args.step)

    rows_raw = []
    rows = []
    print(
        f"threshold grid {len(onset_thresholds)} x {len(frame_thresholds)} "
        f"= {len(onset_thresholds) * len(frame_thresholds)}",
        flush=True,
    )
    for onset_thresh in onset_thresholds:
        for frame_thresh in frame_thresholds:
            metrics = score_cached_predictions(val_preds, onset_thresh, frame_thresh, config)
            rows_raw.append(
                {
                    "onset": onset_thresh,
                    "frame": frame_thresh,
                    "COn": metrics["COn"],
                    "COnP": metrics["COnP"],
                    "COnPOff": metrics["COnPOff"],
                }
            )
            rows.append(metric_row(onset_thresh, frame_thresh, metrics))
        best_so_far = max(rows_raw, key=lambda r: (r["COnP"], r["COn"], r["COnPOff"]))
        print(
            f"searched onset={onset_thresh:.2f}; "
            f"current best onset={best_so_far['onset']:.2f} frame={best_so_far['frame']:.2f} "
            f"val_COnP={best_so_far['COnP']:.6f}",
            flush=True,
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
    write_rows(output_dir / "val_threshold_search.tsv", fields, rows)

    selected = select_thresholds(rows_raw)
    selected_rows = []
    for item in selected:
        row = {"criterion": item["criterion"]}
        row.update(metric_row(item["onset"], item["frame"], item["metrics"]))
        selected_rows.append(row)
    write_rows(output_dir / "selected_thresholds.tsv", ["criterion"] + fields, selected_rows)

    test_preds = infer_split(model, test_song_ids, config, gt_annotations, device, "test")
    test_rows = []
    for item in unique_selected(selected):
        test_metrics = score_cached_predictions(test_preds, item["onset"], item["frame"], config)
        row = {"criterion": item["criterion"]}
        row.update(metric_row(item["onset"], item["frame"], test_metrics))
        test_rows.append(row)
        if item["criterion"] == "best_COnP":
            with (output_dir / "pred_test_best_COnP_thresholds.json").open("w") as f:
                json.dump(test_metrics["pred_json"], f, indent=2, ensure_ascii=False)
    write_rows(output_dir / "test_with_selected_thresholds.tsv", ["criterion"] + fields, test_rows)

    best_conp = max(selected, key=lambda x: x["metrics"]["COnP"])
    print(
        "BEST_VAL_BY_COnP "
        f"onset={best_conp['onset']:.2f} "
        f"frame={best_conp['frame']:.2f} "
        f"val_COn={best_conp['metrics']['COn']:.6f} "
        f"val_COnP={best_conp['metrics']['COnP']:.6f} "
        f"val_COnPOff={best_conp['metrics']['COnPOff']:.6f}",
        flush=True,
    )
    for row in test_rows:
        if row["criterion"] == "best_COnP":
            print(
                "TEST_WITH_BEST_VAL_COnP "
                f"onset={row['onset_thresh']} frame={row['frame_thresh']} "
                f"COn={row['COn_f1']} "
                f"COnP={row['COnP_f1']} "
                f"COnPOff={row['COnPOff_f1']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
