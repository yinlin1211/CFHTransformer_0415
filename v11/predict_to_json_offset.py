"""
Offset-aware CFT_v6 inference.

This script keeps the original output format used by evaluate_github.py:
  {"401": [[onset_sec, offset_sec, midi_pitch], ...], ...}

Default experiment:
  python3 predict_to_json_offset.py

which uses:
  checkpoint: run/20260422_201016_COnP/checkpoints/best_model_epoch0128_COnP0.7958.pt
  onset_thresh=0.50, frame_thresh=0.40, offset_thresh=0.30
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from model import CFT_v6 as CFT


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

MIDI_MIN = 36


def pick_peaks(curve, thresh):
    """Collapse each contiguous above-threshold region to its strongest frame."""
    candidates = np.where(curve > thresh)[0]
    if len(candidates) == 0:
        return candidates

    picked = []
    start = prev = int(candidates[0])
    for frame in candidates[1:]:
        frame = int(frame)
        if frame == prev + 1:
            prev = frame
            continue
        local = curve[start:prev + 1]
        picked.append(start + int(np.argmax(local)))
        start = prev = frame

    local = curve[start:prev + 1]
    picked.append(start + int(np.argmax(local)))
    return np.array(picked, dtype=np.int64)


def estimate_frame_end(frame_curve, start, stop, frame_thresh, max_gap):
    """Fallback end from the frame branch: last active frame before a gap."""
    last_active = start
    gap = 0

    for t in range(start, stop):
        if frame_curve[t] > frame_thresh:
            last_active = t
            gap = 0
        else:
            gap += 1
            if gap > max_gap and t > start + 1:
                break

    return min(last_active + 1, stop)


def frames_to_notes_offset(frame_pred, onset_pred, offset_pred,
                           hop_length, sample_rate,
                           onset_thresh=0.50, frame_thresh=0.40,
                           offset_thresh=0.30, min_note_len=2,
                           max_gap=2):
    """
    Decode notes using offset head for note endings.

    For each pitch:
    1. onset peaks create candidate note starts.
    2. the first offset peak after that onset and before the next same-pitch onset
       becomes the note end.
    3. if no offset peak exists, fall back to frame-continuity end estimation.
    """
    frame_time = hop_length / sample_rate
    T, P = frame_pred.shape
    notes = []

    for p in range(P):
        midi = p + MIDI_MIN
        onset_frames = pick_peaks(onset_pred[:, p], onset_thresh)
        offset_frames = pick_peaks(offset_pred[:, p], offset_thresh)

        if len(onset_frames) == 0:
            active = frame_pred[:, p] > frame_thresh
            in_note = False
            note_start = 0
            for t in range(T):
                if active[t] and not in_note:
                    in_note = True
                    note_start = t
                elif not active[t] and in_note:
                    in_note = False
                    if t - note_start >= min_note_len:
                        notes.append([note_start * frame_time, t * frame_time, float(midi)])
            if in_note and T - note_start >= min_note_len:
                notes.append([note_start * frame_time, T * frame_time, float(midi)])
            continue

        for i, f_on in enumerate(onset_frames):
            next_onset = int(onset_frames[i + 1]) if i + 1 < len(onset_frames) else T
            search_start = int(f_on) + min_note_len
            search_stop = min(next_onset, T)

            valid_offsets = offset_frames[
                (offset_frames >= search_start) & (offset_frames < search_stop)
            ]
            if len(valid_offsets) > 0:
                end_frame = int(valid_offsets[0])
            else:
                end_frame = estimate_frame_end(
                    frame_pred[:, p], int(f_on), search_stop, frame_thresh, max_gap
                )

            if end_frame - int(f_on) >= min_note_len:
                notes.append([int(f_on) * frame_time, end_frame * frame_time, float(midi)])

    return notes


def predict_from_npy(model, npy_path, config, device):
    """Return frame/onset/offset probability maps, each shape (T, 48)."""
    cqt = np.load(npy_path)
    cqt_tensor = torch.from_numpy(cqt).float().unsqueeze(0).to(device)
    segment_frames = config["data"]["segment_frames"]
    T = cqt.shape[1]

    onset_map = np.zeros((T, 48), dtype=np.float32)
    frame_map = np.zeros((T, 48), dtype=np.float32)
    offset_map = np.zeros((T, 48), dtype=np.float32)
    count_map = np.zeros(T, dtype=np.float32)
    step = segment_frames // 2

    model.eval()
    with torch.no_grad():
        for start in range(0, T, step):
            seg = cqt_tensor[:, :, start:start + segment_frames]
            if seg.shape[2] < segment_frames:
                pad = segment_frames - seg.shape[2]
                seg = torch.nn.functional.pad(seg, (0, pad), value=-80.0)

            onset_logit, frame_logit, offset_logit = model(seg)
            onset_prob = torch.sigmoid(onset_logit[0]).cpu().numpy()
            frame_prob = torch.sigmoid(frame_logit[0]).cpu().numpy()
            offset_prob = torch.sigmoid(offset_logit[0]).cpu().numpy()

            actual = min(segment_frames, T - start)
            onset_map[start:start + actual] += onset_prob[:actual]
            frame_map[start:start + actual] += frame_prob[:actual]
            offset_map[start:start + actual] += offset_prob[:actual]
            count_map[start:start + actual] += 1

    count_map = np.maximum(count_map, 1.0)
    onset_map /= count_map[:, None]
    frame_map /= count_map[:, None]
    offset_map /= count_map[:, None]
    return frame_map, onset_map, offset_map


def main():
    parser = argparse.ArgumentParser(description="Offset-aware CFT_v6 inference")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--checkpoint",
        default="run/20260422_201016_COnP/checkpoints/best_model_epoch0128_COnP0.7958.pt",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--onset_thresh", type=float, default=0.50)
    parser.add_argument("--frame_thresh", type=float, default=0.40)
    parser.add_argument("--offset_thresh", type=float, default=0.30)
    parser.add_argument("--output", default="pred_test_epoch0128_offset.json")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    model = CFT(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    log.info(f"Checkpoint: {args.checkpoint}")
    log.info(f"Checkpoint epoch={ckpt.get('epoch', '?')}, COnP={ckpt.get('COnP_f1', 'N/A')}")

    splits_dir = Path(config["data"]["splits_dir"])
    with open(splits_dir / f"{args.split}.txt") as f:
        song_ids = [line.strip() for line in f if line.strip()]

    npy_dir = Path(config["data"]["cqt_cache_dir"])
    hop_length = config["audio"]["hop_length"]
    sample_rate = config["data"]["sample_rate"]

    log.info(
        f"Split={args.split}, songs={len(song_ids)}, "
        f"on={args.onset_thresh:.2f}, fr={args.frame_thresh:.2f}, off={args.offset_thresh:.2f}"
    )

    predictions = {}
    skipped = 0
    for idx, song_id in enumerate(song_ids):
        npy_path = npy_dir / f"{song_id}.npy"
        if not npy_path.exists():
            skipped += 1
            log.warning(f"[{idx + 1}/{len(song_ids)}] {song_id}: missing npy")
            continue

        frame_pred, onset_pred, offset_pred = predict_from_npy(
            model, str(npy_path), config, device
        )
        notes = frames_to_notes_offset(
            frame_pred, onset_pred, offset_pred,
            hop_length, sample_rate,
            onset_thresh=args.onset_thresh,
            frame_thresh=args.frame_thresh,
            offset_thresh=args.offset_thresh,
        )
        predictions[song_id] = notes
        log.info(f"[{idx + 1:3d}/{len(song_ids)}] song {song_id:>4s}: {len(notes):4d} notes")

    with open(args.output, "w") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    log.info(f"Saved: {args.output}; success={len(predictions)}, skipped={skipped}")
    log.info(f"Evaluate: python3 evaluate_github.py {config['data']['label_path']} {args.output} 0.05")


if __name__ == "__main__":
    main()
