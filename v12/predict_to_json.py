import sys
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from model import CFT_v6 as CFT

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

MIDI_MIN = 36

def pick_onset_frames(onset_curve, onset_thresh):

    candidates = np.where(onset_curve > onset_thresh)[0]
    if len(candidates) == 0:
        return candidates

    picked = []
    start = prev = int(candidates[0])
    for frame in candidates[1:]:
        frame = int(frame)
        if frame == prev + 1:
            prev = frame
            continue
        local = onset_curve[start:prev + 1]
        picked.append(start + int(np.argmax(local)))
        start = prev = frame

    local = onset_curve[start:prev + 1]
    picked.append(start + int(np.argmax(local)))
    return np.array(picked, dtype=np.int64)

def frames_to_notes(frame_pred, onset_pred, hop_length, sample_rate,
                    onset_thresh=0.5, frame_thresh=0.5, min_note_len=2,
                    offset_pred=None, offset_thresh=0.5, max_gap=2):

    frame_time = hop_length / sample_rate
    T, P = frame_pred.shape
    notes = []

    for p in range(P):
        midi = p + MIDI_MIN
        onset_frames = pick_onset_frames(onset_pred[:, p], onset_thresh)
        offset_frames = None
        if offset_pred is not None:
            offset_frames = pick_onset_frames(offset_pred[:, p], offset_thresh)

        if len(onset_frames) == 0:

            active = frame_pred[:, p] > frame_thresh
            in_note, note_start = False, 0
            for t in range(T):
                if active[t] and not in_note:
                    in_note, note_start = True, t
                elif (in_note and offset_frames is not None and
                      t in offset_frames and t - note_start >= min_note_len):
                    in_note = False
                    notes.append([note_start * frame_time,
                                  t * frame_time,
                                  float(midi)])
                elif not active[t] and in_note:
                    in_note = False
                    if t - note_start >= min_note_len:
                        notes.append([note_start * frame_time,
                                      t * frame_time,
                                      float(midi)])
            if in_note and T - note_start >= min_note_len:
                notes.append([note_start * frame_time,
                               T * frame_time,
                               float(midi)])
        else:

            for i, f_on in enumerate(onset_frames):
                next_onset = onset_frames[i + 1] if i + 1 < len(onset_frames) else T
                search_end = min(next_onset, T)
                valid_offsets = []
                if offset_frames is not None:
                    valid_offsets = [
                        int(f) for f in offset_frames
                        if f_on + min_note_len <= int(f) < search_end
                    ]

                f_off, gap = f_on, 0
                ended_by_offset = False
                for t in range(f_on, search_end):
                    if valid_offsets and t == valid_offsets[0]:
                        f_off = t
                        ended_by_offset = True
                        break
                    if frame_pred[t, p] > frame_thresh:
                        f_off = t
                        gap = 0
                    else:
                        gap += 1
                        if gap > max_gap and t > f_on + 1:
                            break
                end_frame = f_off if ended_by_offset else f_off + 1
                if end_frame - f_on >= min_note_len:
                    notes.append([f_on * frame_time,
                                  end_frame * frame_time,
                                  float(midi)])

    return notes

def predict_from_npy(model, npy_path, config, device):

    cqt = np.load(npy_path)
    cqt_tensor = torch.from_numpy(cqt).float().unsqueeze(0).to(device)
    segment_frames = config['data']['segment_frames']
    T = cqt.shape[1]

    onset_map = np.zeros((T, 48), dtype=np.float32)
    frame_map = np.zeros((T, 48), dtype=np.float32)
    offset_map = np.zeros((T, 48), dtype=np.float32)
    count_map = np.zeros(T,       dtype=np.float32)
    step = segment_frames // 2

    model.eval()
    with torch.no_grad():
        for start in range(0, T, step):
            end = start + segment_frames
            seg = cqt_tensor[:, :, start:end]
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

    count_map = np.maximum(count_map, 1)
    onset_map /= count_map[:, np.newaxis]
    frame_map  /= count_map[:, np.newaxis]
    offset_map /= count_map[:, np.newaxis]
    return frame_map, onset_map, offset_map

def main():
    parser = argparse.ArgumentParser(
        description='CFT_v6 推理：输出与原论文 evaluate_github.py 兼容的预测 JSON')
    parser.add_argument('--config',       type=str, default='config.yaml')
    parser.add_argument('--checkpoint',   type=str, required=True,
                        help='模型 checkpoint 路径')
    parser.add_argument('--split',        type=str, default='test',
                        help='评估集名称（test / val / train）')
    parser.add_argument('--onset_thresh', type=float, default=0.50,
                        help='onset 阈值（当前 best_model.pt 记录的最优值 0.50）')
    parser.add_argument('--frame_thresh', type=float, default=0.40,
                        help='frame 阈值（当前 best_model.pt 记录的最优值 0.40）')
    parser.add_argument('--output',       type=str, default='pred_test_v10_best.json',
                        help='输出 JSON 路径')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f'Device: {device}')

    model = CFT(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    log.info(f'Checkpoint: epoch={ckpt.get("epoch", "?")}, '
             f'best_val_f1={ckpt.get("best_conp_f1", ckpt.get("best_val_f1", "N/A"))}')

    splits_dir = Path(config['data']['splits_dir'])
    with open(splits_dir / f'{args.split}.txt') as f:
        song_ids = [line.strip() for line in f if line.strip()]

    npy_dir     = Path(config['data']['cqt_cache_dir'])
    hop_length  = config['audio']['hop_length']
    sample_rate = config['data']['sample_rate']

    log.info(f'Split={args.split}, 共 {len(song_ids)} 首')
    log.info(f'onset_thresh={args.onset_thresh}, frame_thresh={args.frame_thresh}')
    log.info('=' * 60)

    predictions = {}
    skipped = 0

    for idx, song_id in enumerate(song_ids):
        npy_path = npy_dir / f'{song_id}.npy'
        if not npy_path.exists():
            log.warning(f'[{idx+1}/{len(song_ids)}] {song_id}: npy 不存在，跳过')
            skipped += 1
            continue

        frame_pred, onset_pred, offset_pred = predict_from_npy(
            model, str(npy_path), config, device)

        notes = frames_to_notes(
            frame_pred, onset_pred, hop_length, sample_rate,
            onset_thresh=args.onset_thresh,
            frame_thresh=args.frame_thresh)

        predictions[song_id] = notes

        log.info(f'[{idx+1:3d}/{len(song_ids)}] song {song_id:>4s}: '
                 f'{len(notes):4d} notes predicted')

    log.info('=' * 60)
    log.info(f'完成: {len(predictions)} 首成功，{skipped} 首跳过')

    with open(args.output, 'w') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    log.info(f'预测结果已保存: {args.output}')

    log.info('')
    log.info('下一步评估命令：')
    log.info(f'  python evaluate_github.py '
             f'{config["data"]["label_path"]} {args.output} 0.05')

if __name__ == '__main__':
    main()
