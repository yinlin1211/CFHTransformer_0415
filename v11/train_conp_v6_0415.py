"""
CFT v6_0415 训练脚本 — train_conp.py
==============================

基于 v6_0415/train_COn.py 改写。

修改说明：
  [修改1] 最佳模型选择标准改为 COnP F1 最大（onset + pitch，不看 offset）
  [修改2] 在 run/<timestamp>_COnP/ 下额外记录 test_monitor.txt
          仅用于 holdout test 监控，不参与阈值搜索或最佳模型选择
  [修改3] 当前默认 config 使用仓库内 label/splits，CQT cache 仍使用外部预计算路径
"""

import argparse
import logging
import sys
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
import json
import yaml

try:
    import mir_eval
    from mir_eval import transcription as mir_transcription, util as mir_util
    HAS_MIR_EVAL = True
except ImportError:
    HAS_MIR_EVAL = False
    print("WARNING: mir_eval not found, F1 metrics will be 0")

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False

from model import CFT_v6, CFTLoss
from dataset import MIR_ST500_Dataset, MIDI_MIN, NUM_PITCHES


def pick_onset_frames(onset_curve, onset_thresh):
    """Collapse a contiguous above-threshold onset region to its strongest frame."""
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


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(run_dir):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(run_dir / 'train_stdout.log', mode='a'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 音符级 F1 评估（v3：对齐原论文 evaluate_github.py）
# ---------------------------------------------------------------------------
# 修正内容（相比 v2）：
#   1. frames_to_notes 的 onset 分支允许 2 帧间隙，避免长音符被抖动截断
#   2. compute_note_f1_single 改用 transcription.evaluate() 一次调用：
#      - pitch 转 Hz（原来直接用 MIDI 编号导致 pitch 距离计算错误）
#      - COn = Onset_F-measure（只看 onset，不看 pitch）
#      - COnP = F-measure_no_offset（onset + pitch）
#      - COnPOff = F-measure（onset + pitch + offset）
#   3. ref 直接从 JSON 标注读取，不再从帧标签反推
# ---------------------------------------------------------------------------

def frames_to_notes(frame_pred, onset_pred, hop_length, sample_rate,
                    onset_thresh=0.5, frame_thresh=0.5, min_note_len=2,
                    offset_pred=None, offset_thresh=0.5, max_gap=2):
    """帧级预测 → 音符列表，返回 (intervals, pitches)，pitches 为 MIDI 编号。

    offset_pred 非空时，优先使用 offset 峰值作为音符终点；否则回退到
    frame 概率连续区间。COnPOff 对终点很敏感，不能只训练 offset head
    而在解码时丢弃它。
    """
    frame_time = hop_length / sample_rate
    T, P = frame_pred.shape
    intervals = []
    pitches = []

    for p in range(P):
        midi = p + MIDI_MIN
        onset_frames = pick_onset_frames(onset_pred[:, p], onset_thresh)
        offset_frames = None
        if offset_pred is not None:
            offset_frames = pick_onset_frames(offset_pred[:, p], offset_thresh)

        if len(onset_frames) == 0:
            # 纯帧模式；如果 offset head 给出结束峰值，则用它修正终点。
            active = frame_pred[:, p] > frame_thresh
            in_note, note_start = False, 0
            for t in range(T):
                if active[t] and not in_note:
                    in_note, note_start = True, t
                elif (in_note and offset_frames is not None and
                      t in offset_frames and t - note_start >= min_note_len):
                    in_note = False
                    intervals.append([note_start * frame_time, t * frame_time])
                    pitches.append(float(midi))
                elif not active[t] and in_note:
                    in_note = False
                    if t - note_start >= min_note_len:
                        intervals.append([note_start * frame_time, t * frame_time])
                        pitches.append(float(midi))
            if in_note and T - note_start >= min_note_len:
                intervals.append([note_start * frame_time, T * frame_time])
                pitches.append(float(midi))
        else:
            # onset 引导模式：offset 峰值优先，frame 连续区间兜底。
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
                for t in range(f_on, min(next_onset, T)):
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
                    intervals.append([f_on * frame_time, end_frame * frame_time])
                    pitches.append(float(midi))

    if len(intervals) == 0:
        return np.zeros((0, 2)), np.zeros(0)
    return np.array(intervals), np.array(pitches, dtype=float)


def compute_note_f1_single(pred_intervals, pred_pitches_midi,
                            ref_intervals, ref_pitches_midi,
                            onset_tolerance=0.05):
    """
    对齐原论文 evaluate_github.py 的评估逻辑。
    输入 pitches 为 MIDI 编号，内部转 Hz 后调用 transcription.evaluate()。
    返回 (COn_f1, COnP_f1, COnPOff_f1)
    """
    if not HAS_MIR_EVAL:
        return 0.0, 0.0, 0.0

    # 过滤 duration <= 0 的音符
    valid_pred = pred_intervals[:, 1] - pred_intervals[:, 0] > 0
    valid_ref  = ref_intervals[:, 1]  - ref_intervals[:, 0]  > 0
    pred_intervals  = pred_intervals[valid_pred]
    pred_pitches_midi = pred_pitches_midi[valid_pred]
    ref_intervals   = ref_intervals[valid_ref]
    ref_pitches_midi  = ref_pitches_midi[valid_ref]

    if len(ref_intervals) == 0:
        return None, None, None
    if len(pred_intervals) == 0:
        return 0.0, 0.0, 0.0

    # pitch 转 Hz（mir_eval 要求 Hz 输入）
    pred_pitches_hz = mir_util.midi_to_hz(pred_pitches_midi)
    ref_pitches_hz  = mir_util.midi_to_hz(ref_pitches_midi)

    try:
        raw = mir_transcription.evaluate(
            ref_intervals, ref_pitches_hz,
            pred_intervals, pred_pitches_hz,
            onset_tolerance=onset_tolerance,
            pitch_tolerance=50,
        )
        con_f1     = raw['Onset_F-measure']
        conp_f1    = raw['F-measure_no_offset']
        conpoff_f1 = raw['F-measure']
    except Exception:
        con_f1 = conp_f1 = conpoff_f1 = 0.0

    return con_f1, conp_f1, conpoff_f1


# ---------------------------------------------------------------------------
# 训练 epoch
# ---------------------------------------------------------------------------

def train_epoch(model, loader, criterion, optimizer, device, epoch, logger,
                grad_clip=1.0, max_batches=None, scaler=None):
    model.train()
    total_loss = 0.0
    onset_loss_sum = 0.0
    frame_loss_sum = 0.0
    offset_loss_sum = 0.0
    n_batches = min(len(loader), max_batches) if max_batches else len(loader)

    for batch_idx, (cqt, labels) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        cqt = cqt.to(device)
        onset_label = labels['onset'].to(device)
        frame_label = labels['frame'].to(device)
        offset_label = labels['offset'].to(device)

        optimizer.zero_grad()
        with autocast():
            onset_pred, frame_pred, offset_pred = model(cqt)
            loss, onset_loss, frame_loss, offset_loss = criterion(
                onset_pred, frame_pred, offset_pred,
                onset_label, frame_label, offset_label
            )

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()
        onset_loss_sum += onset_loss.item()
        frame_loss_sum += frame_loss.item()
        offset_loss_sum += offset_loss.item()

        if (batch_idx + 1) % max(1, n_batches // 3) == 0:
            logger.info(
                f"Epoch {epoch} [{batch_idx+1}/{n_batches}] "
                f"loss={loss.item():.4f} "
                f"onset={onset_loss.item():.4f} "
                f"frame={frame_loss.item():.4f} "
                f"offset={offset_loss.item():.4f}"
            )

    return {
        'total': total_loss / n_batches,
        'onset': onset_loss_sum / n_batches,
        'frame': frame_loss_sum / n_batches,
        'offset': offset_loss_sum / n_batches,
    }


# ---------------------------------------------------------------------------
# 验证（全曲评估）
# ---------------------------------------------------------------------------

def validate_full_song(model, val_dataset, criterion, device, hop_length, sample_rate,
                       onset_thresh=0.5, frame_thresh=0.5, infer_chunk=256,
                       gt_annotations=None, offset_thresh=0.5):
    """
    全曲验证。
    gt_annotations: dict {song_id: [[onset, offset, midi], ...]}，用于 ref 音符。
    如果为 None，则从帧标签反推（不推荐）。
    """
    model.eval()
    total_loss = 0.0
    n_songs = 0
    con_f1_list = []
    conp_f1_list = []
    conpoff_f1_list = []
    onset_sig_list = []
    frame_sig_list = []

    with torch.no_grad():
        for idx in range(len(val_dataset)):
            cqt, labels, song_id = val_dataset[idx]
            F_bins, T_total = cqt.shape

            onset_lbl = labels['onset'].numpy()
            frame_lbl = labels['frame'].numpy()
            offset_lbl = labels['offset'].numpy()

            onset_sig_chunks = []
            frame_sig_chunks = []
            offset_sig_chunks = []
            chunk_losses = []

            for start in range(0, T_total, infer_chunk):
                end = min(start + infer_chunk, T_total)
                cqt_chunk = cqt[:, start:end].unsqueeze(0).to(device)

                chunk_T = end - start
                if chunk_T < infer_chunk:
                    pad_len = infer_chunk - chunk_T
                    cqt_chunk = torch.nn.functional.pad(cqt_chunk, (0, pad_len))

                onset_pred, frame_pred, offset_pred = model(cqt_chunk)

                onset_pred = onset_pred[:, :chunk_T, :]
                frame_pred = frame_pred[:, :chunk_T, :]
                offset_pred = offset_pred[:, :chunk_T, :]

                ol_chunk = torch.from_numpy(onset_lbl[start:end]).unsqueeze(0).to(device)
                fl_chunk = torch.from_numpy(frame_lbl[start:end]).unsqueeze(0).to(device)
                ofl_chunk = torch.from_numpy(offset_lbl[start:end]).unsqueeze(0).to(device)
                loss, _, _, _ = criterion(onset_pred, frame_pred, offset_pred,
                                          ol_chunk, fl_chunk, ofl_chunk)
                chunk_losses.append(loss.item())

                onset_sig_chunks.append(torch.sigmoid(onset_pred[0]).cpu().numpy())
                frame_sig_chunks.append(torch.sigmoid(frame_pred[0]).cpu().numpy())
                offset_sig_chunks.append(torch.sigmoid(offset_pred[0]).cpu().numpy())

            onset_sig = np.concatenate(onset_sig_chunks, axis=0)
            frame_sig = np.concatenate(frame_sig_chunks, axis=0)
            offset_sig = np.concatenate(offset_sig_chunks, axis=0)

            total_loss += float(np.mean(chunk_losses))
            n_songs += 1

            onset_sig_list.append(onset_sig.mean())
            frame_sig_list.append(frame_sig.mean())

            # 预测音符
            pred_intervals, pred_pitches = frames_to_notes(
                frame_sig, onset_sig, hop_length, sample_rate,
                onset_thresh, frame_thresh,
                offset_pred=offset_sig, offset_thresh=offset_thresh
            )

            # ref 音符：优先从 JSON 标注读取，否则从帧标签反推
            if gt_annotations is not None and song_id in gt_annotations:
                raw = gt_annotations[song_id]
                ref_notes = [[float(n[0]), float(n[1]), float(n[2])] for n in raw
                             if float(n[1]) - float(n[0]) > 0]
                if len(ref_notes) == 0:
                    continue
                ref_intervals = np.array([[n[0], n[1]] for n in ref_notes])
                ref_pitches   = np.array([n[2] for n in ref_notes])
            else:
                # 备用：从帧标签反推（不推荐，仅当没有 JSON 时使用）
                ref_intervals, ref_pitches = frames_to_notes(
                    frame_lbl.astype(np.float32), onset_lbl.astype(np.float32),
                    hop_length, sample_rate, onset_thresh=0.5, frame_thresh=0.5,
                    offset_pred=offset_lbl.astype(np.float32), offset_thresh=0.5
                )

            if len(ref_intervals) == 0:
                continue

            con_f1, conp_f1, conpoff_f1 = compute_note_f1_single(
                pred_intervals, pred_pitches,
                ref_intervals, ref_pitches
            )
            if con_f1 is not None:
                con_f1_list.append(con_f1)
                conp_f1_list.append(conp_f1)
                conpoff_f1_list.append(conpoff_f1)

    avg_loss = total_loss / max(n_songs, 1)
    avg_con_f1 = float(np.mean(con_f1_list)) if con_f1_list else 0.0
    avg_conp_f1 = float(np.mean(conp_f1_list)) if conp_f1_list else 0.0
    avg_conpoff_f1 = float(np.mean(conpoff_f1_list)) if conpoff_f1_list else 0.0
    avg_onset_sig = float(np.mean(onset_sig_list)) if onset_sig_list else 0.0
    avg_frame_sig = float(np.mean(frame_sig_list)) if frame_sig_list else 0.0

    return avg_loss, avg_con_f1, avg_conp_f1, avg_conpoff_f1, avg_onset_sig, avg_frame_sig


def find_best_threshold(model, val_dataset, criterion, device, hop_length, sample_rate,
                        logger, gt_annotations=None, metric="COnPOff"):
    n_search = min(30, len(val_dataset))
    metric = metric.upper()
    if metric not in {"CON", "CONP", "CONPOFF"}:
        metric = "CONPOFF"
    best_score = 0.0
    best_ot, best_ft, best_oft = 0.3, 0.3, 0.3

    model.eval()
    infer_chunk = 256
    preds = []  # (frame_sig, onset_sig, offset_sig, ref_intervals, ref_pitches)
    with torch.no_grad():
        for idx in range(n_search):
            cqt, labels, song_id = val_dataset[idx]
            T_total = cqt.shape[1]
            onset_chunks, frame_chunks, offset_chunks = [], [], []
            for start in range(0, T_total, infer_chunk):
                end = min(start + infer_chunk, T_total)
                chunk_T = end - start
                cqt_chunk = cqt[:, start:end].unsqueeze(0).to(device)
                if chunk_T < infer_chunk:
                    cqt_chunk = torch.nn.functional.pad(cqt_chunk, (0, infer_chunk - chunk_T))
                op, fp, ofp = model(cqt_chunk)
                onset_chunks.append(torch.sigmoid(op[0, :chunk_T]).cpu().numpy())
                frame_chunks.append(torch.sigmoid(fp[0, :chunk_T]).cpu().numpy())
                offset_chunks.append(torch.sigmoid(ofp[0, :chunk_T]).cpu().numpy())
            onset_sig = np.concatenate(onset_chunks, axis=0)
            frame_sig = np.concatenate(frame_chunks, axis=0)
            offset_sig = np.concatenate(offset_chunks, axis=0)

            # ref 音符：优先从 JSON 标注读取
            if gt_annotations is not None and song_id in gt_annotations:
                raw = gt_annotations[song_id]
                ref_notes = [[float(n[0]), float(n[1]), float(n[2])] for n in raw
                             if float(n[1]) - float(n[0]) > 0]
                if len(ref_notes) == 0:
                    continue
                ref_intervals = np.array([[n[0], n[1]] for n in ref_notes])
                ref_pitches   = np.array([n[2] for n in ref_notes])
            else:
                onset_lbl = labels['onset'].numpy()
                frame_lbl = labels['frame'].numpy()
                ref_intervals, ref_pitches = frames_to_notes(
                    frame_lbl.astype(np.float32), onset_lbl.astype(np.float32),
                    hop_length, sample_rate, onset_thresh=0.5, frame_thresh=0.5,
                    offset_pred=labels['offset'].numpy().astype(np.float32),
                    offset_thresh=0.5
                )
            preds.append((frame_sig, onset_sig, offset_sig, ref_intervals, ref_pitches))

    onset_thresholds = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    frame_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    offset_thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

    for ot in onset_thresholds:
        for ft in frame_thresholds:
            for oft in offset_thresholds:
                score_list = []
                for frame_sig, onset_sig, offset_sig, ref_intervals, ref_pitches in preds:
                    pred_intervals, pred_pitches = frames_to_notes(
                        frame_sig, onset_sig, hop_length, sample_rate, ot, ft,
                        offset_pred=offset_sig, offset_thresh=oft
                    )
                    if len(ref_intervals) == 0:
                        continue
                    con, conp, conpoff = compute_note_f1_single(
                        pred_intervals, pred_pitches, ref_intervals, ref_pitches
                    )
                    scores = {"CON": con, "CONP": conp, "CONPOFF": conpoff}
                    score = scores[metric]
                    if score is not None:
                        score_list.append(score)
                if score_list:
                    avg_score = np.mean(score_list)
                    if avg_score > best_score:
                        best_score = avg_score
                        best_ot, best_ft, best_oft = ot, ft, oft

    logger.info(
        f"  Threshold search ({metric}): best on={best_ot:.2f}, "
        f"fr={best_ft:.2f}, off={best_oft:.2f}, score={best_score:.4f}"
    )
    return best_ot, best_ft, best_oft


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 按启动时间生成独立 run 目录，格式：run/<时间戳>_COn/
    # 每次启动都会创建全新子目录，多个进程并发训练时互不干扰
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{run_timestamp}_COnP"
    run_base = Path(config['training'].get('run_dir', './run'))
    run_dir = run_base / run_name
    save_dir = run_dir / 'checkpoints'
    log_dir  = run_dir / 'logs'
    run_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    pid_file = Path(f'/tmp/cft_v6_COnP_{run_timestamp}.pid')
    pid_file.write_text(str(os.getpid()))

    logger = setup_logger(log_dir)
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Config: {config}")
    logger.info("=" * 60)
    logger.info("CFT v6_0415 — train_conp.py (best model: COnP F1)")
    logger.info("=" * 60)

    # 加载 JSON 标注（用于验证时 ref 音符）
    gt_json_path = config['data']['label_path']
    with open(gt_json_path) as f:
        gt_annotations = json.load(f)
    logger.info(f"Loaded GT annotations from {gt_json_path}")

    # 数据集
    train_dataset = MIR_ST500_Dataset(config, split='train')
    val_dataset = MIR_ST500_Dataset(config, split='val')
    test_dataset = MIR_ST500_Dataset(config, split='test')
    # Test 仅用于监控，不参与阈值搜索或 best model 选择。
    # 这里直接使用全量 test。
    test_monitor_dataset = test_dataset
    logger.info(
        f"Train samples: {len(train_dataset)}, Val songs: {len(val_dataset)}, "
        f"Test songs: {len(test_dataset)}, Test monitor songs: {len(test_monitor_dataset)}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    # 模型
    model = CFT_v6(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")
    logger.info("Tokenization: continuous kernels [3,5,7] (paper-aligned, no dilation)")

    # 混合精度
    scaler = GradScaler() if device.type == 'cuda' else None
    if scaler is not None:
        logger.info("Mixed precision (AMP) enabled")

    # 损失函数（论文公式1 + 正样本加权修复类别不平衡）
    criterion = CFTLoss(
        onset_weight=config['loss']['onset_weight'],
        frame_weight=config['loss']['frame_weight'],
        offset_weight=config['loss']['offset_weight'],
        onset_pos_weight=config['loss'].get('onset_pos_weight', 1.0),
        frame_pos_weight=config['loss'].get('frame_pos_weight', 1.0),
        offset_pos_weight=config['loss'].get('offset_pos_weight', 1.0),
    ).to(device)

    # 优化器（论文 Section 3.3：Adam, lr=3e-4）
    optimizer = Adam(
        model.parameters(),
        lr=config['training']['learning_rate']
    )

    # 学习率调度：论文未提及，使用 CosineAnnealingLR（无 warmup）
    total_epochs = config['training']['epochs']
    scheduler = CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=1e-6
    )
    logger.info(f"Scheduler: CosineAnnealingLR (no warmup, T_max={total_epochs})")

    writer = None
    if HAS_TB:
        writer = SummaryWriter(str(log_dir / 'tensorboard'))

    start_epoch = 1
    best_conp_f1 = 0.0
    best_onset_thresh = 0.3
    best_frame_thresh = 0.3
    best_offset_thresh = 0.3
    threshold_metric = config['training'].get('threshold_metric', 'COnPOff')
    logger.info(f"Threshold search metric: {threshold_metric}")

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if scaler is not None and 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_conp_f1 = ckpt.get('best_conp_f1', ckpt.get('best_con_f1', 0.0))
        best_onset_thresh = ckpt.get('best_onset_thresh', 0.3)
        best_frame_thresh = ckpt.get('best_frame_thresh', 0.3)
        best_offset_thresh = ckpt.get('best_offset_thresh', 0.3)
        logger.info(f"Resumed from epoch {ckpt['epoch']}, best_COnP_f1={best_conp_f1:.4f}")

    hop_length = config['audio']['hop_length']
    sample_rate = config['data']['sample_rate']

    max_samples = config['data'].get('max_samples_per_epoch', None)
    max_batches = None
    if max_samples:
        max_batches = max(1, max_samples // config['training']['batch_size'])
        logger.info(f"Max batches per epoch: {max_batches}")

    # full test 监控：仅诊断，不参与阈值搜索或 best model 选择
    test_monitor_path = run_dir / 'test_monitor.txt'
    if not test_monitor_path.exists():
        with open(test_monitor_path, 'w') as tf:
            tf.write("epoch\ttrigger\ttest_loss\tCOn_f1\tCOnP_f1\tCOnPOff_f1\tonset_thresh\tframe_thresh\toffset_thresh\n")

    for epoch in range(start_epoch, total_epochs + 1):
        train_losses = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, logger,
            grad_clip=config['training']['grad_clip'],
            max_batches=max_batches,
            scaler=scaler
        )
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        if epoch % 5 == 0 or epoch == 1:
            best_onset_thresh, best_frame_thresh, best_offset_thresh = find_best_threshold(
                model, val_dataset, criterion, device, hop_length, sample_rate,
                logger, gt_annotations=gt_annotations, metric=threshold_metric
            )

        val_loss, con_f1, conp_f1, conpoff_f1, onset_sig, frame_sig = validate_full_song(
            model, val_dataset, criterion, device, hop_length, sample_rate,
            onset_thresh=best_onset_thresh, frame_thresh=best_frame_thresh,
            gt_annotations=gt_annotations, offset_thresh=best_offset_thresh
        )

        logger.info(
            f"Epoch {epoch}/{total_epochs} | "
            f"train_loss={train_losses['total']:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"COn_f1={con_f1:.4f} | "
            f"COnP_f1={conp_f1:.4f} | "
            f"COnPOff_f1={conpoff_f1:.4f} | "
            f"sig_onset={onset_sig:.4f} sig_frame={frame_sig:.4f} | "
            f"thresh(on={best_onset_thresh:.2f},fr={best_frame_thresh:.2f},off={best_offset_thresh:.2f}) | "
            f"lr={lr:.2e}"
        )

        if writer:
            writer.add_scalar('Loss/train', train_losses['total'], epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Metrics/COn_f1', con_f1, epoch)
            writer.add_scalar('Metrics/COnP_f1', conp_f1, epoch)
            writer.add_scalar('Metrics/COnPOff_f1', conpoff_f1, epoch)
            writer.add_scalar('LR', lr, epoch)

        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
            'val_loss': val_loss,
            'COn_f1': con_f1,
            'COnP_f1': conp_f1,
            'COnPOff_f1': conpoff_f1,
            'best_conp_f1': best_conp_f1,
            'best_onset_thresh': best_onset_thresh,
            'best_frame_thresh': best_frame_thresh,
            'best_offset_thresh': best_offset_thresh,
            'threshold_metric': threshold_metric,
            'config': config
        }

        is_best_epoch = conp_f1 > best_conp_f1
        if is_best_epoch:
            best_conp_f1 = conp_f1
            ckpt['best_conp_f1'] = best_conp_f1
            torch.save(ckpt, save_dir / 'best_model.pt')
            torch.save(ckpt, save_dir / f'best_model_epoch{epoch:04d}_COnP{best_conp_f1:.4f}.pt')
            logger.info(f"  -> Best model saved! COnP_f1={best_conp_f1:.4f}")

        if epoch % config['training']['save_every'] == 0:
            torch.save(ckpt, save_dir / f'checkpoint_epoch{epoch:04d}.pt')

        torch.save(ckpt, save_dir / 'latest.pt')
        # full test 监控：每 40 个 epoch 固定评估一次；
        # 另外在 epoch > 20 后，只要出现新的 best model，也额外评估一次。
        test_monitor_triggers = []
        if epoch % 40 == 0 or epoch == 1:
            test_monitor_triggers.append('periodic_40')
        if epoch > 20 and is_best_epoch:
            test_monitor_triggers.append('best_after20')

        if test_monitor_triggers:
            test_mon_loss, test_con, test_conp, test_conpoff, _, _ = validate_full_song(
                model, test_monitor_dataset, criterion, device, hop_length, sample_rate,
                onset_thresh=best_onset_thresh, frame_thresh=best_frame_thresh,
                gt_annotations=gt_annotations, offset_thresh=best_offset_thresh
            )
            with open(test_monitor_path, 'a') as tf:
                tf.write(
                    f"{epoch}\t"
                    f"{'+'.join(test_monitor_triggers)}\t"
                    f"{test_mon_loss:.6f}\t"
                    f"{test_con:.6f}\t"
                    f"{test_conp:.6f}\t"
                    f"{test_conpoff:.6f}\t"
                    f"{best_onset_thresh:.2f}\t"
                    f"{best_frame_thresh:.2f}\t"
                    f"{best_offset_thresh:.2f}\n"
                )
            logger.info(
                f"  [TEST MONITOR FULL TEST:{'+'.join(test_monitor_triggers)}] "
                f"COn_f1={test_con:.4f} | "
                f"COnP_f1={test_conp:.4f} | "
                f"COnPOff_f1={test_conpoff:.4f}"
            )

    if writer:
        writer.close()

    logger.info(f"Training complete! Best COnP_f1: {best_conp_f1:.4f}")
    pid_file.unlink(missing_ok=True)
    logger.info(f"Run directory: {run_dir}")


if __name__ == '__main__':
    main()
