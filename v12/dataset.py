import torch
from torch.utils.data import Dataset
import numpy as np
import json
from pathlib import Path
import random

MIDI_MIN = 36
MIDI_MAX = 83
NUM_PITCHES = MIDI_MAX - MIDI_MIN + 1

class MIR_ST500_Dataset(Dataset):
    def __init__(self, config, split="train", max_songs=None):
        self.config = config
        self.split = split
        self.cqt_cache_dir = Path(config["data"]["cqt_cache_dir"])
        self.label_path = Path(config["data"]["label_path"])
        self.splits_dir = Path(config["data"]["splits_dir"])
        self.segment_frames = config["data"]["segment_frames"]
        self.hop_length = config["audio"]["hop_length"]
        self.sample_rate = config["data"]["sample_rate"]

        with open(self.label_path, "r") as f:
            self.annotations = json.load(f)

        split_file = self.splits_dir / f"{split}.txt"
        with open(split_file, "r") as f:
            file_list = [line.strip() for line in f if line.strip()]

        if max_songs is not None:
            file_list = file_list[:max_songs]

        valid = []
        for sid in file_list:
            if (self.cqt_cache_dir / f"{sid}.npy").exists():
                valid.append(sid)
        if len(valid) < len(file_list):
            print(f"Warning: {len(file_list) - len(valid)} songs missing CQT cache in {split} split")
        self.file_list = valid

        if split == "train":
            self._build_train_index()

    def _build_train_index(self):

        self._train_index = []
        stride = self.segment_frames // 2
        oversample = self.config.get("data", {}).get("extreme_pitch_oversample", 0)
        LOW_THRESH = 50
        HIGH_THRESH = 75

        for sid in self.file_list:
            cqt_path = self.cqt_cache_dir / f"{sid}.npy"
            cqt = np.load(str(cqt_path))
            num_frames = cqt.shape[1]

            notes = self.annotations.get(sid, [])
            frame_time = self.hop_length / self.sample_rate
            note_frames = set()
            extreme_pitch_frames = set()
            for note in notes:
                midi = int(note[2])
                if not (MIDI_MIN <= midi <= MIDI_MAX):
                    continue
                f_on = int(round(float(note[0]) / frame_time))
                f_off = int(round(float(note[1]) / frame_time))
                for f in range(f_on, min(f_off + 1, num_frames)):
                    note_frames.add(f)
                    if midi < LOW_THRESH or midi > HIGH_THRESH:
                        extreme_pitch_frames.add(f)

            if num_frames < self.segment_frames:
                self._train_index.append((sid, 0))
                continue

            for start in range(0, num_frames - self.segment_frames + 1, stride):
                end = start + self.segment_frames
                segment_has_note = any(start <= f < end for f in note_frames)
                if segment_has_note:
                    self._train_index.append((sid, start))
                    if oversample > 0:
                        has_extreme = any(start <= f < end for f in extreme_pitch_frames)
                        if has_extreme:
                            for _ in range(oversample):
                                self._train_index.append((sid, start))
                else:

                    if random.random() < 0.15:
                        self._train_index.append((sid, start))

            if not any(s == sid for s, _ in self._train_index[-20:]):
                self._train_index.append((sid, 0))

    def __len__(self):
        if self.split == "train":
            return len(self._train_index)
        else:
            return len(self.file_list)

    def __getitem__(self, idx):
        if self.split == "train":
            return self._get_train_item(idx)
        else:
            return self._get_full_song(idx)

    def _get_train_item(self, idx):

        song_id, start = self._train_index[idx]

        jitter = random.randint(-self.segment_frames // 8, self.segment_frames // 8)
        cqt = np.load(str(self.cqt_cache_dir / f"{song_id}.npy"))
        num_frames = cqt.shape[1]

        start = max(0, min(start + jitter, num_frames - self.segment_frames))
        end = start + self.segment_frames

        cqt_seg = cqt[:, start:end]
        labels = self._create_labels(song_id, num_frames)
        labels_seg = {k: v[start:end] for k, v in labels.items()}

        cqt_tensor = torch.from_numpy(cqt_seg).float()
        label_tensors = {k: torch.from_numpy(v).float() for k, v in labels_seg.items()}
        return cqt_tensor, label_tensors

    def _get_full_song(self, idx):

        song_id = self.file_list[idx]
        cqt = np.load(str(self.cqt_cache_dir / f"{song_id}.npy"))
        num_frames = cqt.shape[1]

        labels = self._create_labels(song_id, num_frames)

        cqt_tensor = torch.from_numpy(cqt).float()
        label_tensors = {k: torch.from_numpy(v).float() for k, v in labels.items()}
        return cqt_tensor, label_tensors, song_id

    def _create_labels(self, song_id, num_frames):

        notes = self.annotations.get(song_id, [])
        frame_time = self.hop_length / self.sample_rate

        onset  = np.zeros((num_frames, NUM_PITCHES), dtype=np.float32)
        offset = np.zeros((num_frames, NUM_PITCHES), dtype=np.float32)
        frame  = np.zeros((num_frames, NUM_PITCHES), dtype=np.float32)

        for note in notes:
            t_on, t_off, midi = float(note[0]), float(note[1]), int(note[2])
            pitch_idx = midi - MIDI_MIN
            if not (0 <= pitch_idx < NUM_PITCHES):
                continue
            f_on = int(round(t_on / frame_time))
            f_off = int(round(t_off / frame_time))

            if f_on < num_frames:
                onset[f_on, pitch_idx] = 1.0
            if f_off < num_frames:
                offset[f_off, pitch_idx] = 1.0
            for f in range(f_on, min(f_off + 1, num_frames)):
                frame[f, pitch_idx] = 1.0

        return {"onset": onset, "offset": offset, "frame": frame}
