import torch
import torch.nn as nn
import torch.nn.functional as F_func
from typing import List


class PaperHarmConvBlock(nn.Module):
    def __init__(self, n_in_channels: int, n_out_channels: int,
                 octave_depth: int = 4,
                 pitch_class_kernels: List[int] = None,
                 time_width: int = 3):
        super().__init__()
        if pitch_class_kernels is None:
            pitch_class_kernels = [3, 5, 7]

        self.octave_depth = octave_depth
        self.pitch_class_kernels = pitch_class_kernels
        self.time_width = time_width

        self.branches = nn.ModuleList()
        for k_h in pitch_class_kernels:
            conv = nn.Conv3d(
                n_in_channels, n_out_channels,
                kernel_size=(octave_depth, k_h, time_width),
                padding=0,
                dilation=1
            )
            self.branches.append(conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, O, P, T = x.shape
        outputs = None

        for conv, k_h in zip(self.branches, self.pitch_class_kernels):
            pad_o = self.octave_depth - 1   # octave 末尾补零
            pad_p = k_h // 2               # pitch_class 循环 padding
            pad_t = self.time_width - 1    # time 因果 padding

            # pitch_class 循环 padding
            if pad_p > 0:
                left_p  = x[:, :, :, -pad_p:, :]
                right_p = x[:, :, :, :pad_p, :]
                x_p = torch.cat([left_p, x, right_p], dim=3)
            else:
                x_p = x

            # octave 末尾补零
            zero_o = torch.zeros(B, C, pad_o, x_p.shape[3], T,
                                 device=x.device, dtype=x.dtype)
            x_op = torch.cat([x_p, zero_o], dim=2)

            # time 因果 padding（左侧补零）
            x_opt = F_func.pad(x_op, (pad_t, 0))

            y = conv(x_opt)
            outputs = y if outputs is None else outputs + y

        return F_func.relu(outputs)


class From2Dto3D(nn.Module):
    def __init__(self, bins_per_octave: int, n_octaves: int):
        super().__init__()
        self.bins_per_octave = bins_per_octave
        self.n_octaves = n_octaves
        self.total_bins = n_octaves * bins_per_octave

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, bins, T = x.shape
        if bins < self.total_bins:
            x = F_func.pad(x, (0, 0, 0, self.total_bins - bins))
        return x.reshape(B, C, self.n_octaves, self.bins_per_octave, T)


class HarmonicTokenizer(nn.Module):
    def __init__(self, n_octaves: int = 6, bins_per_octave: int = 48,
                 h_dim: int = 128, octave_depth: int = 4,
                 pitch_class_kernels: List[int] = None,
                 conv_channels: int = 32, time_width: int = 3,
                 num_pitches: int = 48, midi_min: int = 36,
                 fmin_hz: float = 48.9994):
        super().__init__()
        if pitch_class_kernels is None:
            pitch_class_kernels = [3, 5, 7]

        self.n_octaves = n_octaves
        self.bins_per_octave = bins_per_octave
        self.h_dim = h_dim
        self.conv_channels = conv_channels
        self.num_pitches = num_pitches
        self.midi_min = midi_min
        self.fmin_hz = fmin_hz

        self.to_3d = From2Dto3D(bins_per_octave, n_octaves)
        self.harm_conv = PaperHarmConvBlock(
            n_in_channels=1,
            n_out_channels=conv_channels,
            octave_depth=octave_depth,
            pitch_class_kernels=pitch_class_kernels,
            time_width=time_width,
        )
        self.proj = nn.Linear(conv_channels, h_dim)

        midi = torch.arange(midi_min, midi_min + num_pitches, dtype=torch.float32)
        midi_hz = 440.0 * torch.pow(torch.tensor(2.0), (midi - 69.0) / 12.0)
        cqt_bins = torch.round(torch.log2(midi_hz / fmin_hz) * bins_per_octave).long()
        max_bin = n_octaves * bins_per_octave - 1
        cqt_bins = cqt_bins.clamp(0, max_bin)
        self.register_buffer("pitch_octave_idx", cqt_bins // bins_per_octave, persistent=False)
        self.register_buffer("pitch_bin_idx", cqt_bins % bins_per_octave, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, F, T = x.shape
        x = x.unsqueeze(1)                       # (B, 1, 288, T)
        x = self.to_3d(x)                        # (B, 1, 6, 48, T)
        x = self.harm_conv(x)                    # (B, conv_ch, 6, 48, T)
        x = x[:, :, self.pitch_octave_idx, self.pitch_bin_idx, :]
        x = x.permute(0, 3, 2, 1)               # (B, T, 48, conv_ch)
        x = self.proj(x)                         # (B, T, 48, H)
        return x


# ═════════════════════════════════════════════════════════════════════════════
# 序列内部位置编码（用于 Transformer 内部序列顺序编码）
# ═════════════════════════════════════════════════════════════════════════════

class LearnablePE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.max_len = max_len
        self.pe = nn.Parameter(torch.randn(max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        if seq_len <= self.max_len:
            return x + self.pe[:seq_len]
        else:
            pe_expanded = F_func.interpolate(
                self.pe.unsqueeze(0).transpose(1, 2),
                size=seq_len, mode='linear', align_corners=False
            ).transpose(1, 2).squeeze(0)
            return x + pe_expanded


class FHTransformer(nn.Module):
    def __init__(self, H: int, nhead: int, dim_ff: int,
                 dropout: float, num_layers: int = 1, max_T: int = 4096):
        super().__init__()
        self.temporal_embed = nn.Embedding(max_T, H)
        self.freq_pe = LearnablePE(H, max_len=64)

        layer = nn.TransformerEncoderLayer(
            d_model=H, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        B, T, F, H = S.shape

        if T <= self.temporal_embed.num_embeddings:
            t_idx = torch.arange(T, device=S.device)
            t_emb = self.temporal_embed(t_idx)          # (T, H)
        else:

            t_emb_all = self.temporal_embed.weight       # (max_T, H)
            t_emb = F_func.interpolate(
                t_emb_all.unsqueeze(0).transpose(1, 2),  # (1, H, max_T)
                size=T, mode='linear', align_corners=False
            ).squeeze(0).transpose(0, 1)                 # (T, H)

        # 广播：(T, H) → (1, T, 1, H) → 加到 S (B, T, F, H)
        S = S + t_emb.unsqueeze(0).unsqueeze(2)

        # 步骤2：T 个时间步并行，每步序列长度=F
        x = S.reshape(B * T, F, H)
        x = self.freq_pe(x)                             # 序列内部位置编码
        x = self.encoder(x)
        return x.reshape(B, T, F, H)


class HTTransformer(nn.Module):
    def __init__(self, F_dim: int, H: int, nhead: int, dim_ff: int,
                 dropout: float, num_layers: int = 1):
        super().__init__()
        # 全局频率标签：每个频率 bin f 有独立的 H 维嵌入（对应论文 H(f)）
        self.freq_embed = nn.Embedding(F_dim, H)
        # 序列内部时间位置 PE（序列长度=T，训练时256，推理时可达4096）
        self.time_pe = LearnablePE(H, max_len=4096)

        layer = nn.TransformerEncoderLayer(
            d_model=H, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        """S: (B, T, F, H) → (B, T, F, H)"""
        B, T, F, H = S.shape

        # 步骤1：加 frequency-wise PE H(f)（论文公式3）
        f_idx = torch.arange(F, device=S.device)
        f_emb = self.freq_embed(f_idx)                  # (F, H)
        # 广播：(F, H) → (1, 1, F, H) → 加到 S (B, T, F, H)
        S = S + f_emb.unsqueeze(0).unsqueeze(0)

        # 步骤2：F 个频率 bin 并行，每个 bin 序列长度=T
        x = S.permute(0, 2, 1, 3).reshape(B * F, T, H)  # (B*F, T, H)
        x = self.time_pe(x)                              # 序列内部位置编码
        x = self.encoder(x)
        return x.reshape(B, F, T, H).permute(0, 2, 1, 3)  # (B, T, F, H)


class TFTransformer(nn.Module):
    def __init__(self, F_dim: int, H: int, nhead: int, dim_ff: int,
                 dropout: float, num_layers: int = 1):
        super().__init__()
        # 全局谐波标签：每个谐波通道 h 有独立的 F 维嵌入（对应论文 T(h)）
        self.harm_embed = nn.Embedding(H, F_dim)
        # 序列内部时间位置 PE（序列长度=T，d_model=F=48）
        self.time_pe = LearnablePE(F_dim, max_len=4096)

        layer = nn.TransformerEncoderLayer(
            d_model=F_dim, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        """S: (B, T, F, H) → (B, T, F, H)"""
        B, T, F, H = S.shape

        # 步骤1：加 harmonic-wise PE T(h)（论文公式4）
        h_idx = torch.arange(H, device=S.device)
        h_emb = self.harm_embed(h_idx)                   # (H, F)
        # 广播：(H, F).T = (F, H) → (1, 1, F, H) → 加到 S (B, T, F, H)
        # 验证：S[b,t,f,h] += h_emb[h,f]，即第h个谐波通道在第f个频率位置的嵌入
        S = S + h_emb.T.unsqueeze(0).unsqueeze(0)

        # 步骤2：H 个谐波通道并行，每个通道序列长度=T，d_model=F
        x = S.permute(0, 3, 1, 2).reshape(B * H, T, F)  # (B*H, T, F)
        x = self.time_pe(x)                              # 序列内部位置编码
        x = self.encoder(x)
        return x.reshape(B, H, T, F).permute(0, 2, 3, 1)  # (B, T, F, H)


class CQTNormalize(nn.Module):
    def __init__(self, cqt_mean: float = -65.0, cqt_std: float = 18.0):
        super().__init__()
        self.mean = cqt_mean
        self.std = cqt_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class CFT_v6(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg['model']
        a = cfg.get('audio', {})

        self.n_octaves    = a.get('n_octaves', 6)
        self.bins_per_oct = a.get('bins_per_octave', 48)
        self.H            = m.get('h_dim', 128)
        self.conv_ch      = m.get('conv_channels', 32)
        self.num_cycles   = m.get('num_cycles', 2)
        self.num_layers   = m.get('num_transformer_layers', 1)
        self.nhead_fh     = m.get('nhead_fh', 8)
        self.nhead_ht     = m.get('nhead_ht', 8)
        self.nhead_tf     = m.get('nhead_tf', 6)
        self.dim_ff       = m.get('dim_feedforward', 512)
        self.dropout      = m.get('dropout', 0.1)
        self.num_pitches  = m.get('num_pitches', 48)
        self.midi_min     = m.get('midi_min', 36)
        self.fmin_hz      = a.get('fmin', 48.9994)

        # 参数合法性检查
        assert self.H % self.nhead_fh == 0, \
            f"H={self.H} 必须能被 nhead_fh={self.nhead_fh} 整除"
        assert self.H % self.nhead_ht == 0, \
            f"H={self.H} 必须能被 nhead_ht={self.nhead_ht} 整除"
        assert self.num_pitches % self.nhead_tf == 0, \
            f"num_pitches={self.num_pitches} 必须能被 nhead_tf={self.nhead_tf} 整除"

        # CQT normalization: zero-mean, unit-variance
        cqt_mean = m.get('cqt_mean', -65.0)
        cqt_std = m.get('cqt_std', 18.0)
        self.cqt_norm = CQTNormalize(cqt_mean, cqt_std)

        # Tokenization（论文对齐：连续大核 3/5/7，dilation=1）
        self.tokenizer = HarmonicTokenizer(
            n_octaves=self.n_octaves,
            bins_per_octave=self.bins_per_oct,
            h_dim=self.H,
            octave_depth=4,
            pitch_class_kernels=[3, 5, 7],
            conv_channels=self.conv_ch,
            time_width=3,
            num_pitches=self.num_pitches,
            midi_min=self.midi_min,
            fmin_hz=self.fmin_hz,
        )

        self.F_token = self.num_pitches  # 48 MIDI pitches, C2~B5

        # CFT 循环（M 次，每次包含三个 Transformer）
        self.fh_transformers = nn.ModuleList([
            FHTransformer(self.H, self.nhead_fh, self.dim_ff,
                          self.dropout, self.num_layers)
            for _ in range(self.num_cycles)
        ])
        self.ht_transformers = nn.ModuleList([
            HTTransformer(self.F_token, self.H, self.nhead_ht, self.dim_ff,
                          self.dropout, self.num_layers)
            for _ in range(self.num_cycles)
        ])
        self.tf_transformers = nn.ModuleList([
            TFTransformer(self.F_token, self.H, self.nhead_tf, self.dim_ff,
                          self.dropout, self.num_layers)
            for _ in range(self.num_cycles)
        ])

        # 输出头（论文 Fig.2：GAP 沿 H 轴 + Linear）
        # GAP 后维度为 F=48，Linear(48, 48) 学习 pitch class → pitch range 映射
        self.onset_head  = nn.Linear(self.F_token, self.num_pitches)
        self.frame_head  = nn.Linear(self.F_token, self.num_pitches)
        self.offset_head = nn.Linear(self.F_token, self.num_pitches)

    def forward(self, x: torch.Tensor):
        """Return onset, frame, offset logits, each shaped (B, T, num_pitches=48)."""
        # 0. Normalize CQT input
        x = self.cqt_norm(x)

        # 1. Tokenization → S: (B, T, 48, H)
        S = self.tokenizer(x)

        # 2. CFT 循环：FH → HT → TF（循环 M 次）
        for m_idx in range(self.num_cycles):
            S = self.fh_transformers[m_idx](S)   # 建立时间-频率依赖
            S = self.ht_transformers[m_idx](S)   # 建立谐波-时间依赖（最关键）
            S = self.tf_transformers[m_idx](S)   # 建立时间-频率依赖

        # 3. GAP 沿 H 轴（论文 Section 2.1）
        out = S.mean(dim=-1)    # (B, T, 48)

        # 4. 输出头
        onset  = self.onset_head(out)   # (B, T, 48)
        frame  = self.frame_head(out)   # (B, T, 48)
        offset = self.offset_head(out)  # (B, T, 48)

        return onset, frame, offset



class CFTLoss(nn.Module):
    def __init__(self, onset_weight: float = 1.0,
                 frame_weight: float = 1.0,
                 offset_weight: float = 1.0,
                 onset_pos_weight: float = 5.0,
                 frame_pos_weight: float = 1.0,
                 offset_pos_weight: float = 1.0):
        super().__init__()
        self.onset_weight  = onset_weight
        self.frame_weight  = frame_weight
        self.offset_weight = offset_weight
        self.onset_pos_weight = onset_pos_weight
        self.frame_pos_weight = frame_pos_weight
        self.offset_pos_weight = offset_pos_weight

    def forward(self, onset_pred, frame_pred, offset_pred,
                onset_label, frame_label, offset_label):
        onset_pos_weight = torch.tensor(
            self.onset_pos_weight, device=onset_pred.device, dtype=onset_pred.dtype
        )
        frame_pos_weight = torch.tensor(
            self.frame_pos_weight, device=frame_pred.device, dtype=frame_pred.dtype
        )
        offset_pos_weight = torch.tensor(
            self.offset_pos_weight, device=offset_pred.device, dtype=offset_pred.dtype
        )
        onset_loss  = F_func.binary_cross_entropy_with_logits(
            onset_pred, onset_label, pos_weight=onset_pos_weight)
        frame_loss  = F_func.binary_cross_entropy_with_logits(
            frame_pred, frame_label, pos_weight=frame_pos_weight)
        offset_loss = F_func.binary_cross_entropy_with_logits(
            offset_pred, offset_label, pos_weight=offset_pos_weight)
        total = (self.onset_weight  * onset_loss +
                 self.frame_weight  * frame_loss +
                 self.offset_weight * offset_loss)
        return total, onset_loss, frame_loss, offset_loss




if __name__ == "__main__":
    cfg = {
        'model': {
            'h_dim': 128,
            'conv_channels': 32,
            'num_cycles': 2,
            'num_transformer_layers': 1,
            'nhead_fh': 8,
            'nhead_ht': 8,
            'nhead_tf': 6,
            'dim_feedforward': 512,
            'dropout': 0.1,
            'num_pitches': 48,
        },
        'audio': {
            'n_octaves': 6,
            'bins_per_octave': 48,
        }
    }

    model = CFT_v6(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CFT_v6 参数量: {n_params:,}")

    # 验证前向传播
    x = torch.randn(2, 288, 256)  # batch=2, F=288, T=256
    onset, frame, offset = model(x)
    print(f"输入: {x.shape}")
    print(f"onset: {onset.shape}  frame: {frame.shape}  offset: {offset.shape}")
    assert onset.shape == (2, 256, 48), f"输出形状错误: {onset.shape}"
    print("✅ 前向传播验证通过！")

    # 验证损失函数
    criterion = CFTLoss()
    label = torch.zeros(2, 256, 48)
    label[:, 10:20, 5] = 1.0
    loss, ol, fl, ofl = criterion(onset, frame, offset, label, label, label)
    print(f"Loss: {loss.item():.4f}  (onset={ol.item():.4f}, frame={fl.item():.4f}, offset={ofl.item():.4f})")
    print("✅ 损失函数验证通过！")
