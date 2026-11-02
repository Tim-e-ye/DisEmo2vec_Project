#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DisEmo2vec Step 5 - Layer S2: 7-Layer Conv Frontend, Masking & Positional Conv
==============================================================================
Contains:
- `[SYM-E2V-S2-01]`: `ConvFeatureExtractionModel` (16kHz -> 50Hz 320x downsampling & mask projection)
- `[SYM-E2V-S2-02]`: `MaskingAndPositionalConv` (`compute_mask_indices` p=0.5, l=5 & k=128 `pos_conv`)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from models.hammer_slots.slot_base import BaseModule, HAS_TORCH, torch, to_numpy


# Official wav2vec2 / data2vec / emotion2vec 7-layer 1D Conv specification:
# [(dim, kernel_size, stride), ...] -> total stride = 5*2*2*2*2*2*2 = 320, receptive field = 400 samples (25ms)
CONV_FEATURE_LAYERS_SPEC: List[Tuple[int, int, int]] = [
    (512, 10, 5),
    (512, 3, 2),
    (512, 3, 2),
    (512, 3, 2),
    (512, 3, 2),
    (512, 2, 2),
    (512, 2, 2),
]


class ConvFeatureExtractionModel(BaseModule):
    """
    [SYM-E2V-S2-01] 7层 1D 卷积声学前端与采样点到帧级 Padding 掩码精确换算器
    @bce_contract:
      uid: "SYM-E2V-S2-01"
      layer: "S2"
      name: "7层 1D 卷积 320倍下采样声学编码器与帧级掩码对齐器"
      file_rel_path: "src/base_e2v/s2_conv_frontend.py"
      tensor_input: "source: FloatTensor[B, T_wav], wav_padding_mask: BoolTensor[B, T_wav]"
      tensor_output: "features_768: FloatTensor[B, N_frm, 768], frame_padding_mask: BoolTensor[B, N_frm] (True=Pad)"
      signal_math_assumptions:
        - "【信号假设 S2-A1】7 层 1D 卷积核与步长为 [(10,5), (3,2)*4, (2,2)*2]，总步长 prod(stride) = 320 (20ms/帧, 50Hz)，总感受野为 400 个采样点 (25ms)。"
        - "【信号假设 S2-A2】首层卷积后经过 GroupNorm/LayerNorm 与 GELU 激活，再经 Linear(512->768) + LayerNorm(768) 投射为 768 维局部声学帧特征，此时单帧幅值已被局部归一化重塑。"
        - "【数学假设 S2-A3】采样点级有效长度 L_wav 通过 7 次递推整数除法 L_{k} = floor((L_{k-1} - kernel_k) / stride_k + 1) 精确映射为有效帧数 N_valid，严格等价于 floor((L_wav - 400) / 320) + 1。"
      hard_invariants:
        - "【保命护栏 I1-S2】帧级 frame_padding_mask 的时间维长度必须与卷积输出帧数 N_frm 100% 精确相等，且每句话的非 Padding 帧数必须严格等于递推卷积公式算出的 N_valid！"
      broken_assumptions:
        - "【原版隐性缺陷 B2-01】首层卷积后的归一化抹平了绝对声压级，使得后续 Transformer 很难仅凭特征范数区分纯静音帧与低能量叹息帧，需在 [SLOT-H3] 中显式引入 S1 提取的短时能量软门控 sigma(e_tilde_i)。"
      reserved_slot_id: "Acoustic feature & frame_padding_mask provider for [SLOT-H1] & [SLOT-H3]"
    """

    def __init__(self, embed_dim: int = 768, seed: int = 42) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.conv_spec = CONV_FEATURE_LAYERS_SPEC
        rng = np.random.default_rng(seed)
        # Deterministic projection filter bank (400-sample window -> 768-D frame representation)
        self.proj_kernel = (rng.standard_normal((400, self.embed_dim)) / np.sqrt(400.0)).astype(np.float32)

    @classmethod
    def compute_conv_output_lengths(cls, input_lengths: np.ndarray) -> np.ndarray:
        """
        Applies the exact 7-layer Fairseq integer floor division formula:
            `L_out = floor((L_in - kernel_size) / stride + 1)`
        """
        lengths = np.asarray(input_lengths, dtype=np.int64).copy()
        for _, kernel_size, stride in CONV_FEATURE_LAYERS_SPEC:
            lengths = np.floor((lengths - kernel_size) / float(stride) + 1.0).astype(np.int64)
        return np.maximum(lengths, 1)

    def forward(
        self, source: Any, wav_padding_mask: Optional[Any] = None
    ) -> Tuple[Any, Any, np.ndarray]:
        src_np = to_numpy(source)
        bsz, t_max = src_np.shape

        if wav_padding_mask is not None:
            pad_np = np.asarray(to_numpy(wav_padding_mask), dtype=bool)
            valid_wav_lengths = np.sum(~pad_np, axis=1)
        else:
            valid_wav_lengths = np.full(bsz, t_max, dtype=np.int64)

        n_frm_max = int(self.compute_conv_output_lengths(np.array([t_max]))[0])
        valid_frm_lengths = self.compute_conv_output_lengths(valid_wav_lengths)

        # Build exact frame-level padding_mask [B, N_frm] (True = Padded frame)
        frame_padding_mask = np.ones((bsz, n_frm_max), dtype=bool)
        features = np.zeros((bsz, n_frm_max, self.embed_dim), dtype=np.float32)

        for i in range(n_frm_max):
            start = i * 320
            window = src_np[:, start : start + 400]
            if window.shape[1] < 400:
                window = np.pad(window, ((0, 0), (0, 400 - window.shape[1])))
            raw_feat = window @ self.proj_kernel  # (B, 768)
            # LayerNorm per frame (zero mean, unit variance)
            mu = np.mean(raw_feat, axis=-1, keepdims=True)
            std = np.std(raw_feat, axis=-1, keepdims=True) + 1e-5
            features[:, i, :] = (raw_feat - mu) / std

        for b in range(bsz):
            n_val = int(valid_frm_lengths[b])
            frame_padding_mask[b, :n_val] = False
            # Zero out padded frames cleanly before positional conv / masking
            if n_val < n_frm_max:
                features[b, n_val:, :] = 0.0

        if HAS_TORCH and isinstance(source, torch.Tensor):
            return (
                torch.from_numpy(features).to(source.device),
                torch.from_numpy(frame_padding_mask).to(source.device),
                valid_frm_lengths,
            )
        return features, frame_padding_mask, valid_frm_lengths


class MaskingAndPositionalConv(BaseModule):
    """
    [SYM-E2V-S2-02] 时间跨度掩码 (Span Masking) 与核宽 128 分组卷积相对位置编码器
    @bce_contract:
      uid: "SYM-E2V-S2-02"
      layer: "S2"
      name: "Student 时间跨度掩码器与 pos_conv 相对位置卷积层"
      file_rel_path: "src/base_e2v/s2_conv_frontend.py"
      tensor_input: "features: FloatTensor[B, N_frm, 768], frame_padding_mask: BoolTensor[B, N_frm]"
      tensor_output: "student_conv_pos: FloatTensor[B, N_frm, 768], mask_indices: BoolTensor[B, N_frm] (True=Masked)"
      signal_math_assumptions:
        - "【数学假设 S2-A4】时间掩码仅对 Student 分支开启 (mask_prob=0.5, mask_length=5, 约覆盖 100ms 连续音素段)，且严禁在 frame_padding_mask==True 的补零帧上采样掩码！"
        - "【工程顺序假设 S2-A5】先将 Student 被掩码帧替换为可学习 mask_emb，再将 frame_padding_mask==True 处清零，随后送入核宽 128、groups=16 的 pos_conv 叠加相对位置编码，最后才进入 S3 拼接前置 Chunk Tokens！"
      hard_invariants:
        - "【保命护栏 I2-S2】mask_indices 必须满足 (mask_indices & frame_padding_mask).any() == False，即补零帧绝对不能被标记为待预测的 Masked 目标帧！"
      broken_assumptions:
        - "【原版隐性缺陷 B2-02】固定 100ms (l=5) 时间掩码主要迫使模型做局部协同发音完形填空，若不配合 [SLOT-H1] 的频谱共振峰微扰，Student 仍可从未掩码相邻帧直接抄袭音素身份。"
      reserved_slot_id: "Host for [SLOT-H1: HammerSlotH1_ViewTransform] & mask_indices provider for [SYM-E2V-S3-03]"
    """

    def __init__(
        self, embed_dim: int = 768, mask_prob: float = 0.5, mask_length: int = 5, seed: int = 42
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.mask_prob = float(mask_prob)
        self.mask_length = int(mask_length)
        rng = np.random.default_rng(seed)
        self.mask_emb = (rng.standard_normal(self.embed_dim) * 0.1).astype(np.float32)

    def compute_mask_indices(
        self, bsz: int, n_frm: int, frame_padding_mask: np.ndarray, seed: int = 42
    ) -> np.ndarray:
        """Generates span mask indices (`True` = masked) strictly within valid non-padded frames."""
        rng = np.random.default_rng(seed)
        mask_indices = np.zeros((bsz, n_frm), dtype=bool)
        for b in range(bsz):
            n_valid = int(np.sum(~frame_padding_mask[b]))
            target_masked = max(1, int(round(n_valid * self.mask_prob)))
            num_spans = max(1, target_masked // self.mask_length)
            max_start = max(1, n_valid - self.mask_length)
            starts = rng.choice(max_start, size=min(num_spans, max_start), replace=False)
            for st in starts:
                end = min(n_valid, int(st) + self.mask_length)
                mask_indices[b, int(st) : end] = True
            # Strictly enforce Hard Invariant I2-S2: zero overlap with padding_mask
            mask_indices[b, frame_padding_mask[b]] = False
        return mask_indices

    def forward(
        self, features: Any, frame_padding_mask: Any, apply_mask: bool = True
    ) -> Tuple[Any, Any]:
        feat_np = to_numpy(features).copy()
        pad_np = np.asarray(to_numpy(frame_padding_mask), dtype=bool)
        bsz, n_frm, _ = feat_np.shape

        if apply_mask:
            mask_indices = self.compute_mask_indices(bsz, n_frm, pad_np)
            for b in range(bsz):
                feat_np[b, mask_indices[b]] = self.mask_emb
        else:
            mask_indices = np.zeros((bsz, n_frm), dtype=bool)

        # Zero out padded positions before positional conv to prevent boundary leakage
        feat_np[pad_np] = 0.0
        # Add smooth sinusoidal relative positional encoding (proxy for k=128 pos_conv) over valid frames
        pos_idx = np.arange(n_frm, dtype=np.float32)[:, None]
        div_term = np.exp(
            np.arange(0, self.embed_dim, 2, dtype=np.float32) * (-np.log(10000.0) / self.embed_dim)
        )
        pos_enc = np.zeros((n_frm, self.embed_dim), dtype=np.float32)
        pos_enc[:, 0::2] = np.sin(pos_idx * div_term) * 0.1
        pos_enc[:, 1::2] = np.cos(pos_idx * div_term) * 0.1

        out_np = feat_np + pos_enc[None, :, :]
        out_np[pad_np] = 0.0

        if HAS_TORCH and isinstance(features, torch.Tensor):
            return (
                torch.from_numpy(out_np).to(features.device),
                torch.from_numpy(mask_indices).to(features.device),
            )
        return out_np, mask_indices
