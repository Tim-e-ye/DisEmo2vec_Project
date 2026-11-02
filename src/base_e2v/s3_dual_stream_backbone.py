#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DisEmo2vec Step 5 - Layer S3 (Part 1): Prepend Chunk Tokens & EMA Teacher Top-K Extractor
=========================================================================================
Contains:
- `[SYM-E2V-S3-01]`: `PrependChunkTokens` (prepends C=10 extra tokens & pads `False` to `padding_mask`)
- `[SYM-E2V-S3-02]`: `EMATeacherTargetExtractor` (unmasked Teacher top-k=8 InstanceNorm/LayerNorm whitening)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from models.hammer_slots.slot_base import BaseModule, HAS_TORCH, torch, to_numpy


class PrependChunkTokens(BaseModule):
    """
    [SYM-E2V-S3-01] 前置 C=10 全局 Chunk Tokens 注入与扩展 Padding 掩码对齐器
    @bce_contract:
      uid: "SYM-E2V-S3-01"
      layer: "S3"
      name: "前置 C=10 可学习 Chunk Tokens 拼接与掩码扩展器"
      file_rel_path: "src/base_e2v/s3_dual_stream_backbone.py"
      tensor_input: "conv_pos_features: FloatTensor[B, N_frm, 768], frame_padding_mask: BoolTensor[B, N_frm]"
      tensor_output: "extended_x: FloatTensor[B, C+N_frm, 768], extended_padding_mask: BoolTensor[B, C+N_frm]"
      signal_math_assumptions:
        - "【工程顺序假设 S3-A1】C=10 个可学习 extra_tokens 必须在 pos_conv 相对位置卷积之后、进入 12 层 Transformer Self-Attention 之前拼接到序列最前端！"
        - "【掩码假设 S3-A2】由于前置 C=10 个全局 Token 对所有样本均有效，frame_padding_mask [B, N_frm] 前端必须拼接全 False 张量 [B, C]，扩展为 [B, C+N_frm]。"
      hard_invariants:
        - "【保命护栏 I0-S3】前置 C=10 个 Chunk Tokens 在预训练阶段必须保持 requires_grad=True，且对应的前 C 位 padding_mask 必须恒为 False，确保全局注意力双向畅通！"
      broken_assumptions:
        - "【原版隐性缺陷 B3-01】原版将全部 C=10 个前置 Token 无差别混同为单一情绪通道，未将其划分为独立的情感专属 Token (前5个) 与词汇海绵 Token (后5个)。"
      reserved_slot_id: "Direct structural host for [SLOT-H2: HammerSlotH2_Disentangler.split_and_pool_chunks]"
    """

    def __init__(self, num_extra_tokens: int = 10, embed_dim: int = 768, seed: int = 42) -> None:
        super().__init__()
        self.num_extra_tokens = int(num_extra_tokens)
        self.embed_dim = int(embed_dim)
        rng = np.random.default_rng(seed)
        self.extra_tokens = (rng.standard_normal((1, self.num_extra_tokens, self.embed_dim)) * 0.02).astype(
            np.float32
        )

    def forward(self, conv_pos_features: Any, frame_padding_mask: Any) -> Tuple[Any, Any]:
        x_np = to_numpy(conv_pos_features)
        pad_np = np.asarray(to_numpy(frame_padding_mask), dtype=bool)
        bsz = x_np.shape[0]

        chunks = np.repeat(self.extra_tokens, bsz, axis=0)
        extended_x = np.concatenate([chunks, x_np], axis=1)  # (B, C + N_frm, 768)

        prefix_false = np.zeros((bsz, self.num_extra_tokens), dtype=bool)
        extended_padding_mask = np.concatenate([prefix_false, pad_np], axis=1)  # (B, C + N_frm)

        if HAS_TORCH and isinstance(conv_pos_features, torch.Tensor):
            return (
                torch.from_numpy(extended_x).to(conv_pos_features.device),
                torch.from_numpy(extended_padding_mask).to(conv_pos_features.device),
            )
        return extended_x, extended_padding_mask


class EMATeacherTargetExtractor(BaseModule):
    """
    [SYM-E2V-S3-02] EMA Teacher 顶部 K=8 层白化帧表征提取器
    @bce_contract:
      uid: "SYM-E2V-S3-02"
      layer: "S3"
      name: "EMA Teacher Top-K 白化帧目标生成器"
      file_rel_path: "src/base_e2v/s3_dual_stream_backbone.py"
      tensor_input: "clean_conv_features: FloatTensor[B, N_frm, 768], frame_padding_mask: BoolTensor[B, N_frm]"
      tensor_output: "y_teacher: FloatTensor[B, N_frm, 768] (whitened per-frame target, ||y^T_i||_2 ~ 27.71)"
      signal_math_assumptions:
        - "【信号假设 S3-A3】Teacher 全程输入未经掩码 (mask=False) 且未经视角微扰的干净声学特征，拥有完整双向上下文视野。"
        - "【数学假设 S3-A4】经每层无仿射参数 LayerNorm 白化并求 8 层平均后，每个有效帧向量 y^T_i in R^768 满足均值 mu=0、方差 sigma^2=1，其欧氏二范数恒定在 ||y^T_i||_2 = sqrt(768) ≈ 27.71！"
      hard_invariants:
        - "【保命护栏 I1】在提取顶部 k=8 层 Transformer 输出时，每一层输出必须先独立执行无仿射参数的 F.instance_norm / F.layer_norm (将每帧每层拉回零均值单位方差)，再对 8 层求算术平均并再次归一化，否则自监督蒸馏会在 100 步内坍缩为常数向量！"
        - "【保命护栏 I2】必须执行 y = y[:, self.num_extra_tokens:, :] 剥离前 C=10 个 Chunk Tokens，且将 frame_padding_mask==True 的补零帧严格置零，确保帧目标仅对应真实声学时间步 N_frm！"
      broken_assumptions:
        - "【原版隐性缺陷 B1】Teacher 的 8 层平均表征 y_teacher 中同时高度纠缠了副语言情绪韵律与词汇音素身份（中高层仍具极高音素可分性），直接作为全局回归目标会导致 Student 死记词汇捷径。"
      reserved_slot_id: "Upstream target provider for [SLOT-H2: HammerSlotH2_Disentangler] & [SLOT-H3: HammerSlotH3_TemporalAggregator]"
    """

    def __init__(
        self,
        num_layers: int = 12,
        top_k_layers: int = 8,
        num_extra_tokens: int = 10,
        embed_dim: int = 768,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.num_layers = int(num_layers)
        self.top_k_layers = int(top_k_layers)
        self.num_extra_tokens = int(num_extra_tokens)
        self.embed_dim = int(embed_dim)
        rng = np.random.default_rng(seed)
        # Orthogonal layer transformations simulating 12 Transformer layers
        self.layer_weights = [
            np.linalg.qr(rng.standard_normal((self.embed_dim, self.embed_dim)).astype(np.float32))[0]
            for _ in range(self.num_layers)
        ]

    @staticmethod
    def _layer_norm_no_affine(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Applies parameter-free LayerNorm across the 768 feature dimension per frame."""
        mu = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return ((x - mu) / np.sqrt(std * std + eps)).astype(np.float32)

    def forward(
        self, extended_clean_x: Any, extended_padding_mask: Any
    ) -> Tuple[Any, Dict[str, float]]:
        x_np = to_numpy(extended_clean_x).copy()
        pad_np = np.asarray(to_numpy(extended_padding_mask), dtype=bool)
        c_tokens = self.num_extra_tokens
        frame_pad_np = pad_np[:, c_tokens:]  # [B, N_frm]

        hidden = x_np
        top_k_whitened: List[np.ndarray] = []

        for layer_idx, w_mat in enumerate(self.layer_weights):
            # Mask out padded positions before each layer transformation
            hidden[pad_np] = 0.0
            hidden = hidden @ w_mat + 0.25 * hidden
            if layer_idx >= (self.num_layers - self.top_k_layers):
                # Hard Invariant I2: Strip prepended C=10 Chunk Tokens -> [B, N_frm, 768]
                frame_only = hidden[:, c_tokens:, :].copy()
                # Hard Invariant I1: Per-layer affine-free whitening before averaging top-k=8 layers
                whitened_layer = self._layer_norm_no_affine(frame_only)
                whitened_layer[frame_pad_np] = 0.0
                top_k_whitened.append(whitened_layer)

        # Average top-k=8 whitened layers and re-normalize valid frames to exact unit variance
        y_teacher = np.mean(np.stack(top_k_whitened, axis=0), axis=0)
        y_teacher = self._layer_norm_no_affine(y_teacher)
        y_teacher[frame_pad_np] = 0.0

        valid_vectors = y_teacher[~frame_pad_np]
        mean_val = float(np.mean(valid_vectors))
        std_val = float(np.std(valid_vectors))
        l2_norm_mean = float(np.mean(np.linalg.norm(valid_vectors, axis=-1)))

        stats = {
            "valid_frame_mean": round(mean_val, 6),
            "valid_frame_std": round(std_val, 6),
            "valid_frame_l2_norm": round(l2_norm_mean, 4),
            "theoretical_sqrt_768": round(float(np.sqrt(self.embed_dim)), 4),
        }

        if HAS_TORCH and isinstance(extended_clean_x, torch.Tensor):
            return torch.from_numpy(y_teacher).to(extended_clean_x.device).detach(), stats
        return y_teacher, stats
