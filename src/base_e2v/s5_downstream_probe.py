#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DisEmo2vec Step 5 - Layer S5: Frozen Feature Slice Extractor & 0.20M Linear Probe
=================================================================================
Contains `[SYM-E2V-S5-01]` (`FrozenFeatureExtractorAndLinearProbe`), locking the exact
slice boundary between global Utterance tokens (`x[:, :C, :]`) and Frame tokens (`x[:, C:, :]`)
for downstream evaluation in `eval/eval_all_8_metrics.py`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np

from models.hammer_slots.slot_base import BaseModule, HAS_TORCH, torch, to_numpy


class FrozenFeatureExtractorAndLinearProbe(BaseModule):
    """
    [SYM-E2V-S5-01] 下游冻结主干特征切片导出器与 0.20M 轻量线性探针接口
    @bce_contract:
      uid: "SYM-E2V-S5-01"
      layer: "S5"
      name: "下游冻结主干 Utt/Frm 切片导出与 0.20M 线性分类探针"
      file_rel_path: "src/base_e2v/s5_downstream_probe.py"
      tensor_input: "encoder_out: FloatTensor[B, C+N_frm, 768], frame_padding_mask: BoolTensor[B, N_frm]"
      tensor_output: "z_utt_emo: FloatTensor[B, 768], x_frm_emo: FloatTensor[B, N_frm, 768]"
      signal_math_assumptions:
        - "【切片边界假设 S5-A1】Student 顶层输出序列长度为 C+N_frm：前 C 个位置 [:C] 为全局 Chunk Tokens，第 C 个位置之后 [C:] 为 50Hz 声学帧特征序列 X^S_emo。"
        - "【探针预算假设 S5-A2】下游评测时主干网络 100% 冻结 (requires_grad=False)，仅训练 768 -> 256 -> num_classes 的两层轻量线性探针 (参数量 197,892 <= 0.20M)。"
      hard_invariants:
        - "【保命护栏 I7】下游情感分类评测时，若启用 [SLOT-H2] 双分支切分 (前5个为情感 Token，后5个为内容海绵 Token)，必须严格只抽取前 5 个情感 Token 求平均作为 z_utt_emo，直接丢弃后 5 个内容海绵 Token 与分类头 W_c，确保下游推理零额外参数开销！"
      broken_assumptions:
        - "【原版隐性缺陷 B5-01 (对应 Table 9 指标 2B)】原版在下游单抽全局向量 z_utt 时准确率仅 62.77%（比单抽帧特征 Frm 的 71.79% 暴跌 9.02%），甚至把 z_utt 与 Frm 拼接后 (71.60%) 比不拼还低，证明原版全局 Token 已沦为负资产。"
      reserved_slot_id: "Downstream evaluation bridge to `eval/eval_all_8_metrics.py`"
    """

    def __init__(
        self,
        num_extra_tokens: int = 10,
        num_emo_tokens: int = 5,
        embed_dim: int = 768,
        hidden_dim: int = 256,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.num_extra_tokens = int(num_extra_tokens)
        self.num_emo_tokens = int(num_emo_tokens)
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.probe_param_count = (
            self.embed_dim * self.hidden_dim
            + self.hidden_dim
            + self.hidden_dim * self.num_classes
            + self.num_classes
        )
        assert self.probe_param_count <= 205_000, "Downstream probe exceeds 0.20M budget!"

    def slice_utterance_and_frame_features(
        self,
        encoder_out: Any,
        frame_padding_mask: Optional[Any] = None,
        use_dual_branch_h2: bool = True,
    ) -> Dict[str, Any]:
        """
        Slices `encoder_out [B, C+N_frm, 768]` into:
        - `z_utt_emo [B, 768]` (from first 5 emotion tokens when `use_dual_branch_h2=True`, else all C=10 tokens)
        - `z_utt_cont [B, 768]` (from last 5 content sponge tokens, used ONLY for Metric 8 probe/R^2)
        - `x_frm_emo [B, N_frm, 768]` and `x_frm_pooled [B, 768]` (valid-frame mean over `[C:]`)
        """
        c = self.num_extra_tokens
        k_emo = self.num_emo_tokens if use_dual_branch_h2 else c

        out_np = to_numpy(encoder_out)
        chunks_np = out_np[:, :c, :]
        frames_np = out_np[:, c:, :]

        z_utt_emo = np.mean(chunks_np[:, :k_emo, :], axis=1)
        z_utt_cont = np.mean(chunks_np[:, self.num_emo_tokens : c, :], axis=1)

        if frame_padding_mask is not None:
            pad_np = np.asarray(to_numpy(frame_padding_mask), dtype=bool)
            valid_f = (~pad_np).astype(np.float32)[:, :, None]
            n_valid = np.maximum(np.sum(valid_f, axis=1), 1.0)
            frames_np = frames_np * valid_f
            x_frm_pooled = np.sum(frames_np, axis=1) / n_valid
        else:
            x_frm_pooled = np.mean(frames_np, axis=1)

        if HAS_TORCH and isinstance(encoder_out, torch.Tensor):
            dev = encoder_out.device
            return {
                "z_utt_emo": torch.from_numpy(z_utt_emo).to(dev),
                "z_utt_cont": torch.from_numpy(z_utt_cont).to(dev),
                "x_frm_emo": torch.from_numpy(frames_np).to(dev),
                "x_frm_pooled": torch.from_numpy(x_frm_pooled).to(dev),
                "probe_param_count": self.probe_param_count,
            }
        return {
            "z_utt_emo": z_utt_emo,
            "z_utt_cont": z_utt_cont,
            "x_frm_emo": frames_np,
            "x_frm_pooled": x_frm_pooled,
            "probe_param_count": self.probe_param_count,
        }
