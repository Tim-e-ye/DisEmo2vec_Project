#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DisEmo2vec Step 5 - Layer S3 (Part 2): Base emotion2vec Utterance & Masked Frame Criterion
==========================================================================================
Contains `[SYM-E2V-S3-03]` (`BaseEmotion2VecCriterion`), exposing the exact lines where
original `emotion2vec` suffers from Broken Assumption A (uniform 1/N dilution) and
Broken Assumption B (single-branch lexical entanglement), and providing the direct host
hook for `models/hammer_modules.py` (`[SLOT-H2]` & `[SLOT-H3]`).
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np

from models.hammer_slots.slot_base import BaseModule, HAS_TORCH, torch, F, to_numpy


class BaseEmotion2VecCriterion(BaseModule):
    """
    [SYM-E2V-S3-03] 原版 emotion2vec 帧级与句级联合蒸馏损失函数
    @bce_contract:
      uid: "SYM-E2V-S3-03"
      layer: "S3"
      name: "原版无差别均值池化句级损失与掩码帧损失计算器"
      file_rel_path: "src/base_e2v/s3_criterion_loss.py"
      tensor_input: "student_out: FloatTensor[B, C+N_frm, 768], y_teacher: FloatTensor[B, N_frm, 768], mask_indices: BoolTensor[B, N_frm], frame_padding_mask: BoolTensor[B, N_frm]"
      tensor_output: "loss_dict: Dict[loss_total, loss_frm, loss_utt, utt_target]"
      signal_math_assumptions:
        - "【数学假设 S3-A5】帧级损失 L_Frm 仅在被时间掩码覆盖且非补零的有效帧集合 M = (mask_indices & ~frame_padding_mask) 上计算 MSE / Smooth-L1 距离。"
        - "【量级假设 S3-A6】由于 y_teacher 每维方差为 1.0，未缩放的单帧各维平方误差均值为 O(1.0)，总损失 L_total = L_Frm + alpha * L_Utt (默认 alpha = 1.0)。"
      hard_invariants:
        - "【保命护栏 I3】计算任何跨时间步聚合 (均值或 Softmax 池化) 时，必须将 frame_padding_mask==True 的补零帧权重严格置零，分母必须取真实有效帧数 N_valid = (~frame_padding_mask).sum(dim=1)！"
        - "【保命护栏 I4】损失累加必须在 FP32 精度 (.float() / float32) 下执行，且 y_teacher 在进入损失前必须执行 .detach() 阻断向 Teacher 的梯度回传！"
      broken_assumptions:
        - "【失效假设 A (对应钉子 1 / Table 9 双重退化 & 指标 ⑦ 长句稀释)】utt_target = (y_teacher * valid_mask).sum(1) / N_valid 强行假设整句话所有有声帧的情绪贡献度恒为均匀 1/N_valid。在自然对话 (MELD) 中 80% 为平淡过场帧，导致均值向量被平淡背景严重主宰与稀释！"
        - "【失效假设 B (对应钉子 2 / 指标 ⑤ 跨剧本掉点 & 指标 ⑧ 音素泄漏)】z_student = student_out[:, :C, :].mean(dim=1) 将全部 C=10 个前置 Chunk Tokens 混为单一向量直接逼近含词汇音素的 utt_target，未给词汇内容留出分流排泄通道！"
      reserved_slot_id: "Direct host for [SLOT-H2: HammerSlotH2_Disentangler] & [SLOT-H3: HammerSlotH3_TemporalAggregator]"
    """

    def __init__(self, num_extra_tokens: int = 10, utt_loss_weight: float = 1.0) -> None:
        super().__init__()
        self.num_extra_tokens = int(num_extra_tokens)
        self.utt_loss_weight = float(utt_loss_weight)

    def forward(
        self,
        student_out: Any,
        y_teacher: Any,
        mask_indices: Any,
        frame_padding_mask: Any,
    ) -> Dict[str, Any]:
        """
        Computes the exact unmodified original `emotion2vec` loss (`L_Frm + alpha * L_Utt`)
        while strictly enforcing Hard Invariants I3 (`padding_mask` exclusion) and I4 (`FP32` + `.detach()`).
        """
        c = self.num_extra_tokens

        if HAS_TORCH and isinstance(student_out, torch.Tensor):
            # Hard Invariant I4: Cast to FP32 and detach Teacher target
            s_out = student_out.float()
            y_t = y_teacher.detach().float()
            pad_bool = frame_padding_mask.bool().to(s_out.device)
            mask_bool = mask_indices.bool().to(s_out.device)

            s_chunks = s_out[:, :c, :]   # [B, C, 768]
            s_frames = s_out[:, c:, :]   # [B, N_frm, 768]

            # Hard Invariant I3: Valid non-padded mask [B, N_frm, 1]
            valid_f = (~pad_bool).to(s_out.dtype).unsqueeze(-1)
            n_valid = valid_f.sum(dim=1).clamp_min(1.0)  # [B, 1]

            # 🎯 [Broken Assumption A — Target for SLOT-H3]: Uniform 1/N_valid temporal mean over all valid frames
            utt_target = (y_t * valid_f).sum(dim=1) / n_valid

            # 🎯 [Broken Assumption B — Target for SLOT-H2]: Single-branch mean over all C=10 Chunk Tokens
            z_student = s_chunks.mean(dim=1)
            loss_utt = F.mse_loss(z_student, utt_target)

            # Masked Frame Loss on active masked positions M = (mask_bool & ~pad_bool)
            active_m = mask_bool & (~pad_bool)
            per_frame_mse = torch.mean((s_frames - y_t) ** 2, dim=-1)
            loss_frm = torch.mean(per_frame_mse[active_m]) if active_m.any() else torch.mean(per_frame_mse)
            loss_total = loss_frm + self.utt_loss_weight * loss_utt

            return {
                "loss_total": loss_total,
                "loss_frm": loss_frm,
                "loss_utt": loss_utt,
                "utt_target": utt_target,
                "z_student": z_student,
            }

        # NumPy FP32 execution path
        s_out_np = to_numpy(student_out).astype(np.float32)
        y_t_np = to_numpy(y_teacher).astype(np.float32)
        pad_np = np.asarray(to_numpy(frame_padding_mask), dtype=bool)
        mask_np = np.asarray(to_numpy(mask_indices), dtype=bool)

        s_chunks_np = s_out_np[:, :c, :]
        s_frames_np = s_out_np[:, c:, :]

        valid_np = (~pad_np).astype(np.float32)[:, :, None]
        n_valid_np = np.maximum(np.sum(valid_np, axis=1), 1.0)

        # 🎯 [Broken Assumption A — Target for SLOT-H3]
        utt_target_np = np.sum(y_t_np * valid_np, axis=1) / n_valid_np

        # 🎯 [Broken Assumption B — Target for SLOT-H2]
        z_student_np = np.mean(s_chunks_np, axis=1)
        loss_utt_val = float(np.mean((z_student_np - utt_target_np) ** 2))

        active_m_np = mask_np & (~pad_np)
        per_frame_mse_np = np.mean((s_frames_np - y_t_np) ** 2, axis=-1)
        loss_frm_val = (
            float(np.mean(per_frame_mse_np[active_m_np]))
            if np.any(active_m_np)
            else float(np.mean(per_frame_mse_np))
        )
        loss_total_val = loss_frm_val + self.utt_loss_weight * loss_utt_val

        return {
            "loss_total": loss_total_val,
            "loss_frm": loss_frm_val,
            "loss_utt": loss_utt_val,
            "utt_target": utt_target_np,
            "z_student": z_student_np,
        }
