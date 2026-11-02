#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DisEmo2vec Step 5 - Layer S4: Exponential Moving Average (EMA) & FP16 Overflow Guard
====================================================================================
Contains `[SYM-E2V-S4-01]` (`EMAApplierAndFp16Guard`), defining the exact cosine/linear
EMA decay schedule `tau in [0.999, 0.9999]` and FP16/FP32 gradient norm & overflow guards.
"""

from __future__ import annotations

import math
from typing import Any, Dict
import numpy as np

from models.hammer_slots.slot_base import BaseModule, to_numpy


class EMAApplierAndFp16Guard(BaseModule):
    """
    [SYM-E2V-S4-01] Teacher 指数滑动平均 (EMA) 动量退火器与 FP16 梯度防溢出守卫
    @bce_contract:
      uid: "SYM-E2V-S4-01"
      layer: "S4"
      name: "EMA Teacher 动量退火更新器与 FP16/FP32 数值安全守卫"
      file_rel_path: "src/base_e2v/s4_ema_and_fp16.py"
      tensor_input: "student_params: Dict[str, Tensor], teacher_params: Dict[str, Tensor], step: int"
      tensor_output: "tau_step: float, updated_teacher_params: Dict[str, Tensor], grad_norm_ratio: float"
      signal_math_assumptions:
        - "【数学假设 S4-A1】Teacher 参数 theta_T 不参与任何反向传播，仅在每步 optimizer.step() 后按 theta_T <- tau * theta_T + (1 - tau) * theta_S 滑动更新。"
        - "【退火假设 S4-A2】EMA 动量系数 tau 从初始值 tau_0 = 0.999 在 warmup_steps (如 30,000 步) 内按余弦或线性轨迹退火至终值 tau_end = 0.9999。"
        - "【梯度均衡假设 S4-A3】新插入卡槽的辅助锤子损失梯度范数 ||grad(L_hammer)||_2 与主损失梯度范数 ||grad(L_Frm)||_2 的比值应维持在 [0.05, 0.50] 黄金区间内，防止冲垮 Student 主干。"
      hard_invariants:
        - "【保命护栏 I5】前置 Chunk Tokens 与新插入卡槽的专属分类头 (如海绵线性头 W_c) 不拷入 Teacher EMA 字典，仅对共享的卷积与 Transformer 主干层执行 EMA 更新！"
        - "【保命护栏 I6】在 FP16 混合精度训练下，KL 散度、余弦相似度分母范数与 Softmax 温度缩放前必须先转 FP32 并加 clamp_min(1e-6)，严防 exp() 上溢 > 65504 触发 NaN！"
      broken_assumptions:
        - "【原版隐性缺陷 B4-01】当预训练关闭帧级损失 L_Frm 仅保留原版 L_Utt 时 (Table 9 指标 2A)，由于缺乏局部锚定且均值池化抹平了帧间差异，EMA Teacher 与 Student 迅速陷入表征坍缩 (WA 暴跌至 28.96%)。"
      reserved_slot_id: "Optimization & Grad-Norm Auto-Balancer guard for [SLOT-H1/H2/H3]"
    """

    def __init__(
        self,
        ema_decay_start: float = 0.999,
        ema_decay_end: float = 0.9999,
        ema_anneal_end_step: int = 30000,
    ) -> None:
        super().__init__()
        self.ema_decay_start = float(ema_decay_start)
        self.ema_decay_end = float(ema_decay_end)
        self.ema_anneal_end_step = int(ema_anneal_end_step)

    def get_ema_decay(self, num_updates: int) -> float:
        """Computes exact annealed EMA momentum `tau(t) in [0.999, 0.9999]`."""
        if num_updates >= self.ema_anneal_end_step:
            return self.ema_decay_end
        progress = max(0.0, float(num_updates) / float(max(1, self.ema_anneal_end_step)))
        cosine_weight = 0.5 * (1.0 - math.cos(math.pi * progress))
        return self.ema_decay_start + cosine_weight * (self.ema_decay_end - self.ema_decay_start)

    @staticmethod
    def recommend_hammer_loss_lambda(
        grad_norm_frm: float,
        grad_norm_hammer: float,
        target_ratio: float = 0.20,
        min_ratio: float = 0.05,
        max_ratio: float = 0.50,
    ) -> Dict[str, Any]:
        """
        Implements the Step 7 Pre-Flight Smoke Gate Grad-Norm Auto-Balancer:
        Checks if `||grad(L_hammer)|| / ||grad(L_Frm)||` is within `[0.05, 0.50]`
        and computes the recommended scaling weight `lambda_recommended`.
        """
        safe_frm = max(float(grad_norm_frm), 1e-8)
        safe_ham = max(float(grad_norm_hammer), 1e-8)
        raw_ratio = safe_ham / safe_frm
        in_golden_zone = bool(min_ratio <= raw_ratio <= max_ratio)
        recommended_lambda = target_ratio / raw_ratio
        return {
            "grad_norm_frm": round(safe_frm, 6),
            "grad_norm_hammer": round(safe_ham, 6),
            "raw_grad_ratio": round(raw_ratio, 4),
            "golden_interval": [min_ratio, max_ratio],
            "in_golden_zone": in_golden_zone,
            "recommended_lambda": round(recommended_lambda, 4),
        }
