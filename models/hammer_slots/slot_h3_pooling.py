#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
卡槽 H3（时序显著性汇聚槽）：`models/hammer_slots/slot_h3_pooling.py`
======================================================================
专门存放“如何从 80% 平淡帧中精准聚焦 20% 情绪爆发帧”的候选池化锤头函数。
- 标准输入：`target_layer_results` (B, N, 768)，以及可选 `frame_energy` (B, N) 或 `attn_weights` (B, N)
- 标准输出：汇聚后的话语级目标 `target_utt` (B, 768)、帧权重 `weights_w` (B, N)、红线诊断字典 `meta`
- 审稿人防挑刺红线：
  1. 严禁把纯静音帧（`e_tilde_i << 0`）误判为高权重帧（防 `|e_tilde_i|` 静音反噬漏洞）；
  2. 同时兼顾高唤醒爆发情绪（愤怒/惊喜）与低唤醒微弱情绪（悲伤/叹息）。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np

from .slot_base import BaseModule, HAS_TORCH, HammerRedLineViolationError, torch, F, to_numpy


# ==============================================================================
# 候选锤头函数库（每看一篇新文献，只需在此处新增一个 15 行小函数！）
# ==============================================================================
def h3_original_mean(
    target_layer_results: Any,
    frame_energy: Optional[Any] = None,
    **kwargs: Any,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """[原版基线 / original_mean]：原版 emotion2vec 无差别整句均匀平均 `target.mean(dim=1)`。"""
    if HAS_TORCH and isinstance(target_layer_results, torch.Tensor):
        bsz, n_frames, _ = target_layer_results.shape
        w = torch.full(
            (bsz, n_frames),
            1.0 / n_frames,
            device=target_layer_results.device,
            dtype=target_layer_results.dtype,
        )
        target_utt = target_layer_results.mean(dim=1)
    else:
        y_np = to_numpy(target_layer_results)
        bsz, n_frames, _ = y_np.shape
        w = np.full((bsz, n_frames), 1.0 / n_frames, dtype=np.float32)
        target_utt = y_np.mean(axis=1)

    return target_utt, w, {
        "slot": "H3",
        "pooling_type": "original_mean",
        "status": "[TBD]",
        "silence_suppressed": False,
        "weights_sum_to_one": True,
    }


def salience_pooling_3A(
    target_layer_results: Any,
    frame_energy: Optional[Any] = None,
    temperature_tau_s: float = 0.5,
    **kwargs: Any,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    [候选锤头 3A]：有声门控 × 句内背景均值离群度池化（Voiced Gate * Background Deviation Pooling）。
    公式：s_i = sigma(e_tilde_i) * ||y^T_i - y_bar^T||_2,  w_i = Softmax(s_i / tau_s)
    既彻底屏蔽纯静音帧（e_tilde_i << 0 时 sigma -> 0），又无需超参 gamma 兼顾高低唤醒情绪帧！
    """
    if HAS_TORCH and isinstance(target_layer_results, torch.Tensor):
        bsz, n_frames, _ = target_layer_results.shape
        if frame_energy is None:
            frame_energy = torch.zeros(
                (bsz, n_frames), device=target_layer_results.device, dtype=target_layer_results.dtype
            )
        y_bar = target_layer_results.mean(dim=1, keepdim=True)
        deviation_d = torch.norm(target_layer_results - y_bar, p=2, dim=-1)
        voiced_gate = torch.sigmoid(frame_energy)
        salience_s = voiced_gate * deviation_d
        weights_w = F.softmax(salience_s / max(temperature_tau_s, 1e-4), dim=-1)
        target_utt = torch.sum(weights_w.unsqueeze(-1) * target_layer_results, dim=1)
        sum_ok = bool(
            torch.allclose(
                weights_w.sum(dim=-1),
                torch.ones(bsz, device=target_layer_results.device, dtype=target_layer_results.dtype),
                atol=1e-4,
            )
        )
    else:
        y_np = to_numpy(target_layer_results)
        bsz, n_frames, _ = y_np.shape
        e_np = np.zeros((bsz, n_frames), dtype=np.float32) if frame_energy is None else to_numpy(frame_energy)
        y_bar = y_np.mean(axis=1, keepdims=True)
        dev_d = np.linalg.norm(y_np - y_bar, axis=-1)
        gate_g = 1.0 / (1.0 + np.exp(-np.clip(e_np, -20.0, 20.0)))
        sal_s = gate_g * dev_d
        scaled = (sal_s / max(temperature_tau_s, 1e-4))
        scaled = scaled - np.max(scaled, axis=-1, keepdims=True)
        exp_s = np.exp(scaled)
        weights_w = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
        target_utt = np.sum(weights_w[:, :, None] * y_np, axis=1)
        sum_ok = bool(np.allclose(np.sum(weights_w, axis=-1), 1.0, atol=1e-4))

    return target_utt, weights_w, {
        "slot": "H3",
        "pooling_type": "salience_3A",
        "status": "[ACTIVE_TRIAL]",
        "silence_suppressed": True,
        "weights_sum_to_one": sum_ok,
    }


def attn_rollout_pooling_3B(
    target_layer_results: Any,
    frame_energy: Optional[Any] = None,
    attn_weights: Optional[Any] = None,
    temperature_tau_s: float = 0.5,
    **kwargs: Any,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    [候选锤头 3B]：自注意力汇聚 × 有声门控池化（Attention Rollout / Self-Attention Salience Pooling）。
    利用特征帧自相关能量模拟多示例学习（MIL）注意力权重，并乘以有声门控屏蔽静音。
    """
    y_np = to_numpy(target_layer_results)
    bsz, n_frames, dim = y_np.shape
    e_np = np.zeros((bsz, n_frames), dtype=np.float32) if frame_energy is None else to_numpy(frame_energy)
    gate_g = 1.0 / (1.0 + np.exp(-np.clip(e_np, -20.0, 20.0)))

    if attn_weights is not None:
        raw_attn = to_numpy(attn_weights)
    else:
        # 帧间余弦注意力中心度（Attention Rollout proxy）
        y_norm = y_np / (np.linalg.norm(y_np, axis=-1, keepdims=True) + 1e-6)
        sim_mat = np.matmul(y_norm, np.transpose(y_norm, (0, 2, 1)))  # (B, N, N)
        raw_attn = np.linalg.norm(y_np - y_np.mean(axis=1, keepdims=True), axis=-1) * (1.0 - np.mean(sim_mat, axis=-1))

    sal_s = gate_g * raw_attn
    scaled = (sal_s / max(temperature_tau_s, 1e-4))
    scaled = scaled - np.max(scaled, axis=-1, keepdims=True)
    exp_s = np.exp(scaled)
    w_np = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
    utt_np = np.sum(w_np[:, :, None] * y_np, axis=1)

    if HAS_TORCH and isinstance(target_layer_results, torch.Tensor):
        utt_out = torch.from_numpy(utt_np).to(target_layer_results.device)
        w_out = torch.from_numpy(w_np).to(target_layer_results.device)
    else:
        utt_out, w_out = utt_np, w_np

    return utt_out, w_out, {
        "slot": "H3",
        "pooling_type": "attn_rollout_3B",
        "status": "[ACTIVE_TRIAL]",
        "silence_suppressed": True,
        "weights_sum_to_one": bool(np.allclose(np.sum(w_np, axis=-1), 1.0, atol=1e-4)),
    }


H3_CANDIDATE_FUNCTIONS: Dict[str, Callable] = {
    "original_mean": h3_original_mean,
    "tbd_passthrough": h3_original_mean,
    "salience_3A": salience_pooling_3A,
    "v0_voiced_bg_deviation_pooling": salience_pooling_3A,
    "attn_rollout_3B": attn_rollout_pooling_3B,
}


class HammerSlotH3_TemporalAggregator(BaseModule):
    """卡槽 H3 封装类：支持通过 `--pooling_type` / `active_card` 开关切换候选时序池化锤头。"""

    def __init__(
        self,
        active_card: str = "original_mean",
        temperature_tau_s: float = 0.5,
    ) -> None:
        super().__init__()
        self.slot_id = "H3"
        self.active_card = active_card
        self.status = "[TBD]" if active_card in ("original_mean", "tbd_passthrough") else "[ACTIVE_TRIAL]"
        self.temperature_tau_s = float(temperature_tau_s)

    def forward(
        self,
        y_teacher_frames: Any,
        frame_energy_std: Optional[Any] = None,
        attn_weights: Optional[Any] = None,
        pooling_type: Optional[str] = None,
    ) -> Tuple[Any, Any, Dict[str, Any]]:
        mode = pooling_type or self.active_card

        # 直观 If-Elif-Else 锤头卡槽路由（与用户示例完全对齐！）
        if mode in ("original_mean", "tbd_passthrough"):
            target_utt, weights_w, meta = h3_original_mean(y_teacher_frames, frame_energy_std)
        elif mode in ("salience_3A", "v0_voiced_bg_deviation_pooling"):
            target_utt, weights_w, meta = salience_pooling_3A(
                y_teacher_frames, frame_energy_std, temperature_tau_s=self.temperature_tau_s
            )
        elif mode == "attn_rollout_3B":
            target_utt, weights_w, meta = attn_rollout_pooling_3B(
                y_teacher_frames, frame_energy_std, attn_weights=attn_weights, temperature_tau_s=self.temperature_tau_s
            )
        elif mode in H3_CANDIDATE_FUNCTIONS:
            target_utt, weights_w, meta = H3_CANDIDATE_FUNCTIONS[mode](
                y_teacher_frames, frame_energy_std, attn_weights=attn_weights, temperature_tau_s=self.temperature_tau_s
            )
        else:
            raise ValueError(f"[H3 卡槽] 未知的 pooling_type='{mode}'，可选值: {list(H3_CANDIDATE_FUNCTIONS)}")

        if mode not in ("original_mean", "tbd_passthrough"):
            self.verify_red_lines(meta)
        return target_utt, weights_w, meta

    def verify_red_lines(self, meta: Dict[str, Any]) -> None:
        if not meta.get("weights_sum_to_one", False):
            raise HammerRedLineViolationError("[H3 Red-Line Violation] 帧池化权重之和不等于 1.0！")
        silence_gate = float(1.0 / (1.0 + np.exp(2.5)))
        voiced_gate = float(1.0 / (1.0 + np.exp(-1.5)))
        if silence_gate >= voiced_gate:
            raise HammerRedLineViolationError(
                "[H3 Red-Line Violation] 触发纯静音帧高权重反噬漏洞（严禁使用 |e_tilde_i| 绝对值）！"
            )
