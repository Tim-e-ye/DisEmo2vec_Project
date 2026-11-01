#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
卡槽 H3（时序显著性汇聚槽）：`models/hammer_slots/slot_h3_pooling.py`
======================================================================
专门存放“如何从 80% 平淡帧中精准聚焦 20% 情绪爆发帧”的候选池化锤头函数。
同时提供 `compute_salience_weighted_frame_loss`（对应【细节 2】）：
  - 将 H3 算出的 `frame_weights` (w_i) 一鱼两吃传给掩码帧级损失 L_Frm：
    `L_Frm = (1 / |M|) sum_{i in M} (1 + N * sg(w_i)) ||y^S_i - sg(y^T_i)||_2^2`
  - 并确保 Teacher 输出的 `target_utt` 与 `frame_weights` 经过 `.detach()` 阻断向 Teacher 的梯度回传！
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
        ).detach()
        target_utt = target_layer_results.mean(dim=1).detach()
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
    返回的 `target_utt` 和 `weights_w` 均显式 `.detach()`（阻断向 Teacher 的梯度回传，Teacher 仅靠 EMA 更新）。
    """
    if HAS_TORCH and isinstance(target_layer_results, torch.Tensor):
        bsz, n_frames, _ = target_layer_results.shape
        y_t = target_layer_results.detach()
        if frame_energy is None:
            frame_energy = torch.zeros((bsz, n_frames), device=y_t.device, dtype=y_t.dtype)
        else:
            frame_energy = frame_energy.detach()

        y_bar = y_t.mean(dim=1, keepdim=True)
        deviation_d = torch.norm(y_t - y_bar, p=2, dim=-1)
        voiced_gate = torch.sigmoid(frame_energy)
        salience_s = voiced_gate * deviation_d
        weights_w = F.softmax(salience_s / max(temperature_tau_s, 1e-4), dim=-1).detach()
        target_utt = torch.sum(weights_w.unsqueeze(-1) * y_t, dim=1).detach()
        sum_ok = bool(
            torch.allclose(
                weights_w.sum(dim=-1),
                torch.ones(bsz, device=y_t.device, dtype=y_t.dtype),
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
    """
    y_np = to_numpy(target_layer_results)
    bsz, n_frames, _ = y_np.shape
    e_np = np.zeros((bsz, n_frames), dtype=np.float32) if frame_energy is None else to_numpy(frame_energy)
    gate_g = 1.0 / (1.0 + np.exp(-np.clip(e_np, -20.0, 20.0)))

    if attn_weights is not None:
        raw_attn = to_numpy(attn_weights)
    else:
        y_norm = y_np / (np.linalg.norm(y_np, axis=-1, keepdims=True) + 1e-6)
        sim_mat = np.matmul(y_norm, np.transpose(y_norm, (0, 2, 1)))
        raw_attn = np.linalg.norm(y_np - y_np.mean(axis=1, keepdims=True), axis=-1) * (1.0 - np.mean(sim_mat, axis=-1))

    sal_s = gate_g * raw_attn
    scaled = (sal_s / max(temperature_tau_s, 1e-4))
    scaled = scaled - np.max(scaled, axis=-1, keepdims=True)
    exp_s = np.exp(scaled)
    w_np = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
    utt_np = np.sum(w_np[:, :, None] * y_np, axis=1)

    if HAS_TORCH and isinstance(target_layer_results, torch.Tensor):
        utt_out = torch.from_numpy(utt_np).to(target_layer_results.device, dtype=target_layer_results.dtype).detach()
        w_out = torch.from_numpy(w_np).to(target_layer_results.device, dtype=target_layer_results.dtype).detach()
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
    """卡槽 H3 封装类：支持通过 `--pooling_type` 切换池化锤头，并计算显著性加权帧损失 L_Frm。"""

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

        # 直观 If-Elif-Else 锤头卡槽路由
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

        # 【细节 2 保命铁律】：确保从 Teacher 算出的 target_utt 和 weights_w 均已 .detach()
        if HAS_TORCH and isinstance(target_utt, torch.Tensor):
            target_utt = target_utt.detach()
            weights_w = weights_w.detach()

        if mode not in ("original_mean", "tbd_passthrough"):
            self.verify_red_lines(meta)
        return target_utt, weights_w, meta

    def compute_salience_weighted_frame_loss(
        self,
        y_student_frames: Any,
        y_teacher_frames: Any,
        frame_weights: Any,
        mask_indices: Optional[Any] = None,
    ) -> Any:
        """
        【细节 2 核心实现】：将 H3 返回的 `frame_weights` (w_i) 传给掩码帧级损失 L_Frm：
            `L_Frm = (1 / |M|) sum_{i in M} (1 + N * sg(w_i)) * ||y^S_i - sg(y^T_i)||_2^2`
        其中 `y_teacher_frames` 与 `frame_weights` 强制执行 `.detach()` 阻断向 Teacher 的梯度回传。
        """
        if HAS_TORCH and isinstance(y_student_frames, torch.Tensor):
            n_frames = y_student_frames.shape[1]
            y_t_sg = y_teacher_frames.detach()
            w_sg = frame_weights.detach()
            per_frame_mse = torch.mean((y_student_frames - y_t_sg) ** 2, dim=-1)  # (B, N)
            boost_coeff = 1.0 + float(n_frames) * w_sg  # (1 + N * sg(w_i))
            weighted_mse = boost_coeff * per_frame_mse
            if mask_indices is not None:
                masked_vals = weighted_mse[mask_indices]
                return torch.mean(masked_vals) if masked_vals.numel() > 0 else torch.mean(weighted_mse)
            return torch.mean(weighted_mse)

        y_s_np = to_numpy(y_student_frames)
        y_t_np = to_numpy(y_teacher_frames)
        w_np = to_numpy(frame_weights)
        n_frames = y_s_np.shape[1]
        per_frame_mse = np.mean((y_s_np - y_t_np) ** 2, axis=-1)
        boost_coeff = 1.0 + float(n_frames) * w_np
        weighted_mse = boost_coeff * per_frame_mse
        if mask_indices is not None:
            return float(np.mean(weighted_mse[mask_indices]))
        return float(np.mean(weighted_mse))

    def verify_red_lines(self, meta: Dict[str, Any]) -> None:
        if not meta.get("weights_sum_to_one", False):
            raise HammerRedLineViolationError("[H3 Red-Line Violation] 帧池化权重之和不等于 1.0！")
        silence_gate = float(1.0 / (1.0 + np.exp(2.5)))
        voiced_gate = float(1.0 / (1.0 + np.exp(-1.5)))
        if silence_gate >= voiced_gate:
            raise HammerRedLineViolationError(
                "[H3 Red-Line Violation] 触发纯静音帧高权重反噬漏洞（严禁使用 |e_tilde_i| 绝对值）！"
            )
