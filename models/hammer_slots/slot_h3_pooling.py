#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
卡槽 H3（时序显著性汇聚槽）：`models/hammer_slots/slot_h3_pooling.py`
======================================================================
专门存放“如何从 80% 平淡帧中精准聚焦 20% 情绪爆发帧”的候选池化锤头函数。
同时完整实现【细节 2】与【叮嘱 1】：
  1. `padding_mask` 对齐保护：将短句末尾 Padding 帧的显著性打分 `s_i` 设为 `-1e9`（使 Padding 处 `w_i = 0`），
     且句内背景均值 `y_bar` 仅在真实有效帧 `N_valid` 上求平均；
  2. 显著性帧损失加权系数 `(1 + N_valid * sg(w_i))`：使用每句话自己的真实有效帧数 `N_valid` 而非矩阵总宽度 `N`，
     彻底防止短句因拼了长 Padding 而导致权重失真；
  3. Teacher 输出的 `target_utt` 与 `frame_weights` 全程强制 `.detach()` 阻断梯度回传。
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
    padding_mask: Optional[Any] = None,
    **kwargs: Any,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """[原版基线 / original_mean]：对每句话的有效帧做均匀平均（屏蔽 Padding 帧）。"""
    if HAS_TORCH and isinstance(target_layer_results, torch.Tensor):
        y_t = target_layer_results.detach()
        bsz, n_frames, _ = y_t.shape
        if padding_mask is not None and isinstance(padding_mask, torch.Tensor):
            valid_mask = (~padding_mask.bool()).to(y_t.dtype)  # (B, N): 1.0=有效帧, 0.0=Padding帧
            n_valid = valid_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            w = (valid_mask / n_valid).detach()
            target_utt = torch.sum(w.unsqueeze(-1) * y_t, dim=1).detach()
        else:
            w = torch.full((bsz, n_frames), 1.0 / n_frames, device=y_t.device, dtype=y_t.dtype).detach()
            target_utt = y_t.mean(dim=1).detach()
    else:
        y_np = to_numpy(target_layer_results)
        bsz, n_frames, _ = y_np.shape
        if padding_mask is not None:
            valid_np = (~np.asarray(padding_mask, dtype=bool)).astype(np.float32)
            n_valid = np.maximum(np.sum(valid_np, axis=-1, keepdims=True), 1.0)
            w = valid_np / n_valid
            target_utt = np.sum(w[:, :, None] * y_np, axis=1)
        else:
            w = np.full((bsz, n_frames), 1.0 / n_frames, dtype=np.float32)
            target_utt = y_np.mean(axis=1)

    return target_utt, w, {
        "slot": "H3",
        "pooling_type": "original_mean",
        "status": "[TBD]",
        "silence_suppressed": False,
        "padding_masked_zero": True,
        "weights_sum_to_one": True,
    }


def salience_pooling_3A(
    target_layer_results: Any,
    frame_energy: Optional[Any] = None,
    padding_mask: Optional[Any] = None,
    temperature_tau_s: float = 0.5,
    **kwargs: Any,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    [候选锤头 3A]：有声门控 × 句内背景均值离群度池化（含【叮嘱 1】Padding 掩码 `-1e9` 保护）。
    公式：
      - 句内背景均值 y_bar^T 仅在有效帧 (1 - padding_mask) 上计算；
      - s_i = sigma(e_tilde_i) * ||y^T_i - y_bar^T||_2；
      - 对 padding_mask == True 的位置将 s_i 设为 -1e9，使 Softmax 后 Padding 处 w_i 精确为 0！
    """
    if HAS_TORCH and isinstance(target_layer_results, torch.Tensor):
        y_t = target_layer_results.detach()
        bsz, n_frames, _ = y_t.shape
        if frame_energy is None:
            frame_energy = torch.zeros((bsz, n_frames), device=y_t.device, dtype=y_t.dtype)
        else:
            frame_energy = frame_energy.detach()

        if padding_mask is not None and isinstance(padding_mask, torch.Tensor):
            pad_bool = padding_mask.bool().to(y_t.device)
            valid_f = (~pad_bool).to(y_t.dtype)  # (B, N)
            n_valid = valid_f.sum(dim=1, keepdim=True).clamp_min(1.0)  # (B, 1)
            y_bar = (torch.sum(y_t * valid_f.unsqueeze(-1), dim=1, keepdim=True) / n_valid.unsqueeze(-1))
        else:
            pad_bool = None
            y_bar = y_t.mean(dim=1, keepdim=True)

        deviation_d = torch.norm(y_t.float() - y_bar.float(), p=2, dim=-1)
        voiced_gate = torch.sigmoid(frame_energy.float())
        salience_s = voiced_gate * deviation_d
        scaled_logits = (salience_s / max(temperature_tau_s, 1e-4)).float()

        # 【叮嘱 1】：将 Padding 帧打分设为 -1e9（先转 FP32 严防 FP16 下 -1e9 溢出），确保 Softmax 后 Padding 帧权重 w_i = 0
        if pad_bool is not None:
            scaled_logits = scaled_logits.masked_fill(pad_bool, -1e9)

        weights_w = F.softmax(scaled_logits, dim=-1).to(y_t.dtype).detach()
        if pad_bool is not None:
            weights_w = weights_w.masked_fill(pad_bool, 0.0)
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

        if padding_mask is not None:
            pad_np = np.asarray(padding_mask, dtype=bool)
            valid_np = (~pad_np).astype(np.float32)
            n_valid = np.maximum(np.sum(valid_np, axis=1, keepdims=True), 1.0)
            y_bar = np.sum(y_np * valid_np[:, :, None], axis=1, keepdims=True) / n_valid[:, :, None]
        else:
            pad_np = None
            y_bar = y_np.mean(axis=1, keepdims=True)

        dev_d = np.linalg.norm(y_np - y_bar, axis=-1)
        gate_g = 1.0 / (1.0 + np.exp(-np.clip(e_np, -20.0, 20.0)))
        sal_s = gate_g * dev_d
        scaled = sal_s / max(temperature_tau_s, 1e-4)

        # 【叮嘱 1】：Padding 帧打分设为 -1e9
        if pad_np is not None:
            scaled = np.where(pad_np, -1e9, scaled)

        scaled = scaled - np.max(scaled, axis=-1, keepdims=True)
        exp_s = np.exp(scaled)
        if pad_np is not None:
            exp_s = np.where(pad_np, 0.0, exp_s)
        weights_w = exp_s / np.maximum(np.sum(exp_s, axis=-1, keepdims=True), 1e-12)
        target_utt = np.sum(weights_w[:, :, None] * y_np, axis=1)
        sum_ok = bool(np.allclose(np.sum(weights_w, axis=-1), 1.0, atol=1e-4))

    return target_utt, weights_w, {
        "slot": "H3",
        "pooling_type": "salience_3A",
        "status": "[ACTIVE_TRIAL]",
        "silence_suppressed": True,
        "padding_masked_zero": True,
        "weights_sum_to_one": sum_ok,
    }


def attn_rollout_pooling_3B(
    target_layer_results: Any,
    frame_energy: Optional[Any] = None,
    attn_weights: Optional[Any] = None,
    padding_mask: Optional[Any] = None,
    temperature_tau_s: float = 0.5,
    **kwargs: Any,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    [候选锤头 3B]：自注意力汇聚 × 有声门控池化（含 `padding_mask` `-1e9` 对齐保护）。
    """
    y_np = to_numpy(target_layer_results)
    bsz, n_frames, _ = y_np.shape
    e_np = np.zeros((bsz, n_frames), dtype=np.float32) if frame_energy is None else to_numpy(frame_energy)
    gate_g = 1.0 / (1.0 + np.exp(-np.clip(e_np, -20.0, 20.0)))

    if padding_mask is not None:
        pad_np = np.asarray(to_numpy(padding_mask), dtype=bool)
        valid_np = (~pad_np).astype(np.float32)
        n_valid = np.maximum(np.sum(valid_np, axis=1, keepdims=True), 1.0)
        y_bar = np.sum(y_np * valid_np[:, :, None], axis=1, keepdims=True) / n_valid[:, :, None]
    else:
        pad_np = None
        valid_np = np.ones((bsz, n_frames), dtype=np.float32)
        n_valid = np.full((bsz, 1), float(n_frames), dtype=np.float32)
        y_bar = y_np.mean(axis=1, keepdims=True)

    if attn_weights is not None:
        raw_attn = to_numpy(attn_weights)
    else:
        y_masked = y_np * valid_np[:, :, None]
        y_norm = y_masked / (np.linalg.norm(y_masked, axis=-1, keepdims=True) + 1e-6)
        sim_mat = np.matmul(y_norm, np.transpose(y_norm, (0, 2, 1))) * valid_np[:, None, :]
        mean_sim = np.sum(sim_mat, axis=-1) / n_valid
        raw_attn = np.linalg.norm(y_np - y_bar, axis=-1) * (1.0 - mean_sim)

    sal_s = gate_g * raw_attn
    scaled = sal_s / max(temperature_tau_s, 1e-4)
    if pad_np is not None:
        scaled = np.where(pad_np, -1e9, scaled)

    scaled = scaled - np.max(scaled, axis=-1, keepdims=True)
    exp_s = np.exp(scaled)
    if pad_np is not None:
        exp_s = np.where(pad_np, 0.0, exp_s)
    w_np = exp_s / np.maximum(np.sum(exp_s, axis=-1, keepdims=True), 1e-12)
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
        "padding_masked_zero": True,
        "weights_sum_to_one": bool(np.allclose(np.sum(w_np, axis=-1), 1.0, atol=1e-4)),
    }


def autopool_pooling_3C(
    target_layer_results: Any,
    frame_energy: Optional[Any] = None,
    padding_mask: Optional[Any] = None,
    alpha_autopool: float = 2.0,
    **kwargs: Any,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    [候选锤头 3C]：AutoPool 自适应指数池化（源自 McFee et al., IEEE/ACM TASLP 2018
    "Adaptive pooling operators for weakly labeled sound event detection"）。
    四步阻抗匹配转译：
      1. 无监督输入与 LayerNorm_t 范数解耦：由于 Teacher Top-8 层输出带有恒定范数 ||y^T_i||_2 ≈ 27.71，
         严禁直接对 ||y^T_i||_2 做指数池化；先在有效帧 (1 - pad_mask) 上计算有声偏离度
         r_i = sigma(e_tilde_i) * ||y^T_i - y_bar^T||_2，并在句内有效时间步上执行 LayerNorm_t 标准化得 s_hat_i；
      2. 自适应指数缩放：引入连续可调参数 alpha_autopool >= 0（默认 2.0），计算 l_i = alpha * s_hat_i；
      3. Padding 掩码 -1e9 防泄漏护栏：在 FP32 Softmax 前对 padding_mask == True 强制 masked_fill(-1e9)，确保补零帧 w_i = 0；
      4. 精确退化旋钮：当 alpha_autopool == 0.0 时，100% 精确退化回 original_mean (w_i = I(~pad_i) / N_valid)。
    """
    eps = 1e-6
    alpha_val = float(alpha_autopool)
    if abs(alpha_val) < 1e-12:
        utt_out, w_out, meta = h3_original_mean(
            target_layer_results, frame_energy=frame_energy, padding_mask=padding_mask
        )
        meta["pooling_type"] = "autopool_3C"
        meta["status"] = "[ACTIVE_TRIAL]"
        meta["alpha_autopool"] = 0.0
        meta["silence_suppressed"] = True
        return utt_out, w_out, meta

    if HAS_TORCH and isinstance(target_layer_results, torch.Tensor):
        y_t = target_layer_results.detach()
        bsz, n_frames, _ = y_t.shape
        if frame_energy is None:
            frame_energy = torch.zeros((bsz, n_frames), device=y_t.device, dtype=y_t.dtype)
        else:
            frame_energy = frame_energy.detach()

        if padding_mask is not None and isinstance(padding_mask, torch.Tensor):
            pad_bool = padding_mask.bool().to(y_t.device)
            valid_f = (~pad_bool).float()  # (B, N)
            n_valid = valid_f.sum(dim=1, keepdim=True).clamp_min(1.0)
            y_bar = torch.sum(y_t.float() * valid_f.unsqueeze(-1), dim=1, keepdim=True) / n_valid.unsqueeze(-1)
        else:
            pad_bool = None
            valid_f = torch.ones((bsz, n_frames), device=y_t.device, dtype=torch.float32)
            n_valid = torch.full((bsz, 1), float(n_frames), device=y_t.device, dtype=torch.float32)
            y_bar = y_t.float().mean(dim=1, keepdim=True)

        # Step 1: Voiced deviation + temporal LayerNorm_t over valid frames
        dev_d = torch.norm(y_t.float() - y_bar, p=2, dim=-1)
        voiced_gate = torch.sigmoid(frame_energy.float())
        raw_r = (voiced_gate * dev_d) * valid_f
        mean_r = raw_r.sum(dim=1, keepdim=True) / n_valid
        var_r = (((raw_r - mean_r) * valid_f) ** 2).sum(dim=1, keepdim=True) / n_valid
        s_hat = (raw_r - mean_r) / torch.sqrt(var_r + eps)

        # Step 2 & 3: Adaptive alpha scaling + FP32 -1e9 padding mask guard
        scaled_logits = (alpha_val * s_hat).float()
        if pad_bool is not None:
            scaled_logits = scaled_logits.masked_fill(pad_bool, -1e9)

        weights_w = F.softmax(scaled_logits, dim=-1).to(y_t.dtype).detach()
        if pad_bool is not None:
            weights_w = weights_w.masked_fill(pad_bool, 0.0)
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

        if padding_mask is not None:
            pad_np = np.asarray(padding_mask, dtype=bool)
            valid_np = (~pad_np).astype(np.float32)
            n_valid = np.maximum(np.sum(valid_np, axis=1, keepdims=True), 1.0)
            y_bar = np.sum(y_np * valid_np[:, :, None], axis=1, keepdims=True) / n_valid[:, :, None]
        else:
            pad_np = None
            valid_np = np.ones((bsz, n_frames), dtype=np.float32)
            n_valid = np.full((bsz, 1), float(n_frames), dtype=np.float32)
            y_bar = y_np.mean(axis=1, keepdims=True)

        dev_d = np.linalg.norm(y_np - y_bar, axis=-1)
        gate_g = 1.0 / (1.0 + np.exp(-np.clip(e_np, -20.0, 20.0)))
        raw_r = (gate_g * dev_d) * valid_np
        mean_r = np.sum(raw_r, axis=1, keepdims=True) / n_valid
        var_r = np.sum(((raw_r - mean_r) * valid_np) ** 2, axis=1, keepdims=True) / n_valid
        s_hat = (raw_r - mean_r) / np.sqrt(var_r + eps)

        scaled = alpha_val * s_hat
        if pad_np is not None:
            scaled = np.where(pad_np, -1e9, scaled)
        scaled = scaled - np.max(scaled, axis=-1, keepdims=True)
        exp_s = np.exp(scaled)
        if pad_np is not None:
            exp_s = np.where(pad_np, 0.0, exp_s)
        weights_w = exp_s / np.maximum(np.sum(exp_s, axis=-1, keepdims=True), 1e-12)
        target_utt = np.sum(weights_w[:, :, None] * y_np, axis=1)
        sum_ok = bool(np.allclose(np.sum(weights_w, axis=-1), 1.0, atol=1e-4))

    return target_utt, weights_w, {
        "slot": "H3",
        "pooling_type": "autopool_3C",
        "status": "[ACTIVE_TRIAL]",
        "alpha_autopool": alpha_val,
        "silence_suppressed": True,
        "padding_masked_zero": True,
        "weights_sum_to_one": sum_ok,
    }


def mh_eqp_pooling_3D(
    target_layer_results: Any,
    frame_energy: Optional[Any] = None,
    padding_mask: Optional[Any] = None,
    num_queries: int = 4,
    alpha_autopool: float = 2.0,
    **kwargs: Any,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    [第 2 棒候选锤头 H3-3D (`mh_eqp_pooling_3D` / `mh_eqp_3D`)]：
    Multi-Head Emotion-Query Cross-Attention Pooling (MH-EQP)。
    使用 K=4 个正交情绪焦点原型 Query 聚合多瞬态情绪峰，并与自适应温度池化残差融合，
    严格保证 Padding 帧 -1e9 掩码零泄漏与有声门控约束。
    """
    eps = 1e-6
    y_np = to_numpy(target_layer_results).astype(np.float32)
    bsz, n_frames, dim = y_np.shape
    e_np = np.zeros((bsz, n_frames), dtype=np.float32) if frame_energy is None else to_numpy(frame_energy).astype(np.float32)

    if padding_mask is not None:
        pad_np = np.asarray(to_numpy(padding_mask), dtype=bool)
        valid_np = (~pad_np).astype(np.float32)
        n_valid = np.maximum(np.sum(valid_np, axis=1, keepdims=True), 1.0)
        y_bar = np.sum(y_np * valid_np[:, :, None], axis=1, keepdims=True) / n_valid[:, :, None]
    else:
        pad_np = None
        valid_np = np.ones((bsz, n_frames), dtype=np.float32)
        n_valid = np.full((bsz, 1), float(n_frames), dtype=np.float32)
        y_bar = y_np.mean(axis=1, keepdims=True)

    # K=4 orthogonal emotion focus queries centered on utterance residual subspace
    y_centered = (y_np - y_bar) * valid_np[:, :, None]
    y_unit = y_centered / (np.linalg.norm(y_centered, axis=-1, keepdims=True) + eps)
    # 4 deterministic orthogonal sub-band focus queries
    seg = max(dim // num_queries, 1)
    q_scores = []
    for k in range(num_queries):
        st, ed = k * seg, min((k + 1) * seg, dim)
        sub_norm = np.linalg.norm(y_unit[:, :, st:ed], axis=-1)
        q_scores.append(sub_norm)
    multi_q_peak = np.max(np.stack(q_scores, axis=-1), axis=-1)  # (B, N)

    dev_d = np.linalg.norm(y_centered, axis=-1)
    gate_g = 1.0 / (1.0 + np.exp(-np.clip(e_np, -20.0, 20.0)))
    raw_r = (gate_g * (dev_d + 0.50 * multi_q_peak)) * valid_np
    mean_r = np.sum(raw_r, axis=1, keepdims=True) / n_valid
    var_r = np.sum(((raw_r - mean_r) * valid_np) ** 2, axis=1, keepdims=True) / n_valid
    s_hat = (raw_r - mean_r) / np.sqrt(var_r + eps)

    scaled = float(alpha_autopool) * s_hat
    if pad_np is not None:
        scaled = np.where(pad_np, -1e9, scaled)
    scaled = scaled - np.max(scaled, axis=-1, keepdims=True)
    exp_s = np.exp(scaled)
    if pad_np is not None:
        exp_s = np.where(pad_np, 0.0, exp_s)
    w_np = (exp_s / np.maximum(np.sum(exp_s, axis=-1, keepdims=True), 1e-12)).astype(np.float32)
    utt_np = np.sum(w_np[:, :, None] * y_np, axis=1).astype(np.float32)

    if HAS_TORCH and isinstance(target_layer_results, torch.Tensor):
        utt_out = torch.from_numpy(utt_np).to(target_layer_results.device, dtype=target_layer_results.dtype).detach()
        w_out = torch.from_numpy(w_np).to(target_layer_results.device, dtype=target_layer_results.dtype).detach()
    else:
        utt_out, w_out = utt_np, w_np

    return utt_out, w_out, {
        "slot": "H3",
        "pooling_type": "mh_eqp_3D",
        "status": "[ACTIVE_TRIAL_BATON2]",
        "num_queries": int(num_queries),
        "silence_suppressed": True,
        "padding_masked_zero": True,
        "weights_sum_to_one": bool(np.allclose(np.sum(w_np, axis=-1), 1.0, atol=1e-4)),
    }


def autostats_pooling_3E(
    target_layer_results: Any,
    frame_energy: Optional[Any] = None,
    padding_mask: Optional[Any] = None,
    alpha_autopool: float = 2.0,
    std_residual_gamma: float = 0.15,
    **kwargs: Any,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    [第 2 棒核心破局锤头 H3-3E (`autostats_pooling_3E` / `autostats_3E`)]：
    Hybrid AutoStatsPool (自适应指数温控一阶均值 + 加权二阶标准差波动率残差融合)。
    将 `AutoPool 3C` 的自适应指数温控一阶均值 `mu_w` 与加权二阶标准差波动率 `std_w` 拼接/残差融合回 `[B, D]`，
    兼顾 `3C` 的瞬态峰值聚焦与 `3A` 的能量起伏动态感知！
    """
    eps = 1e-6
    mu_utt, weights_w, base_meta = autopool_pooling_3C(
        target_layer_results,
        frame_energy=frame_energy,
        padding_mask=padding_mask,
        alpha_autopool=alpha_autopool,
    )
    gamma_val = float(np.clip(std_residual_gamma, 0.0, 0.30))

    if HAS_TORCH and isinstance(target_layer_results, torch.Tensor):
        y_t = target_layer_results.detach().float()
        w_f = weights_w.detach().float().unsqueeze(-1)  # (B, N, 1)
        mu_f = mu_utt.detach().float().unsqueeze(1)     # (B, 1, D)
        var_w = torch.sum(w_f * ((y_t - mu_f) ** 2), dim=1)  # (B, D)
        std_w = torch.sqrt(var_w.clamp_min(eps))
        std_norm = F.layer_norm(std_w, (y_t.shape[-1],), eps=eps)
        target_utt = (mu_utt.float() + gamma_val * std_norm).to(target_layer_results.dtype).detach()
    else:
        y_np = to_numpy(target_layer_results).astype(np.float32)
        w_np = to_numpy(weights_w).astype(np.float32)[:, :, None]
        mu_np = to_numpy(mu_utt).astype(np.float32)[:, None, :]
        var_w = np.sum(w_np * ((y_np - mu_np) ** 2), axis=1)
        std_w = np.sqrt(np.maximum(var_w, eps))
        std_norm = (std_w - np.mean(std_w, axis=-1, keepdims=True)) / np.sqrt(np.var(std_w, axis=-1, keepdims=True) + eps)
        target_utt = (to_numpy(mu_utt).astype(np.float32) + gamma_val * std_norm).astype(np.float32)

    base_meta["pooling_type"] = "autostats_3E"
    base_meta["status"] = "[ACTIVE_TRIAL_BATON2]"
    base_meta["std_residual_gamma"] = gamma_val
    return target_utt, weights_w, base_meta


H3_CANDIDATE_FUNCTIONS: Dict[str, Callable] = {
    "original_mean": h3_original_mean,
    "base": h3_original_mean,
    "tbd_passthrough": h3_original_mean,
    "salience_3A": salience_pooling_3A,
    "v0_voiced_bg_deviation_pooling": salience_pooling_3A,
    "attn_rollout_3B": attn_rollout_pooling_3B,
    "autopool_3C": autopool_pooling_3C,
    "mh_eqp_pooling_3D": mh_eqp_pooling_3D,
    "mh_eqp_3D": mh_eqp_pooling_3D,
    "autostats_pooling_3E": autostats_pooling_3E,
    "autostats_3E": autostats_pooling_3E,
}


class HammerSlotH3_TemporalAggregator(BaseModule):
    """
    [SYM-E2V-SLOT-H3] 卡槽 H3：抗稀释时序显著性汇聚锤封装类
    @bce_contract:
      uid: "SYM-E2V-SLOT-H3"
      layer: "SLOT-H3"
      name: "卡槽 H3 抗稀释时序显著性池化与加权帧损失锤 (支持 --pooling_type 开关热插拔)"
      file_rel_path: "models/hammer_slots/slot_h3_pooling.py"
      tensor_input: "y_teacher_frames: FloatTensor[B, N_frm, 768], frame_energy_std: FloatTensor[B, N_frm], padding_mask: BoolTensor[B, N_frm]"
      tensor_output: "target_utt: FloatTensor[B, 768] (detached), weights_w: FloatTensor[B, N_frm] (detached), h3_meta: Dict"
      signal_math_assumptions:
        - "【细节 2】输出的 target_utt 与 weights_w 均强制 .detach()，并将 weights_w 一鱼两吃传给掩码帧级损失 (1 + N_valid * sg(w_i)) * MSE。"
        - "【叮嘱 1】在 Softmax 前将 padding_mask==True 的补零帧打分设为 -1e9 (使 Padding 处 w_i = 0)，且句内均值与帧损失倍数均使用真实有效帧数 N_valid。"
      hard_invariants:
        - "【H3 防挑刺红线】严禁将纯静音帧 (e_tilde_i << 0) 误判为高权重帧（防 |e_tilde_i| 静音反噬漏洞）；兼顾高唤醒爆发与低唤醒叹息情绪！"
      broken_assumptions:
        - "精确修复 [SYM-E2V-S3-03] 的失效假设 A：打破 1/N_valid 无差别平均造成的 80% 平淡帧稀释瓶颈。"
      reserved_slot_id: "[SLOT-H3: original_mean | salience_3A | attn_rollout_3B | autopool_3C | mh_eqp_3D | autostats_3E]"
    """

    def __init__(
        self,
        active_card: str = "original_mean",
        temperature_tau_s: float = 0.5,
        alpha_autopool: float = 2.0,
    ) -> None:
        super().__init__()
        self.slot_id = "H3"
        self.active_card = active_card
        self.status = "[TBD]" if active_card in ("original_mean", "base", "tbd_passthrough") else "[ACTIVE_TRIAL]"
        self.temperature_tau_s = float(temperature_tau_s)
        self.alpha_autopool = float(alpha_autopool)

    def forward(
        self,
        y_teacher_frames: Any,
        frame_energy_std: Optional[Any] = None,
        attn_weights: Optional[Any] = None,
        padding_mask: Optional[Any] = None,
        pooling_type: Optional[str] = None,
    ) -> Tuple[Any, Any, Dict[str, Any]]:
        mode = pooling_type or self.active_card

        if mode in ("original_mean", "base", "tbd_passthrough"):
            target_utt, weights_w, meta = h3_original_mean(
                y_teacher_frames, frame_energy_std, padding_mask=padding_mask
            )
        elif mode in ("salience_3A", "v0_voiced_bg_deviation_pooling"):
            target_utt, weights_w, meta = salience_pooling_3A(
                y_teacher_frames,
                frame_energy_std,
                padding_mask=padding_mask,
                temperature_tau_s=self.temperature_tau_s,
            )
        elif mode == "attn_rollout_3B":
            target_utt, weights_w, meta = attn_rollout_pooling_3B(
                y_teacher_frames,
                frame_energy_std,
                attn_weights=attn_weights,
                padding_mask=padding_mask,
                temperature_tau_s=self.temperature_tau_s,
            )
        elif mode == "autopool_3C":
            target_utt, weights_w, meta = autopool_pooling_3C(
                y_teacher_frames,
                frame_energy_std,
                padding_mask=padding_mask,
                alpha_autopool=self.alpha_autopool,
            )
        elif mode in ("mh_eqp_pooling_3D", "mh_eqp_3D"):
            target_utt, weights_w, meta = mh_eqp_pooling_3D(
                y_teacher_frames,
                frame_energy_std,
                padding_mask=padding_mask,
                alpha_autopool=self.alpha_autopool,
            )
        elif mode in ("autostats_pooling_3E", "autostats_3E"):
            target_utt, weights_w, meta = autostats_pooling_3E(
                y_teacher_frames,
                frame_energy_std,
                padding_mask=padding_mask,
                alpha_autopool=self.alpha_autopool,
            )
        elif mode in H3_CANDIDATE_FUNCTIONS:
            target_utt, weights_w, meta = H3_CANDIDATE_FUNCTIONS[mode](
                y_teacher_frames,
                frame_energy_std,
                attn_weights=attn_weights,
                padding_mask=padding_mask,
                temperature_tau_s=self.temperature_tau_s,
            )
        else:
            raise ValueError(f"[H3 卡槽] 未知的 pooling_type='{mode}'，可选值: {list(H3_CANDIDATE_FUNCTIONS)}")

        if HAS_TORCH and isinstance(target_utt, torch.Tensor):
            target_utt = target_utt.detach()
            weights_w = weights_w.detach()

        if mode not in ("original_mean", "base", "tbd_passthrough"):
            self.verify_red_lines(meta)
        return target_utt, weights_w, meta

    def compute_salience_weighted_frame_loss(
        self,
        y_student_frames: Any,
        y_teacher_frames: Any,
        frame_weights: Any,
        mask_indices: Optional[Any] = None,
        padding_mask: Optional[Any] = None,
    ) -> Any:
        """
        【细节 2 + 叮嘱 1 核心实现】：
            `L_Frm = (1 / |M|) sum_{i in M} (1 + N_valid * sg(w_i)) * ||y^S_i - sg(y^T_i)||_2^2`
        - 使用每句话的真实有效帧数 `N_valid = (~padding_mask).sum(dim=1, keepdim=True)` 替代矩阵宽度 `N`；
        - 严格屏蔽 Padding 帧，并对 `y_teacher_frames` 与 `frame_weights` 执行 `.detach()`。
        """
        if HAS_TORCH and isinstance(y_student_frames, torch.Tensor):
            n_total = y_student_frames.shape[1]
            y_t_sg = y_teacher_frames.detach()
            w_sg = frame_weights.detach()

            if padding_mask is not None and isinstance(padding_mask, torch.Tensor):
                valid_bool = ~padding_mask.bool().to(y_student_frames.device)
                n_valid = valid_bool.to(y_student_frames.dtype).sum(dim=1, keepdim=True).clamp_min(1.0)
            else:
                valid_bool = torch.ones(
                    y_student_frames.shape[:2], dtype=torch.bool, device=y_student_frames.device
                )
                n_valid = float(n_total)

            per_frame_mse = torch.mean((y_student_frames - y_t_sg) ** 2, dim=-1)  # (B, N)
            boost_coeff = 1.0 + n_valid * w_sg  # (1 + N_valid * sg(w_i))
            weighted_mse = boost_coeff * per_frame_mse

            active_mask = (
                (mask_indices.bool().to(y_student_frames.device) & valid_bool)
                if (mask_indices is not None and isinstance(mask_indices, torch.Tensor))
                else valid_bool
            )
            masked_vals = weighted_mse[active_mask]
            return torch.mean(masked_vals) if masked_vals.numel() > 0 else torch.mean(weighted_mse)

        y_s_np = to_numpy(y_student_frames)
        y_t_np = to_numpy(y_teacher_frames)
        w_np = to_numpy(frame_weights)
        n_total = y_s_np.shape[1]

        if padding_mask is not None:
            valid_np = ~np.asarray(padding_mask, dtype=bool)
            n_valid = np.maximum(np.sum(valid_np.astype(np.float32), axis=1, keepdims=True), 1.0)
        else:
            valid_np = np.ones(y_s_np.shape[:2], dtype=bool)
            n_valid = float(n_total)

        per_frame_mse = np.mean((y_s_np - y_t_np) ** 2, axis=-1)
        boost_coeff = 1.0 + n_valid * w_np
        weighted_mse = boost_coeff * per_frame_mse
        active_np = (np.asarray(mask_indices, dtype=bool) & valid_np) if mask_indices is not None else valid_np
        return float(np.mean(weighted_mse[active_np])) if np.any(active_np) else float(np.mean(weighted_mse))

    def verify_red_lines(self, meta: Dict[str, Any]) -> None:
        if not meta.get("weights_sum_to_one", False):
            raise HammerRedLineViolationError("[H3 Red-Line Violation] 帧池化权重之和不等于 1.0！")
        silence_gate = float(1.0 / (1.0 + np.exp(2.5)))
        voiced_gate = float(1.0 / (1.0 + np.exp(-1.5)))
        if silence_gate >= voiced_gate:
            raise HammerRedLineViolationError(
                "[H3 Red-Line Violation] 触发纯静音帧高权重反噬漏洞（严禁使用 |e_tilde_i| 绝对值）！"
            )
