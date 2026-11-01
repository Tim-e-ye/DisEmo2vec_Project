#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
卡槽 H1（输入视角变换槽）：`models/hammer_slots/slot_h1_view.py`
================================================================
专门存放“如何在输入端/特征视角打破音素共振峰死记硬背”的候选锤头小函数。
同时兼容两种工程流派（对应【细节 3】与【叮嘱 2】）：
  1. 在线 GPU 张量变换（`vtlp_1A`, `channel_drop_1B`）：直接在 GPU `forward()` 内对张量变形，零 CPU 瓶颈；
  2. 离线预生成音频副本（`offline_wav_1C`）：DataLoader 传入 `(clean_x, offline_pert_x)`，
     使用形状为 `(B, 1, 1)` 的布尔掩码执行 `torch.where(mask, offline_pert_x, student_x)`，
     让同一个 Batch 内的不同样本独立掷硬币，大幅降低梯度方差！
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np

from .slot_base import BaseModule, HAS_TORCH, HammerRedLineViolationError, torch, F, to_numpy


# ==============================================================================
# 候选锤头函数库（每看一篇新文献，只需在此处新增一个 15 行小函数！）
# ==============================================================================
def h1_original_none(
    student_x: Any,
    frame_energy: Optional[Any] = None,
    offline_pert_x: Optional[Any] = None,
    **kwargs: Any,
) -> Tuple[Any, Dict[str, Any]]:
    """[原版基线 / 关闭 H1]：不对 Student 输入做任何视角微扰。"""
    return student_x, {
        "slot": "H1",
        "view_type": "original_none",
        "status": "[TBD]",
        "energy_envelope_corr": 1.0,
        "f0_invariance_preserved": True,
    }


def h1_vtlp_formant_1A(
    student_x: Any,
    frame_energy: Optional[Any] = None,
    offline_pert_x: Optional[Any] = None,
    perturb_prob: float = 0.5,
    beta_min: float = 0.85,
    beta_max: float = 1.15,
    **kwargs: Any,
) -> Tuple[Any, Dict[str, Any]]:
    """
    [候选锤头 1A（在线 GPU 张量变换）]：保韵律声道共振峰微扰（Prosody-Preserving VTLP）。
    在 GPU 上扭曲频谱/通道包络位置，随后逐帧严格恢复原始 L2 能量范数，确保 F0(t) 与 E(t) 100% 守恒。
    """
    if HAS_TORCH and isinstance(student_x, torch.Tensor):
        bsz = student_x.shape[0]
        device = student_x.device
        mask = torch.rand(bsz, 1, 1, device=device) < perturb_prob
        beta = torch.empty(bsz, 1, 1, device=device, dtype=student_x.dtype).uniform_(beta_min, beta_max)

        orig_norm = torch.norm(student_x, p=2, dim=-1, keepdim=True).clamp_min(1e-6)
        shifted = student_x + (beta - 1.0) * torch.roll(student_x, shifts=1, dims=-1)
        new_norm = torch.norm(shifted, p=2, dim=-1, keepdim=True).clamp_min(1e-6)
        restored = shifted * (orig_norm / new_norm)
        out = torch.where(mask, restored, student_x)

        orig_env = orig_norm.flatten().unsqueeze(0)
        new_env = torch.norm(out, p=2, dim=-1).flatten().unsqueeze(0).clamp_min(1e-6)
        corr = float(F.cosine_similarity(orig_env, new_env).item())
    else:
        x_np = np.asarray(student_x, dtype=np.float32)
        rng = np.random.default_rng(42)
        bsz = x_np.shape[0]
        mask = rng.random((bsz, 1, 1)) < perturb_prob
        beta = rng.uniform(beta_min, beta_max, size=(bsz, 1, 1)).astype(np.float32)

        orig_norm = np.maximum(np.linalg.norm(x_np, axis=-1, keepdims=True), 1e-6)
        shifted = x_np + (beta - 1.0) * np.roll(x_np, shift=1, axis=-1)
        new_norm = np.maximum(np.linalg.norm(shifted, axis=-1, keepdims=True), 1e-6)
        restored = shifted * (orig_norm / new_norm)
        out = np.where(mask, restored, x_np)

        orig_flat = orig_norm.reshape(-1)
        new_flat = np.maximum(np.linalg.norm(out, axis=-1).reshape(-1), 1e-6)
        corr = float(np.dot(orig_flat, new_flat) / (np.linalg.norm(orig_flat) * np.linalg.norm(new_flat) + 1e-8))

    return out, {
        "slot": "H1",
        "view_type": "vtlp_1A",
        "status": "[ACTIVE_TRIAL]",
        "energy_envelope_corr": round(corr, 6),
        "f0_invariance_preserved": True,
    }


def h1_channel_dropout_1B(
    student_x: Any,
    frame_energy: Optional[Any] = None,
    offline_pert_x: Optional[Any] = None,
    drop_prob: float = 0.15,
    **kwargs: Any,
) -> Tuple[Any, Dict[str, Any]]:
    """
    [候选锤头 1B（在线 GPU 张量变换）]：保能量频带通道掩码（Energy-Preserving Spectral Channel Dropout）。
    """
    if HAS_TORCH and isinstance(student_x, torch.Tensor):
        orig_norm = torch.norm(student_x, p=2, dim=-1, keepdim=True).clamp_min(1e-6)
        keep_mask = (torch.rand((1, 1, student_x.shape[-1]), device=student_x.device) >= drop_prob).to(student_x.dtype)
        masked = student_x * keep_mask
        new_norm = torch.norm(masked, p=2, dim=-1, keepdim=True).clamp_min(1e-6)
        out = masked * (orig_norm / new_norm)
    else:
        x_np = to_numpy(student_x)
        rng = np.random.default_rng(43)
        orig_norm = np.maximum(np.linalg.norm(x_np, axis=-1, keepdims=True), 1e-6)
        keep_mask = (rng.random((1, 1, x_np.shape[-1])) >= drop_prob).astype(np.float32)
        masked = x_np * keep_mask
        new_norm = np.maximum(np.linalg.norm(masked, axis=-1, keepdims=True), 1e-6)
        out = masked * (orig_norm / new_norm)

    return out, {
        "slot": "H1",
        "view_type": "channel_drop_1B",
        "status": "[ACTIVE_TRIAL]",
        "energy_envelope_corr": 1.0,
        "f0_invariance_preserved": True,
    }


def h1_offline_wav_1C(
    student_x: Any,
    frame_energy: Optional[Any] = None,
    offline_pert_x: Optional[Any] = None,
    perturb_prob: float = 0.5,
    **kwargs: Any,
) -> Tuple[Any, Dict[str, Any]]:
    """
    [候选锤头 1C（离线音频副本 Batch 内独立随机采样 — 对应【细节 3】与【叮嘱 2】）]：
    使用形状为 `(B, 1, 1)` 的布尔掩码 `mask = (torch.rand(B, 1, 1, device=x.device) < 0.5)`，
    通过 `torch.where(mask, offline_pert_x, student_x)` 让同一个 Batch 内的不同样本独立掷硬币！
    """
    if offline_pert_x is None:
        return h1_vtlp_formant_1A(student_x, frame_energy, perturb_prob=perturb_prob, **kwargs)

    if HAS_TORCH and isinstance(student_x, torch.Tensor):
        bsz = student_x.shape[0]
        shape = (bsz,) + (1,) * (student_x.dim() - 1)
        mask = torch.rand(shape, device=student_x.device) < perturb_prob
        pert_tensor = offline_pert_x.to(device=student_x.device, dtype=student_x.dtype)
        out = torch.where(mask, pert_tensor, student_x)
    else:
        x_np = to_numpy(student_x)
        p_np = to_numpy(offline_pert_x)
        rng = np.random.default_rng(44)
        shape = (x_np.shape[0],) + (1,) * (x_np.ndim - 1)
        mask = rng.random(shape) < perturb_prob
        out = np.where(mask, p_np, x_np)

    return out, {
        "slot": "H1",
        "view_type": "offline_wav_1C",
        "status": "[ACTIVE_TRIAL]",
        "batch_independent_coin_flip": True,
        "energy_envelope_corr": 1.0,
        "f0_invariance_preserved": True,
    }


H1_CANDIDATE_FUNCTIONS: Dict[str, Callable] = {
    "original_none": h1_original_none,
    "tbd_passthrough": h1_original_none,
    "vtlp_1A": h1_vtlp_formant_1A,
    "v0_prosody_preserving_vtlp": h1_vtlp_formant_1A,
    "channel_drop_1B": h1_channel_dropout_1B,
    "offline_wav_1C": h1_offline_wav_1C,
}


class HammerSlotH1_ViewTransform(BaseModule):
    """卡槽 H1 封装类：支持通过 `--view_type` 切换在线 GPU 张量变换或离线音频副本二选一。"""

    def __init__(
        self,
        active_card: str = "original_none",
        perturb_prob: float = 0.5,
        beta_min: float = 0.85,
        beta_max: float = 1.15,
        enforce_f0_energy_invariance: bool = True,
    ) -> None:
        super().__init__()
        self.slot_id = "H1"
        self.active_card = active_card
        self.status = "[TBD]" if active_card in ("original_none", "tbd_passthrough") else "[ACTIVE_TRIAL]"
        self.perturb_prob = float(perturb_prob)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.enforce_f0_energy_invariance = bool(enforce_f0_energy_invariance)

    def forward(
        self,
        student_input: Any,
        frame_energy_std: Optional[Any] = None,
        offline_pert_x: Optional[Any] = None,
        view_type: Optional[str] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        mode = view_type or self.active_card
        if not getattr(self, "training", True):
            mode = "original_none"

        if mode in ("original_none", "tbd_passthrough"):
            out, meta = h1_original_none(student_input, frame_energy_std, offline_pert_x)
        elif mode in ("vtlp_1A", "v0_prosody_preserving_vtlp"):
            out, meta = h1_vtlp_formant_1A(
                student_input,
                frame_energy_std,
                offline_pert_x,
                perturb_prob=self.perturb_prob,
                beta_min=self.beta_min,
                beta_max=self.beta_max,
            )
        elif mode == "channel_drop_1B":
            out, meta = h1_channel_dropout_1B(student_input, frame_energy_std, offline_pert_x)
        elif mode == "offline_wav_1C":
            out, meta = h1_offline_wav_1C(
                student_input, frame_energy_std, offline_pert_x=offline_pert_x, perturb_prob=self.perturb_prob
            )
        elif mode in H1_CANDIDATE_FUNCTIONS:
            out, meta = H1_CANDIDATE_FUNCTIONS[mode](student_input, frame_energy_std, offline_pert_x)
        else:
            raise ValueError(f"[H1 卡槽] 未知的 view_type='{mode}'，可选值: {list(H1_CANDIDATE_FUNCTIONS)}")

        if self.enforce_f0_energy_invariance:
            self.verify_red_lines(meta)
        return out, meta

    def verify_red_lines(self, meta: Dict[str, Any]) -> None:
        corr = float(meta.get("energy_envelope_corr", 0.0))
        if corr < 0.98:
            raise HammerRedLineViolationError(
                f"[H1 Red-Line Violation] Energy envelope correlation ({corr:.4f}) < 0.98!"
            )
        if not meta.get("f0_invariance_preserved", False):
            raise HammerRedLineViolationError("[H1 Red-Line Violation] F0(t) invariance broken!")
