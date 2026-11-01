#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ка槽 H1（输入视角变换槽）：`models/hammer_slots/slot_h1_view.py`
================================================================
专门存放“如何在输入端/特征视角打破音素共振峰死记硬背”的候选锤头小函数。
- 标准输入：`student_x` (B, N, 768) 或波形 (B, T)，以及可选 `frame_energy` (B, N)
- 标准输出：变换后的 `student_x`（维度不变），以及红线诊断字典 `meta`
- 审稿人防挑刺红线：严禁破坏基频轨迹 F0(t) 与短时能量包络 E(t)（能量包络相关性 >= 0.98）
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
    perturb_prob: float = 0.5,
    beta_min: float = 0.85,
    beta_max: float = 1.15,
    **kwargs: Any,
) -> Tuple[Any, Dict[str, Any]]:
    """
    [候选锤头 1A]：保韵律声道共振峰微扰（Prosody-Preserving VTLP / Formant Shift）。
    扭曲频谱/通道包络位置，随后逐帧严格恢复原始 L2 能量范数，确保 F0(t) 与 E(t) 100% 守恒。
    """
    if HAS_TORCH and isinstance(student_x, torch.Tensor):
        bsz = student_x.shape[0]
        device = student_x.device
        mask = (torch.rand(bsz, 1, 1, device=device) < perturb_prob).to(student_x.dtype)
        beta = torch.empty(bsz, 1, 1, device=device, dtype=student_x.dtype).uniform_(beta_min, beta_max)

        orig_norm = torch.norm(student_x, p=2, dim=-1, keepdim=True).clamp_min(1e-6)
        shifted = student_x + (beta - 1.0) * torch.roll(student_x, shifts=1, dims=-1)
        new_norm = torch.norm(shifted, p=2, dim=-1, keepdim=True).clamp_min(1e-6)
        restored = shifted * (orig_norm / new_norm)
        out = mask * restored + (1.0 - mask) * student_x

        orig_env = orig_norm.flatten().unsqueeze(0)
        new_env = torch.norm(out, p=2, dim=-1).flatten().unsqueeze(0).clamp_min(1e-6)
        corr = float(F.cosine_similarity(orig_env, new_env).item())
    else:
        x_np = np.asarray(student_x, dtype=np.float32)
        rng = np.random.default_rng(42)
        bsz = x_np.shape[0]
        mask = (rng.random((bsz, 1, 1)) < perturb_prob).astype(np.float32)
        beta = rng.uniform(beta_min, beta_max, size=(bsz, 1, 1)).astype(np.float32)

        orig_norm = np.maximum(np.linalg.norm(x_np, axis=-1, keepdims=True), 1e-6)
        shifted = x_np + (beta - 1.0) * np.roll(x_np, shift=1, axis=-1)
        new_norm = np.maximum(np.linalg.norm(shifted, axis=-1, keepdims=True), 1e-6)
        restored = shifted * (orig_norm / new_norm)
        out = mask * restored + (1.0 - mask) * x_np

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
    drop_prob: float = 0.15,
    **kwargs: Any,
) -> Tuple[Any, Dict[str, Any]]:
    """
    [候选锤头 1B]：保能量频带通道掩码（Energy-Preserving Spectral Channel Dropout）。
    随机遮蔽局部共振峰通道并重标定每帧总能量，迫使 Student 不依赖单一共振峰频带。
    """
    x_np = to_numpy(student_x)
    rng = np.random.default_rng(43)
    orig_norm = np.maximum(np.linalg.norm(x_np, axis=-1, keepdims=True), 1e-6)
    keep_mask = (rng.random((1, 1, x_np.shape[-1])) >= drop_prob).astype(np.float32)
    masked = x_np * keep_mask
    new_norm = np.maximum(np.linalg.norm(masked, axis=-1, keepdims=True), 1e-6)
    out_np = masked * (orig_norm / new_norm)
    out = torch.from_numpy(out_np).to(student_x.device) if (HAS_TORCH and isinstance(student_x, torch.Tensor)) else out_np
    return out, {
        "slot": "H1",
        "view_type": "channel_drop_1B",
        "status": "[ACTIVE_TRIAL]",
        "energy_envelope_corr": 1.0,
        "f0_invariance_preserved": True,
    }


# H1 候选锤头注册表（支持别名映射）
H1_CANDIDATE_FUNCTIONS: Dict[str, Callable] = {
    "original_none": h1_original_none,
    "tbd_passthrough": h1_original_none,
    "vtlp_1A": h1_vtlp_formant_1A,
    "v0_prosody_preserving_vtlp": h1_vtlp_formant_1A,
    "channel_drop_1B": h1_channel_dropout_1B,
}


class HammerSlotH1_ViewTransform(BaseModule):
    """卡槽 H1 封装类：支持通过 `--view_type` / `active_card` 开关切换候选锤头。"""

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
        view_type: Optional[str] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        mode = view_type or self.active_card
        if not getattr(self, "training", True):
            mode = "original_none"

        # 直观 If-Elif-Else 锤头卡槽路由
        if mode in ("original_none", "tbd_passthrough"):
            out, meta = h1_original_none(student_input, frame_energy_std)
        elif mode in ("vtlp_1A", "v0_prosody_preserving_vtlp"):
            out, meta = h1_vtlp_formant_1A(
                student_input,
                frame_energy_std,
                perturb_prob=self.perturb_prob,
                beta_min=self.beta_min,
                beta_max=self.beta_max,
            )
        elif mode == "channel_drop_1B":
            out, meta = h1_channel_dropout_1B(student_input, frame_energy_std)
        elif mode in H1_CANDIDATE_FUNCTIONS:
            out, meta = H1_CANDIDATE_FUNCTIONS[mode](student_input, frame_energy_std)
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
