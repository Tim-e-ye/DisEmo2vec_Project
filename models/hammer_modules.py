#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
`models/hammer_modules.py` —— 锤头卡槽总控中心与模型 `forward()` 开关调用层（V1.3）
==================================================================================
完整融入 4 卡 A10 训练零报错、不断梯度的【三大工程防坑细节 + 两大微观张量保护叮嘱】：

- 【细节 1（防断梯度）】：
  预训练 `forward()` 内调用 `self.forward_student_backbone(student_x)`，确保 Student 主干与前置 10 个
  Chunk Tokens 全程保持 `requires_grad=True`；而 `extract_frozen_representations()` 仅在下游评测时
  开启 `with torch.no_grad():` 冻结使用。
- 【细节 2（帧权重复用 + Teacher `.detach()` 保命铁律）】：
  卡槽 H3 算出的 `target_utt` 与 `frame_weights` ($w_i$) 均强制 `.detach()`（阻断向 Teacher 的梯度回传），
  并将 `frame_weights` 传给 `compute_salience_weighted_frame_loss` 计算显著性加权帧级损失。
- 【细节 3（H1 在线 GPU 张量与离线音频副本双兼容）】：
  `self.h1_slot` 同时支持 GPU 在线实时张量变换（`vtlp_1A`, `channel_drop_1B`）与离线音频副本（`offline_wav_1C`）。
- 【叮嘱 1（H3 Padding 掩码 `-1e9` 对齐与有效帧长 $N_{\text{valid}}$）】：
  传入 `padding_mask` 将短句末尾 Padding 帧打分设为 `-1e9`（使 Padding 处 $w_i = 0$），且在
  $(1 + N_{\text{valid}} \cdot w_i)$ 中使用每句话的真实有效帧数 $N_{\text{valid}}$，防止短句权重被长 Padding 稀释。
- 【叮嘱 2（H1 `offline_wav_1C` Batch 内独立随机采样）】：
  使用形状为 `(B, 1, 1)` 的布尔掩码执行 `torch.where(mask, offline_pert_frames, source_frames)`，
  让同一个 Batch 里的不同样本独立掷硬币，降低梯度方差。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np

from models.hammer_slots import (
    BaseModule,
    H1_CANDIDATE_FUNCTIONS,
    H2_CANDIDATE_FUNCTIONS,
    H3_CANDIDATE_FUNCTIONS,
    HAS_TORCH,
    HammerRedLineViolationError,
    HammerSlotH1_ViewTransform,
    HammerSlotH2_Disentangler,
    HammerSlotH3_TemporalAggregator,
    attn_rollout_pooling_3B,
    h1_channel_dropout_1B,
    h1_offline_wav_1C,
    h1_original_none,
    h1_vtlp_formant_1A,
    h2_clean_proj_2A,
    h2_grl_adversarial_2B,
    h2_original_single,
    h3_original_mean,
    salience_pooling_3A,
    scalar_value,
    to_numpy,
)

if HAS_TORCH:
    import torch
    import torch.nn as nn


@dataclass
class HammerSwitchConfig:
    """命令行与 YAML 统一使用的 3 个锤头卡槽开关参数。"""

    view_type: str = "original_none"           # H1 开关: original_none | vtlp_1A | channel_drop_1B | offline_wav_1C
    disentangle_type: str = "original_single"  # H2 开关: original_single | clean_proj_2A | grl_adv_2B
    pooling_type: str = "original_mean"        # H3 开关: original_mean | salience_3A | attn_rollout_3B
    embed_dim: int = 768
    num_chunk_tokens: int = 10


def register_hammer_card(slot_name: str, card_id: str) -> Callable:
    """允许外部快速往 H1 / H2 / H3 卡槽注册新文献锤头函数。"""
    registries = {
        "H1": H1_CANDIDATE_FUNCTIONS,
        "H2": H2_CANDIDATE_FUNCTIONS,
        "H3": H3_CANDIDATE_FUNCTIONS,
    }
    if slot_name not in registries:
        raise ValueError(f"Unknown slot '{slot_name}'. Choose from H1, H2, H3.")

    def _decorator(fn: Callable) -> Callable:
        registries[slot_name][card_id] = fn
        return fn

    return _decorator


class PluggableDisEmo2vecFrontend(BaseModule):
    """
    带 3 个标准卡槽（H1 / H2 / H3）的 `emotion2vec` 乌龟壳主干：
    - `forward_student_backbone()`     : 预训练专用（全程保留计算图与 `requires_grad=True`）
    - `extract_frozen_representations()`: 下游评测专用（`torch.no_grad()` 冻结主干吐特征）
    """

    def __init__(
        self,
        h1_slot: Optional[HammerSlotH1_ViewTransform] = None,
        h2_slot: Optional[HammerSlotH2_Disentangler] = None,
        h3_slot: Optional[HammerSlotH3_TemporalAggregator] = None,
        cfg: Optional[HammerSwitchConfig] = None,
        embed_dim: int = 768,
        num_chunk_tokens: int = 10,
    ) -> None:
        super().__init__()
        self.cfg = cfg or HammerSwitchConfig(embed_dim=embed_dim, num_chunk_tokens=num_chunk_tokens)
        self.h1_slot = h1_slot or HammerSlotH1_ViewTransform(active_card=self.cfg.view_type)
        self.h2_slot = h2_slot or HammerSlotH2_Disentangler(
            active_card=self.cfg.disentangle_type, embed_dim=self.cfg.embed_dim
        )
        self.h3_slot = h3_slot or HammerSlotH3_TemporalAggregator(active_card=self.cfg.pooling_type)
        self.embed_dim = int(self.cfg.embed_dim)
        self.num_chunk_tokens = int(self.cfg.num_chunk_tokens)

        rng = np.random.default_rng(42)
        self.chunk_tokens_np = (
            rng.standard_normal((1, self.num_chunk_tokens, self.embed_dim)) * 0.02
        ).astype(np.float32)
        if HAS_TORCH:
            self.chunk_tokens = nn.Parameter(
                torch.from_numpy(self.chunk_tokens_np.copy()),
                requires_grad=True,
            )
            self.encoder_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward_student_backbone(self, student_x: Any) -> Tuple[Any, Any, Any]:
        """
        【细节 1 — 预训练专用：全程开启梯度】
        将前置 10 个可学习 Chunk Tokens 拼接在 `student_x` 前端送入 Student 主干编码，
        并按 H2 卡槽切分出 `z_utt_emo`（前 5 个 Tokens）与 `z_utt_cont`（后 5 个 Tokens）。
        """
        if HAS_TORCH and isinstance(student_x, torch.Tensor):
            bsz = student_x.shape[0]
            chunks = self.chunk_tokens.expand(bsz, -1, -1) + student_x[:, : self.num_chunk_tokens, :]
            concat_seq = torch.cat([chunks, student_x], dim=1)
            encoded = self.encoder_proj(concat_seq)
            encoded_chunks = encoded[:, : self.num_chunk_tokens, :]
            x_frm_emo = encoded[:, self.num_chunk_tokens :, :]
            z_utt_emo, z_utt_cont = self.h2_slot.split_and_pool_chunks(
                encoded_chunks, disentangle_type=self.cfg.disentangle_type
            )
            return x_frm_emo, z_utt_emo, z_utt_cont

        x_np = to_numpy(student_x)
        bsz = x_np.shape[0]
        chunks = np.repeat(self.chunk_tokens_np, bsz, axis=0)
        z_utt_emo, z_utt_cont = self.h2_slot.split_and_pool_chunks(
            chunks + x_np[:, : self.num_chunk_tokens, :],
            disentangle_type=self.cfg.disentangle_type,
        )
        return x_np, z_utt_emo, z_utt_cont

    def extract_frozen_representations(self, acoustic_frames: Any) -> Tuple[Any, Any, Any]:
        """
        【细节 1 — 下游评测专用：彻底冻结主干梯度】
        仅供 `eval/eval_all_8_metrics.py` 在下游线性探针测试时调用，内部显式包裹 `torch.no_grad()`。
        """
        if HAS_TORCH and isinstance(acoustic_frames, torch.Tensor):
            was_training = self.training
            self.eval()
            with torch.no_grad():
                x_frm_emo, z_utt_emo, z_utt_cont = self.forward_student_backbone(acoustic_frames)
            if was_training:
                self.train()
            return x_frm_emo, z_utt_emo, z_utt_cont
        return self.forward_student_backbone(acoustic_frames)

    def forward(
        self,
        source_frames: Any,
        target_layer_results: Any,
        frame_energy: Optional[Any] = None,
        p_hubert_target: Optional[Any] = None,
        attn_weights: Optional[Any] = None,
        offline_pert_frames: Optional[Any] = None,
        mask_indices: Optional[Any] = None,
        padding_mask: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        =======================================================================
        在原版 emotion2vec.py 的乌龟壳 forward() 里，只通过 3 个卡槽开关调用 H1, H3, H2：
        =======================================================================
        """
        # 1. [卡槽 H1 开关: --view_type] 兼容 GPU 在线张量微扰与离线副本 Batch 内独立掷硬币（细节 3 + 叮嘱 2）
        student_x, h1_meta = self.h1_slot(
            source_frames,
            frame_energy_std=frame_energy,
            offline_pert_x=offline_pert_frames,
            view_type=self.cfg.view_type,
        )

        # 2. [卡槽 H3 开关: --pooling_type] 含 padding_mask -1e9 保护，强制 .detach() 封死 Teacher 梯度（细节 2 + 叮嘱 1）
        target_utt, frame_weights, h3_meta = self.h3_slot(
            target_layer_results,
            frame_energy_std=frame_energy,
            attn_weights=attn_weights,
            padding_mask=padding_mask,
            pooling_type=self.cfg.pooling_type,
        )
        if HAS_TORCH and isinstance(target_utt, torch.Tensor):
            target_utt = target_utt.detach()
            frame_weights = frame_weights.detach()

        # 3. [预训练 Student 主干前向传播 — 全程保留梯度 requires_grad=True！（细节 1）]
        x_frm_emo, z_utt_emo, z_utt_cont = self.forward_student_backbone(student_x)

        # 4. [卡槽 H3 (1 + N_valid * w_i) 帧损失 + 卡槽 H2 开关: --disentangle_type]（细节 2 + 叮嘱 1）
        loss_frm = self.h3_slot.compute_salience_weighted_frame_loss(
            y_student_frames=x_frm_emo,
            y_teacher_frames=target_layer_results.detach() if (HAS_TORCH and isinstance(target_layer_results, torch.Tensor)) else target_layer_results,
            frame_weights=frame_weights,
            mask_indices=mask_indices,
            padding_mask=padding_mask,
        )
        h2_meta = self.h2_slot.compute_disentangle_losses(
            z_emo=z_utt_emo,
            z_cont=z_utt_cont,
            y_teacher_sal=target_utt,
            p_hubert_target=p_hubert_target,
            disentangle_type=self.cfg.disentangle_type,
        )

        loss_total = loss_frm + h2_meta["loss_total_h2"]

        return {
            "x_frm_emo": x_frm_emo,
            "z_utt_emo": z_utt_emo,
            "z_utt_cont": z_utt_cont,
            "target_utt": target_utt,
            "frame_weights": frame_weights,
            "loss_frm": loss_frm,
            "loss_total": loss_total,
            "h1_meta": h1_meta,
            "h2_meta": h2_meta,
            "h3_meta": h3_meta,
        }


def build_hammer_suite_from_config(cfg: Optional[Dict[str, Any]] = None) -> PluggableDisEmo2vecFrontend:
    """支持从 YAML 配置或命令行开关字典构建包含 H1/H2/H3 的前端模型。"""
    cfg = cfg or {}
    slots_cfg = cfg.get("hammer_slots", {})
    h1_cfg = slots_cfg.get("H1_view_transform", {})
    h2_cfg = slots_cfg.get("H2_disentangler", {})
    h3_cfg = slots_cfg.get("H3_temporal_aggregator", {})

    view_type = cfg.get("view_type", h1_cfg.get("active_card", "original_none"))
    disentangle_type = cfg.get("disentangle_type", h2_cfg.get("active_card", "original_single"))
    pooling_type = cfg.get("pooling_type", h3_cfg.get("active_card", "original_mean"))
    embed_dim = cfg.get("pretraining", {}).get("embed_dim", 768)

    h1 = HammerSlotH1_ViewTransform(active_card=view_type, **h1_cfg.get("params", {}))
    h2 = HammerSlotH2_Disentangler(
        active_card=disentangle_type, embed_dim=embed_dim, **h2_cfg.get("params", {})
    )
    h3 = HammerSlotH3_TemporalAggregator(
        active_card=pooling_type,
        temperature_tau_s=h3_cfg.get("params", {}).get("temperature_tau_s", 0.5),
    )
    switch_cfg = HammerSwitchConfig(
        view_type=view_type,
        disentangle_type=disentangle_type,
        pooling_type=pooling_type,
        embed_dim=embed_dim,
    )
    return PluggableDisEmo2vecFrontend(h1_slot=h1, h2_slot=h2, h3_slot=h3, cfg=switch_cfg)


def run_hammer_slots_self_test() -> Dict[str, Any]:
    """自检：依次验证原版开关、候选锤头开关、三大防坑细节及两大微观张量保护叮嘱。"""
    bsz, n_frames, dim = 4, 50, 768
    if HAS_TORCH:
        torch.manual_seed(42)
        dummy_frames = torch.randn(bsz, n_frames, dim)
        dummy_offline_pert = torch.randn(bsz, n_frames, dim)
        dummy_energy = torch.randn(bsz, n_frames)
        dummy_padding_mask = torch.zeros(bsz, n_frames, dtype=torch.bool)
        dummy_padding_mask[1, 35:] = True  # 模拟第 2 句话为短句（末尾 15 帧是 Padding）
    else:
        rng = np.random.default_rng(42)
        dummy_frames = rng.standard_normal((bsz, n_frames, dim)).astype(np.float32)
        dummy_offline_pert = rng.standard_normal((bsz, n_frames, dim)).astype(np.float32)
        dummy_energy = rng.standard_normal((bsz, n_frames)).astype(np.float32)
        dummy_padding_mask = np.zeros((bsz, n_frames), dtype=bool)
        dummy_padding_mask[1, 35:] = True  # 模拟第 2 句话为短句（末尾 15 帧是 Padding）

    dummy_energy[:, :10] = -2.5   # 纯静音段
    dummy_energy[:, 20:30] = +1.8  # 情绪爆发段

    # 1. 原版基线开关自检
    suite_tbd = build_hammer_suite_from_config(
        {
            "view_type": "original_none",
            "disentangle_type": "original_single",
            "pooling_type": "original_mean",
        }
    )
    suite_tbd.train()
    out_tbd = suite_tbd(
        dummy_frames, dummy_frames, frame_energy=dummy_energy, padding_mask=dummy_padding_mask
    )

    # 2. 候选锤头组合开关自检 (--view_type=vtlp_1A --disentangle_type=clean_proj_2A --pooling_type=salience_3A)
    suite_v0 = build_hammer_suite_from_config(
        {
            "view_type": "vtlp_1A",
            "disentangle_type": "clean_proj_2A",
            "pooling_type": "salience_3A",
        }
    )
    suite_v0.train()
    out_v0 = suite_v0(
        dummy_frames,
        dummy_frames,
        frame_energy=dummy_energy,
        offline_pert_frames=dummy_offline_pert,
        padding_mask=dummy_padding_mask,
    )

    # 验证【叮嘱 1】：短句末尾 Padding 帧上的权重 w_i 必须精确为 0.0！
    w_v0_np = to_numpy(out_v0["frame_weights"])
    padding_weight_sum = float(np.sum(np.abs(w_v0_np[1, 35:])))
    assert padding_weight_sum == 0.0, f"Padding 帧权重未清零: {padding_weight_sum}!"

    # 验证【叮嘱 2】：离线音频副本 Batch 内独立掷硬币采样 (offline_wav_1C)
    suite_offline = build_hammer_suite_from_config({"view_type": "offline_wav_1C"})
    suite_offline.train()
    out_off = suite_offline(dummy_frames, dummy_frames, offline_pert_frames=dummy_offline_pert)
    assert out_off["h1_meta"]["batch_independent_coin_flip"] is True

    # 若在 PyTorch 环境下，验证 loss_total = loss_frm + loss_h2 能真实反向传播至 chunk_tokens 与 encoder_proj！
    if HAS_TORCH:
        suite_v0.zero_grad()
        out_v0["loss_total"].backward()
        assert suite_v0.chunk_tokens.grad is not None, "chunk_tokens 梯度断流！"
        assert torch.norm(suite_v0.chunk_tokens.grad).item() > 0.0, "chunk_tokens 梯度为 0！"

    silence_mean_w = float(w_v0_np[:, :10].mean())
    burst_mean_w = float(w_v0_np[:, 20:30].mean())
    assert silence_mean_w < burst_mean_w, (
        f"H3 silence weight ({silence_mean_w}) must be < burst weight ({burst_mean_w})!"
    )

    return {
        "tbd_default_status": {
            "H1": out_tbd["h1_meta"]["status"],
            "H2": out_tbd["h2_meta"]["status"],
            "H3": out_tbd["h3_meta"]["status"],
        },
        "v0_redline_verification": {
            "H1_energy_envelope_corr": out_v0["h1_meta"]["energy_envelope_corr"],
            "H1_batch_independent_coin_flip": out_off["h1_meta"]["batch_independent_coin_flip"],
            "H2_teacher_backflow_blocked": out_v0["h2_meta"]["teacher_backflow_blocked"],
            "H2_loss_utt": round(scalar_value(out_v0["h2_meta"]["loss_utt"]), 4),
            "H2_loss_sponge": round(scalar_value(out_v0["h2_meta"]["loss_sponge"]), 4),
            "H2_loss_dis": round(scalar_value(out_v0["h2_meta"]["loss_dis"]), 4),
            "H3_loss_frm_weighted": round(scalar_value(out_v0["loss_frm"]), 4),
            "H3_padding_weight_sum": padding_weight_sum,
            "H3_silence_mean_weight": round(silence_mean_w, 6),
            "H3_burst_mean_weight": round(burst_mean_w, 6),
            "H3_weights_sum_to_one": out_v0["h3_meta"]["weights_sum_to_one"],
        },
    }
