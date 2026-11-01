#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`models/hammer_modules.py` —— 锤头卡槽总控中心与模型 `forward()` 开关调用层
==========================================================================
结合「分文件管理 3 大卡槽（`models/hammer_slots/`）」与「`forward()` 中 10 行 If-Else 开关调卡槽」：

1. 具体的候选锤头函数按职责拆放在 `models/hammer_slots/` 文件夹下的 3 个卡槽文件里：
   - `models/hammer_slots/slot_h1_view.py`        -> 卡槽 H1（输入视角变换，`--view_type`）
   - `models/hammer_slots/slot_h2_disentangle.py` -> 卡槽 H2（词汇-情感解耦，`--disentangle_type`）
   - `models/hammer_slots/slot_h3_pooling.py`     -> 卡槽 H3（时序显著性汇聚，`--pooling_type`）

2. 本文件 `hammer_modules.py` 统一汇总导入 H1、H2、H3，并在模型 `forward()` 中只用极简开关：
   - `--view_type`        : `"original_none"` | `"vtlp_1A"` | `"channel_drop_1B"`
   - `--disentangle_type` : `"original_single"` | `"clean_proj_2A"` | `"grl_adv_2B"`
   - `--pooling_type`     : `"original_mean"` | `"salience_3A"` | `"attn_rollout_3B"`
   实现单变量小步快跑试锤！
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

    view_type: str = "original_none"         # H1 开关: original_none | vtlp_1A | channel_drop_1B
    disentangle_type: str = "original_single"  # H2 开关: original_single | clean_proj_2A | grl_adv_2B
    pooling_type: str = "original_mean"      # H3 开关: original_mean | salience_3A | attn_rollout_3B
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
    在 `forward()` 中仅用清晰的开关调用 `self.h1_slot`、`self.h3_slot`、`self.h2_slot`！
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

        if HAS_TORCH:
            self.chunk_tokens = nn.Parameter(torch.randn(1, self.num_chunk_tokens, self.embed_dim) * 0.02)
            self.encoder_proj = nn.Linear(self.embed_dim, self.embed_dim)
        else:
            rng = np.random.default_rng(42)
            self.chunk_tokens_np = (
                rng.standard_normal((1, self.num_chunk_tokens, self.embed_dim)) * 0.02
            ).astype(np.float32)

    def forward(
        self,
        source_frames: Any,
        target_layer_results: Any,
        frame_energy: Optional[Any] = None,
        p_hubert_target: Optional[Any] = None,
        attn_weights: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        =======================================================================
        在原版 emotion2vec.py 的乌龟壳 forward() 里，只通过 3 个卡槽开关调用 H1, H3, H2：
        =======================================================================
        """
        # 1. [卡槽 H1 开关: --view_type] Student 输入视角变换（Teacher 始终吃干净输入）
        student_x, h1_meta = self.h1_slot(
            source_frames, frame_energy, view_type=self.cfg.view_type
        )

        # 2. [卡槽 H3 开关: --pooling_type] 替换原版写死的 target_layer_results.mean(dim=1)
        target_utt, frame_weights, h3_meta = self.h3_slot(
            target_layer_results,
            frame_energy_std=frame_energy,
            attn_weights=attn_weights,
            pooling_type=self.cfg.pooling_type,
        )

        # 3. [卡槽 H2 开关: --disentangle_type] Student 双分支切分与去词汇解耦损失
        x_frm_emo, z_utt_emo, z_utt_cont = self.extract_frozen_representations(student_x)
        h2_meta = self.h2_slot.compute_disentangle_losses(
            z_emo=z_utt_emo,
            z_cont=z_utt_cont,
            y_teacher_sal=target_utt,
            p_hubert_target=p_hubert_target,
            disentangle_type=self.cfg.disentangle_type,
        )

        return {
            "x_frm_emo": x_frm_emo,
            "z_utt_emo": z_utt_emo,
            "z_utt_cont": z_utt_cont,
            "target_utt": target_utt,
            "frame_weights": frame_weights,
            "h1_meta": h1_meta,
            "h2_meta": h2_meta,
            "h3_meta": h3_meta,
        }

    def extract_frozen_representations(self, acoustic_frames: Any) -> Tuple[Any, Any, Any]:
        """下游评测冻结主干时唯一调用的特征导出接口。"""
        if HAS_TORCH and isinstance(acoustic_frames, torch.Tensor):
            bsz = acoustic_frames.shape[0]
            chunks = self.chunk_tokens.expand(bsz, -1, -1)
            concat_seq = torch.cat([chunks, acoustic_frames], dim=1)
            encoded = self.encoder_proj(concat_seq)
            encoded_chunks = encoded[:, : self.num_chunk_tokens, :]
            x_frm_emo = encoded[:, self.num_chunk_tokens :, :]
            z_utt_emo, z_utt_cont = self.h2_slot.split_and_pool_chunks(
                encoded_chunks, disentangle_type=self.cfg.disentangle_type
            )
            return x_frm_emo, z_utt_emo, z_utt_cont

        x_np = to_numpy(acoustic_frames)
        bsz = x_np.shape[0]
        chunks = np.repeat(self.chunk_tokens_np, bsz, axis=0)
        z_utt_emo, z_utt_cont = self.h2_slot.split_and_pool_chunks(
            chunks + x_np[:, : self.num_chunk_tokens, :],
            disentangle_type=self.cfg.disentangle_type,
        )
        return x_np, z_utt_emo, z_utt_cont


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
    """自检：依次验证原版开关 (`original_*`) 与候选锤头开关 (`vtlp_1A`, `clean_proj_2A`, `salience_3A`)。"""
    bsz, n_frames, dim = 4, 50, 768
    if HAS_TORCH:
        torch.manual_seed(42)
        dummy_frames = torch.randn(bsz, n_frames, dim)
        dummy_energy = torch.randn(bsz, n_frames)
    else:
        rng = np.random.default_rng(42)
        dummy_frames = rng.standard_normal((bsz, n_frames, dim)).astype(np.float32)
        dummy_energy = rng.standard_normal((bsz, n_frames)).astype(np.float32)

    dummy_energy[:, :10] = -2.5  # 纯静音段
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
    out_tbd = suite_tbd(dummy_frames, dummy_frames, frame_energy=dummy_energy)

    # 2. 候选锤头组合开关自检 (--view_type=vtlp_1A --disentangle_type=clean_proj_2A --pooling_type=salience_3A)
    suite_v0 = build_hammer_suite_from_config(
        {
            "view_type": "vtlp_1A",
            "disentangle_type": "clean_proj_2A",
            "pooling_type": "salience_3A",
        }
    )
    suite_v0.train()
    out_v0 = suite_v0(dummy_frames, dummy_frames, frame_energy=dummy_energy)

    w_v0_np = to_numpy(out_v0["frame_weights"])
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
            "H2_teacher_backflow_blocked": out_v0["h2_meta"]["teacher_backflow_blocked"],
            "H2_loss_utt": round(scalar_value(out_v0["h2_meta"]["loss_utt"]), 4),
            "H2_loss_sponge": round(scalar_value(out_v0["h2_meta"]["loss_sponge"]), 4),
            "H2_loss_dis": round(scalar_value(out_v0["h2_meta"]["loss_dis"]), 4),
            "H3_silence_mean_weight": round(silence_mean_w, 6),
            "H3_burst_mean_weight": round(burst_mean_w, 6),
            "H3_weights_sum_to_one": out_v0["h3_meta"]["weights_sum_to_one"],
        },
    }
