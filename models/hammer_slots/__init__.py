#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`models/hammer_slots/` —— 存放 3 个独立锤头卡槽文件的专属文件夹
================================================================
- `slot_h1_view.py`       : 卡槽 H1（输入视角变换槽 `HammerSlotH1_ViewTransform`）
- `slot_h2_disentangle.py`: 卡槽 H2（词汇-情感解耦槽 `HammerSlotH2_Disentangler`）
- `slot_h3_pooling.py`    : 卡槽 H3（时序显著性汇聚槽 `HammerSlotH3_TemporalAggregator`）
"""

from .slot_base import BaseModule, HAS_TORCH, HammerRedLineViolationError, scalar_value, to_numpy
from .slot_h1_view import (
    H1_CANDIDATE_FUNCTIONS,
    HammerSlotH1_ViewTransform,
    h1_channel_dropout_1B,
    h1_offline_wav_1C,
    h1_original_none,
    h1_vtlp_formant_1A,
)
from .slot_h2_disentangle import (
    H2_CANDIDATE_FUNCTIONS,
    HammerSlotH2_Disentangler,
    h2_clean_proj_2A,
    h2_grl_adversarial_2B,
    h2_original_single,
    h2_vicreg_cov_2C,
)
from .slot_h3_pooling import (
    H3_CANDIDATE_FUNCTIONS,
    HammerSlotH3_TemporalAggregator,
    attn_rollout_pooling_3B,
    autopool_pooling_3C,
    h3_original_mean,
    salience_pooling_3A,
)

__all__ = [
    "BaseModule",
    "HAS_TORCH",
    "HammerRedLineViolationError",
    "to_numpy",
    "scalar_value",
    "HammerSlotH1_ViewTransform",
    "HammerSlotH2_Disentangler",
    "HammerSlotH3_TemporalAggregator",
    "H1_CANDIDATE_FUNCTIONS",
    "H2_CANDIDATE_FUNCTIONS",
    "H3_CANDIDATE_FUNCTIONS",
    "h1_original_none",
    "h1_vtlp_formant_1A",
    "h1_channel_dropout_1B",
    "h2_original_single",
    "h2_clean_proj_2A",
    "h2_grl_adversarial_2B",
    "h2_vicreg_cov_2C",
    "h3_original_mean",
    "salience_pooling_3A",
    "attn_rollout_pooling_3B",
    "autopool_pooling_3C",
]
