#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backward-compatible entry point re-exporting all Hammer Slots (H1, H2, H3)
from `models/hammer_modules.py` and `models/hammer_slots/`.
"""

from models.hammer_modules import (
    BaseModule,
    H1_CANDIDATE_FUNCTIONS,
    H2_CANDIDATE_FUNCTIONS,
    H3_CANDIDATE_FUNCTIONS,
    HAS_TORCH,
    HammerRedLineViolationError,
    HammerSlotH1_ViewTransform,
    HammerSlotH2_Disentangler,
    HammerSlotH3_TemporalAggregator,
    HammerSwitchConfig,
    PluggableDisEmo2vecFrontend,
    attn_rollout_pooling_3B,
    build_hammer_suite_from_config,
    h1_channel_dropout_1B,
    h1_original_none,
    h1_vtlp_formant_1A,
    h2_clean_proj_2A,
    h2_grl_adversarial_2B,
    h2_original_single,
    h3_original_mean,
    register_hammer_card,
    run_hammer_slots_self_test,
    salience_pooling_3A,
)

__all__ = [
    "BaseModule",
    "HAS_TORCH",
    "HammerRedLineViolationError",
    "HammerSwitchConfig",
    "HammerSlotH1_ViewTransform",
    "HammerSlotH2_Disentangler",
    "HammerSlotH3_TemporalAggregator",
    "PluggableDisEmo2vecFrontend",
    "H1_CANDIDATE_FUNCTIONS",
    "H2_CANDIDATE_FUNCTIONS",
    "H3_CANDIDATE_FUNCTIONS",
    "h1_original_none",
    "h1_vtlp_formant_1A",
    "h1_channel_dropout_1B",
    "h2_original_single",
    "h2_clean_proj_2A",
    "h2_grl_adversarial_2B",
    "h3_original_mean",
    "salience_pooling_3A",
    "attn_rollout_pooling_3B",
    "build_hammer_suite_from_config",
    "register_hammer_card",
    "run_hammer_slots_self_test",
]
