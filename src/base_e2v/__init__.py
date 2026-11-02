#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`src/base_e2v/` — DisEmo2vec Step 5 Base Code 5-Layer Topology (`S1 ~ S5`)
==========================================================================
Exports all 8 core static symbols (`[SYM-E2V-S1-01]` ~ `[SYM-E2V-S5-01]`):
- `S1`: `BaseAudioCollaterAndNormalizer` (`[SYM-E2V-S1-01]`)
- `S2`: `ConvFeatureExtractionModel` (`[SYM-E2V-S2-01]`), `MaskingAndPositionalConv` (`[SYM-E2V-S2-02]`)
- `S3`: `PrependChunkTokens` (`[SYM-E2V-S3-01]`), `EMATeacherTargetExtractor` (`[SYM-E2V-S3-02]`),
        `BaseEmotion2VecCriterion` (`[SYM-E2V-S3-03]`)
- `S4`: `EMAApplierAndFp16Guard` (`[SYM-E2V-S4-01]`)
- `S5`: `FrozenFeatureExtractorAndLinearProbe` (`[SYM-E2V-S5-01]`)
"""

from .s1_audio_dataset import BaseAudioCollaterAndNormalizer
from .s2_conv_frontend import CONV_FEATURE_LAYERS_SPEC, ConvFeatureExtractionModel, MaskingAndPositionalConv
from .s3_criterion_loss import BaseEmotion2VecCriterion
from .s3_dual_stream_backbone import EMATeacherTargetExtractor, PrependChunkTokens
from .s4_ema_and_fp16 import EMAApplierAndFp16Guard
from .s5_downstream_probe import FrozenFeatureExtractorAndLinearProbe

__all__ = [
    "BaseAudioCollaterAndNormalizer",
    "CONV_FEATURE_LAYERS_SPEC",
    "ConvFeatureExtractionModel",
    "MaskingAndPositionalConv",
    "PrependChunkTokens",
    "EMATeacherTargetExtractor",
    "BaseEmotion2VecCriterion",
    "EMAApplierAndFp16Guard",
    "FrozenFeatureExtractorAndLinearProbe",
]
