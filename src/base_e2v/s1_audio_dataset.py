#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DisEmo2vec Step 5 - Layer S1: Audio Dataset, Waveform Normalization & Dynamic Padding
=====================================================================================
Contains `[SYM-E2V-S1-01]` (`BaseAudioCollaterAndNormalizer`), reproducing the exact
Fairseq `FileAudioDataset.__getitem__` and `collater` signal pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np

from models.hammer_slots.slot_base import BaseModule, HAS_TORCH, torch, F, to_numpy


class BaseAudioCollaterAndNormalizer(BaseModule):
    """
    [SYM-E2V-S1-01] 音频单句零均值单位方差归一化与动态补零组批器
    @bce_contract:
      uid: "SYM-E2V-S1-01"
      layer: "S1"
      name: "FileAudioDataset 单句波形层归一化与 Collater 动态补零掩码器"
      file_rel_path: "src/base_e2v/s1_audio_dataset.py"
      tensor_input: "raw_wavs: List[FloatArray[T_b]] (16kHz mono PCM, variable length T_b)"
      tensor_output: "source: FloatTensor[B, T_max], wav_padding_mask: BoolTensor[B, T_max] (True=Pad), frame_energy_std: FloatTensor[B, N_frm]"
      signal_math_assumptions:
        - "【信号假设 S1-A1】每条语音在补零前独立执行整句零均值单位方差归一化: x_hat = (x - mu_x) / sqrt(sigma_x^2 + 1e-5)，消除录音设备增益差异，但保留句内相对能量包络起伏。"
        - "【信号假设 S1-A2】动态组批时按 Batch 内最长语音 T_max 在右侧补 0.0 (恰等于归一化后的静音均值)，同步生成采样点级布尔掩码 wav_padding_mask (True 代表补零点，False 代表真实采样点)。"
        - "【信号假设 S1-A3】同步以窗长 400 (25ms)、步长 320 (20ms, 对齐 S2 320倍下采样) 仅在有效帧上计算句内零均值单位方差短时能量 e_tilde_i = (e_i - mu_e) / (sigma_e + 1e-6)。"
      hard_invariants:
        - "【保命护栏 I0】单句 LayerNorm 必须在右侧补零 (Padding) 之前对真实采样点 [0 : T_b] 独立执行！严禁先补零再做整句归一化，否则大量补零会拉低方差、扭曲短句波形幅值！"
      broken_assumptions:
        - "【原版隐性缺陷 B0】Student 与 Teacher 共享完全相同的原始波形输入 source，未对 Student 施加任何破坏固定音素共振峰的视角微扰，导致模型可通过死记特定共振峰包络走情绪分类捷径。"
      reserved_slot_id: "Upstream provider for [SLOT-H1: HammerSlotH1_ViewTransform] & [SLOT-H3: frame_energy_std]"
    """

    def __init__(self, normalize: bool = True, hop_length: int = 320, win_length: int = 400) -> None:
        super().__init__()
        self.normalize = bool(normalize)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)

    @staticmethod
    def normalize_single_utterance(wav: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Performs Fairseq `F.layer_norm(wav, wav.shape)` on a single unpadded 1D waveform."""
        wav_f = np.asarray(wav, dtype=np.float32)
        mu = float(np.mean(wav_f))
        var = float(np.var(wav_f))
        return ((wav_f - mu) / np.sqrt(var + eps)).astype(np.float32)

    def compute_aligned_frame_energy(
        self, raw_wav: np.ndarray, target_num_frames: int, valid_num_frames: int, eps: float = 1e-6
    ) -> np.ndarray:
        """
        Computes short-time RMS energy at 50Hz frame rate (`hop=320`, `win=400`),
        standardizing ONLY over the `valid_num_frames` non-padded frames.
        """
        energies = np.zeros(target_num_frames, dtype=np.float32)
        for i in range(valid_num_frames):
            start = i * self.hop_length
            seg = raw_wav[start : start + self.win_length]
            if len(seg) > 0:
                energies[i] = float(np.sqrt(np.mean(seg * seg) + eps))
        if valid_num_frames > 1:
            valid_e = energies[:valid_num_frames]
            mu_e = float(np.mean(valid_e))
            std_e = float(np.std(valid_e))
            energies[:valid_num_frames] = (valid_e - mu_e) / (std_e + eps)
        if valid_num_frames < target_num_frames:
            energies[valid_num_frames:] = -5.0  # Padded frames marked as deep silence
        return energies

    def forward(self, raw_wavs: List[np.ndarray]) -> Dict[str, Any]:
        """
        Collates a list of variable-length 16kHz waveforms into a padded batch:
        1. Normalizes each utterance independently BEFORE padding (Hard Invariant I0).
        2. Right-pads to `T_max` with `0.0` and builds `wav_padding_mask` (`True` = padded).
        3. Extracts frame-aligned standardized short-time energy `frame_energy_std`.
        """
        bsz = len(raw_wavs)
        lengths = [len(w) for w in raw_wavs]
        t_max = max(lengths)
        n_max_frames = max(1, (t_max - self.win_length) // self.hop_length + 1)

        padded_source = np.zeros((bsz, t_max), dtype=np.float32)
        wav_padding_mask = np.ones((bsz, t_max), dtype=bool)
        frame_energy_std = np.zeros((bsz, n_max_frames), dtype=np.float32)
        valid_frame_counts = np.zeros(bsz, dtype=np.int64)

        for b, wav in enumerate(raw_wavs):
            t_b = len(wav)
            norm_wav = self.normalize_single_utterance(wav) if self.normalize else np.asarray(wav, dtype=np.float32)
            padded_source[b, :t_b] = norm_wav
            wav_padding_mask[b, :t_b] = False

            n_valid_b = max(1, (t_b - self.win_length) // self.hop_length + 1)
            valid_frame_counts[b] = n_valid_b
            frame_energy_std[b] = self.compute_aligned_frame_energy(
                norm_wav, target_num_frames=n_max_frames, valid_num_frames=n_valid_b
            )

        if HAS_TORCH:
            return {
                "source": torch.from_numpy(padded_source),
                "wav_padding_mask": torch.from_numpy(wav_padding_mask),
                "frame_energy_std": torch.from_numpy(frame_energy_std),
                "valid_lengths": lengths,
                "valid_frame_counts": valid_frame_counts.tolist(),
            }
        return {
            "source": padded_source,
            "wav_padding_mask": wav_padding_mask,
            "frame_energy_std": frame_energy_std,
            "valid_lengths": lengths,
            "valid_frame_counts": valid_frame_counts.tolist(),
        }
