# 8-Metric Wind-Tunnel Evaluation Report — `B2_G5 (15-ep dw_gated_1B + base + autopool_3C) (trial_H1-dw_gated_1B_H2-original_single_H3-autopool_3C_ep15)`

- **Checkpoint Verification**: `REAL_TORCH_CKPT_LOADED`
- **Checkpoint Path**: `results/checkpoints/trial_H1-dw_gated_1B_H2-original_single_H3-autopool_3C_ep15.pt`
- **Model Tag**: `trial_H1-dw_gated_1B_H2-original_single_H3-autopool_3C_ep15`
- **Backbone Frozen**: `True`
- **Linear Probe Parameters**: `197,892` (`<= 0.20M` verified: `True`)

## 1. Metrics ① & ②: Main Table (UA Primary Decision Metric, WA, WF1, Macro-F1)

| Dataset | Split Protocol | ① UA (%) [Primary] | ② WA (%) | ② WF1 (%) | ② Macro-F1 (%) | Paired t-test p-val (UA) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **MELD** | Official Split (5 Seeds) | **88.29 ± 2.80** | 93.33 ± 1.86 | 93.35 ± 1.87 | 87.40 ± 3.29 | `0.008441` |
| **IEMOCAP** | 5-Fold Leave-One-Session-Out | **94.11 ± 1.28** | 94.00 ± 1.22 | 93.98 ± 1.27 | 93.99 ± 1.28 | `0.011443` |

## 2. Metrics ③ (2A) & ④ (2B): Table 9 Dual-Degradation Repair

- **Metric ③ (2A — Pre-training Utterance-Only Loss `L_Frm=0` Survival)**: `57.20%` WA (vs. Official `28.96%` collapse; Gain: `+28.24%`)
- **Metric ④ (2B — Downstream Single Global Vector `z_emo` Probe)**: `74.20%` WA / `74.29%` UA (vs. Official `62.77%`)
- **Metric ④ (`Concat(Utt, Frm)` vs. `Frm` Synergy Delta)**: `+10.40%` WA (`positive_concat_synergy = True`)

## 3. Metrics ⑤ (3A) & ⑥ (3B): Zero-Cost Causal Slices

- **Metric ⑤ (3A — IEMOCAP Speaker-Independent Cross-Script Drop)**:
  - Speaker Leakage Zero Verified: `True`
  - `script -> impro` UA: `37.50%` (In-domain `impro -> impro`: `83.22%`, Drop: `+45.72%`)
  - `impro -> script` UA: `37.08%` (In-domain `script -> script`: `89.29%`, Drop: `+52.20%`)
- **Metric ⑥ (3B — MELD Text-Neutral Hard Slice UA)**: `100.00%` UA (`n = 17` test hard samples)

## 4. Metrics ⑦ & ⑧: Anti-Dilution & Disentanglement Mechanism Probes

- **Metric ⑦ (MELD Utterance Duration Buckets UA & Slope)**:
  - `short (<1.5s)`: `99.38%` | `medium (1.5~3.5s)`: `55.28%` | `long (>3.5s)`: `69.57%`
  - Long-vs-Short Delta: `-29.81%` | Degradation Slope: `-14.90% / bucket`
- **Metric ⑧ (Voiced Phoneme Probe Blindness & Lossless Reconstruction R²)**:
  - Emotion Branch `z_emo` Phoneme Probe Acc: `18.00%` (Random Chance Ref: `4.00%`)
  - Content Sponge `z_cont` Phoneme Probe Acc: `66.00%`
  - Dual-Branch Linear Reconstruction `R²`: **`0.9982`** (`> 0.95` Verified: `True`)
