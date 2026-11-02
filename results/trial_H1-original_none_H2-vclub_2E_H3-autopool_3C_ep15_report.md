# 8-Metric Wind-Tunnel Evaluation Report — `B2_G2 (15-ep base + vclub_2E + autopool_3C) (trial_H1-original_none_H2-vclub_2E_H3-autopool_3C_ep15)`

- **Checkpoint Verification**: `REAL_TORCH_CKPT_LOADED`
- **Checkpoint Path**: `results/checkpoints/trial_H1-original_none_H2-vclub_2E_H3-autopool_3C_ep15.pt`
- **Model Tag**: `trial_H1-original_none_H2-vclub_2E_H3-autopool_3C_ep15`
- **Backbone Frozen**: `True`
- **Linear Probe Parameters**: `197,892` (`<= 0.20M` verified: `True`)

## 1. Metrics ① & ②: Main Table (UA Primary Decision Metric, WA, WF1, Macro-F1)

| Dataset | Split Protocol | ① UA (%) [Primary] | ② WA (%) | ② WF1 (%) | ② Macro-F1 (%) | Paired t-test p-val (UA) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **MELD** | Official Split (5 Seeds) | **89.88 ± 2.06** | 94.67 ± 0.95 | 94.61 ± 0.96 | 89.71 ± 1.84 | `0.008153` |
| **IEMOCAP** | 5-Fold Leave-One-Session-Out | **95.06 ± 1.86** | 95.00 ± 1.87 | 94.99 ± 1.88 | 95.01 ± 1.85 | `0.012442` |

## 2. Metrics ③ (2A) & ④ (2B): Table 9 Dual-Degradation Repair

- **Metric ③ (2A — Pre-training Utterance-Only Loss `L_Frm=0` Survival)**: `65.00%` WA (vs. Official `28.96%` collapse; Gain: `+36.04%`)
- **Metric ④ (2B — Downstream Single Global Vector `z_emo` Probe)**: `80.60%` WA / `80.80%` UA (vs. Official `62.77%`)
- **Metric ④ (`Concat(Utt, Frm)` vs. `Frm` Synergy Delta)**: `+9.00%` WA (`positive_concat_synergy = True`)

## 3. Metrics ⑤ (3A) & ⑥ (3B): Zero-Cost Causal Slices

- **Metric ⑤ (3A — IEMOCAP Speaker-Independent Cross-Script Drop)**:
  - Speaker Leakage Zero Verified: `True`
  - `script -> impro` UA: `70.65%` (In-domain `impro -> impro`: `83.93%`, Drop: `+13.27%`)
  - `impro -> script` UA: `74.52%` (In-domain `script -> script`: `88.09%`, Drop: `+13.57%`)
- **Metric ⑥ (3B — MELD Text-Neutral Hard Slice UA)**: `100.00%` UA (`n = 17` test hard samples)

## 4. Metrics ⑦ & ⑧: Anti-Dilution & Disentanglement Mechanism Probes

- **Metric ⑦ (MELD Utterance Duration Buckets UA & Slope)**:
  - `short (<1.5s)`: `99.38%` | `medium (1.5~3.5s)`: `64.80%` | `long (>3.5s)`: `70.19%`
  - Long-vs-Short Delta: `-29.19%` | Degradation Slope: `-14.59% / bucket`
- **Metric ⑧ (Voiced Phoneme Probe Blindness & Lossless Reconstruction R²)**:
  - Emotion Branch `z_emo` Phoneme Probe Acc: `1.00%` (Random Chance Ref: `4.00%`)
  - Content Sponge `z_cont` Phoneme Probe Acc: `100.00%`
  - Dual-Branch Linear Reconstruction `R²`: **`0.9990`** (`> 0.95` Verified: `True`)
