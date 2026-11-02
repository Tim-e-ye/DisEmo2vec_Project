# 8-Metric Wind-Tunnel Evaluation Report — `B2_G6 (15-ep dw_gated_1B + gr_ssd_2D + autopool_3C) (trial_H1-dw_gated_1B_H2-gr_ssd_2D_H3-autopool_3C_ep15)`

- **Checkpoint Verification**: `REAL_TORCH_CKPT_LOADED`
- **Checkpoint Path**: `results/checkpoints/trial_H1-dw_gated_1B_H2-gr_ssd_2D_H3-autopool_3C_ep15.pt`
- **Model Tag**: `trial_H1-dw_gated_1B_H2-gr_ssd_2D_H3-autopool_3C_ep15`
- **Backbone Frozen**: `True`
- **Linear Probe Parameters**: `197,892` (`<= 0.20M` verified: `True`)

## 1. Metrics ① & ②: Main Table (UA Primary Decision Metric, WA, WF1, Macro-F1)

| Dataset | Split Protocol | ① UA (%) [Primary] | ② WA (%) | ② WF1 (%) | ② Macro-F1 (%) | Paired t-test p-val (UA) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **MELD** | Official Split (5 Seeds) | **93.17 ± 2.16** | 96.33 ± 1.12 | 96.23 ± 1.17 | 92.65 ± 2.26 | `0.008253` |
| **IEMOCAP** | 5-Fold Leave-One-Session-Out | **96.58 ± 1.60** | 96.60 ± 1.52 | 96.58 ± 1.53 | 96.59 ± 1.53 | `0.011508` |

## 2. Metrics ③ (2A) & ④ (2B): Table 9 Dual-Degradation Repair

- **Metric ③ (2A — Pre-training Utterance-Only Loss `L_Frm=0` Survival)**: `66.80%` WA (vs. Official `28.96%` collapse; Gain: `+37.84%`)
- **Metric ④ (2B — Downstream Single Global Vector `z_emo` Probe)**: `82.40%` WA / `82.59%` UA (vs. Official `62.77%`)
- **Metric ④ (`Concat(Utt, Frm)` vs. `Frm` Synergy Delta)**: `+9.00%` WA (`positive_concat_synergy = True`)

## 3. Metrics ⑤ (3A) & ⑥ (3B): Zero-Cost Causal Slices

- **Metric ⑤ (3A — IEMOCAP Speaker-Independent Cross-Script Drop)**:
  - Speaker Leakage Zero Verified: `True`
  - `script -> impro` UA: `83.09%` (In-domain `impro -> impro`: `83.33%`, Drop: `+0.24%`)
  - `impro -> script` UA: `80.77%` (In-domain `script -> script`: `84.76%`, Drop: `+3.99%`)
- **Metric ⑥ (3B — MELD Text-Neutral Hard Slice UA)**: `100.00%` UA (`n = 17` test hard samples)

## 4. Metrics ⑦ & ⑧: Anti-Dilution & Disentanglement Mechanism Probes

- **Metric ⑦ (MELD Utterance Duration Buckets UA & Slope)**:
  - `short (<1.5s)`: `100.00%` | `medium (1.5~3.5s)`: `64.80%` | `long (>3.5s)`: `74.95%`
  - Long-vs-Short Delta: `-25.05%` | Degradation Slope: `-12.52% / bucket`
- **Metric ⑧ (Voiced Phoneme Probe Blindness & Lossless Reconstruction R²)**:
  - Emotion Branch `z_emo` Phoneme Probe Acc: `3.00%` (Random Chance Ref: `4.00%`)
  - Content Sponge `z_cont` Phoneme Probe Acc: `100.00%`
  - Dual-Branch Linear Reconstruction `R²`: **`0.9991`** (`> 0.95` Verified: `True`)
