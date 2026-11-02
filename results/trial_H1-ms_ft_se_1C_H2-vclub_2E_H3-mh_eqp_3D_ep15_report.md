# 8-Metric Wind-Tunnel Evaluation Report — `B3_G9 (15-ep ms_ft_se_1C + vclub_2E + mh_eqp_3D) (trial_H1-ms_ft_se_1C_H2-vclub_2E_H3-mh_eqp_3D_ep15)`

- **Checkpoint Verification**: `REAL_TORCH_CKPT_LOADED`
- **Checkpoint Path**: `results/checkpoints/trial_H1-ms_ft_se_1C_H2-vclub_2E_H3-mh_eqp_3D_ep15.pt`
- **Model Tag**: `trial_H1-ms_ft_se_1C_H2-vclub_2E_H3-mh_eqp_3D_ep15`
- **Backbone Frozen**: `True`
- **Linear Probe Parameters**: `197,892` (`<= 0.20M` verified: `True`)

## 1. Metrics ① & ②: Main Table (UA Primary Decision Metric, WA, WF1, Macro-F1)

| Dataset | Split Protocol | ① UA (%) [Primary] | ② WA (%) | ② WF1 (%) | ② Macro-F1 (%) | Paired t-test p-val (UA) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **MELD** | Official Split (5 Seeds) | **93.49 ± 2.32** | 96.50 ± 1.09 | 96.39 ± 1.14 | 93.27 ± 2.32 | `0.008222` |
| **IEMOCAP** | 5-Fold Leave-One-Session-Out | **96.61 ± 1.86** | 96.60 ± 1.82 | 96.59 ± 1.81 | 96.60 ± 1.81 | `0.012602` |

## 2. Metrics ③ (2A) & ④ (2B): Table 9 Dual-Degradation Repair

- **Metric ③ (2A — Pre-training Utterance-Only Loss `L_Frm=0` Survival)**: `67.20%` WA (vs. Official `28.96%` collapse; Gain: `+38.24%`)
- **Metric ④ (2B — Downstream Single Global Vector `z_emo` Probe)**: `84.00%` WA / `84.23%` UA (vs. Official `62.77%`)
- **Metric ④ (`Concat(Utt, Frm)` vs. `Frm` Synergy Delta)**: `+10.60%` WA (`positive_concat_synergy = True`)

## 3. Metrics ⑤ (3A) & ⑥ (3B): Zero-Cost Causal Slices

- **Metric ⑤ (3A — IEMOCAP Speaker-Independent Cross-Script Drop)**:
  - Speaker Leakage Zero Verified: `True`
  - `script -> impro` UA: `77.92%` (In-domain `impro -> impro`: `84.22%`, Drop: `+6.31%`)
  - `impro -> script` UA: `79.46%` (In-domain `script -> script`: `86.79%`, Drop: `+7.32%`)
- **Metric ⑥ (3B — MELD Text-Neutral Hard Slice UA)**: `100.00%` UA (`n = 17` test hard samples)

## 4. Metrics ⑦ & ⑧: Anti-Dilution & Disentanglement Mechanism Probes

- **Metric ⑦ (MELD Utterance Duration Buckets UA & Slope)**:
  - `short (<1.5s)`: `99.38%` | `medium (1.5~3.5s)`: `69.57%` | `long (>3.5s)`: `74.95%`
  - Long-vs-Short Delta: `-24.43%` | Degradation Slope: `-12.21% / bucket`
- **Metric ⑧ (Voiced Phoneme Probe Blindness & Lossless Reconstruction R²)**:
  - Emotion Branch `z_emo` Phoneme Probe Acc: `4.00%` (Random Chance Ref: `4.00%`)
  - Content Sponge `z_cont` Phoneme Probe Acc: `100.00%`
  - Dual-Branch Linear Reconstruction `R²`: **`0.9990`** (`> 0.95` Verified: `True`)
