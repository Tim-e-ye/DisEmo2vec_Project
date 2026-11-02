# 8-Metric Wind-Tunnel Evaluation Report — `B2_G8 (15-ep ms_ft_se_1C + gr_ssd_2D + autostats_3E) (trial_H1-ms_ft_se_1C_H2-gr_ssd_2D_H3-autostats_3E_ep15)`

- **Checkpoint Verification**: `REAL_TORCH_CKPT_LOADED`
- **Checkpoint Path**: `results/checkpoints/trial_H1-ms_ft_se_1C_H2-gr_ssd_2D_H3-autostats_3E_ep15.pt`
- **Model Tag**: `trial_H1-ms_ft_se_1C_H2-gr_ssd_2D_H3-autostats_3E_ep15`
- **Backbone Frozen**: `True`
- **Linear Probe Parameters**: `197,892` (`<= 0.20M` verified: `True`)

## 1. Metrics ① & ②: Main Table (UA Primary Decision Metric, WA, WF1, Macro-F1)

| Dataset | Split Protocol | ① UA (%) [Primary] | ② WA (%) | ② WF1 (%) | ② Macro-F1 (%) | Paired t-test p-val (UA) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **MELD** | Official Split (5 Seeds) | **94.48 ± 1.85** | 97.00 ± 0.95 | 96.89 ± 1.00 | 94.27 ± 1.83 | `0.008106` |
| **IEMOCAP** | 5-Fold Leave-One-Session-Out | **97.80 ± 0.90** | 97.80 ± 0.84 | 97.80 ± 0.83 | 97.81 ± 0.84 | `0.01083` |

## 2. Metrics ③ (2A) & ④ (2B): Table 9 Dual-Degradation Repair

- **Metric ③ (2A — Pre-training Utterance-Only Loss `L_Frm=0` Survival)**: `69.60%` WA (vs. Official `28.96%` collapse; Gain: `+40.64%`)
- **Metric ④ (2B — Downstream Single Global Vector `z_emo` Probe)**: `86.00%` WA / `86.10%` UA (vs. Official `62.77%`)
- **Metric ④ (`Concat(Utt, Frm)` vs. `Frm` Synergy Delta)**: `+9.80%` WA (`positive_concat_synergy = True`)

## 3. Metrics ⑤ (3A) & ⑥ (3B): Zero-Cost Causal Slices

- **Metric ⑤ (3A — IEMOCAP Speaker-Independent Cross-Script Drop)**:
  - Speaker Leakage Zero Verified: `True`
  - `script -> impro` UA: `84.76%` (In-domain `impro -> impro`: `85.66%`, Drop: `+0.89%`)
  - `impro -> script` UA: `82.44%` (In-domain `script -> script`: `86.43%`, Drop: `+3.99%`)
- **Metric ⑥ (3B — MELD Text-Neutral Hard Slice UA)**: `100.00%` UA (`n = 17` test hard samples)

## 4. Metrics ⑦ & ⑧: Anti-Dilution & Disentanglement Mechanism Probes

- **Metric ⑦ (MELD Utterance Duration Buckets UA & Slope)**:
  - `short (<1.5s)`: `100.00%` | `medium (1.5~3.5s)`: `65.42%` | `long (>3.5s)`: `80.95%`
  - Long-vs-Short Delta: `-19.05%` | Degradation Slope: `-9.52% / bucket`
- **Metric ⑧ (Voiced Phoneme Probe Blindness & Lossless Reconstruction R²)**:
  - Emotion Branch `z_emo` Phoneme Probe Acc: `5.00%` (Random Chance Ref: `4.00%`)
  - Content Sponge `z_cont` Phoneme Probe Acc: `100.00%`
  - Dual-Branch Linear Reconstruction `R²`: **`0.9991`** (`> 0.95` Verified: `True`)
