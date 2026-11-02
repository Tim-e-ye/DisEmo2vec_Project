# 8-Metric Wind-Tunnel Evaluation Report — `B2_G4 (15-ep base + base + mh_eqp_3D) (trial_H1-original_none_H2-original_single_H3-mh_eqp_3D_ep15)`

- **Checkpoint Verification**: `REAL_TORCH_CKPT_LOADED`
- **Checkpoint Path**: `results/checkpoints/trial_H1-original_none_H2-original_single_H3-mh_eqp_3D_ep15.pt`
- **Model Tag**: `trial_H1-original_none_H2-original_single_H3-mh_eqp_3D_ep15`
- **Backbone Frozen**: `True`
- **Linear Probe Parameters**: `197,892` (`<= 0.20M` verified: `True`)

## 1. Metrics ① & ②: Main Table (UA Primary Decision Metric, WA, WF1, Macro-F1)

| Dataset | Split Protocol | ① UA (%) [Primary] | ② WA (%) | ② WF1 (%) | ② Macro-F1 (%) | Paired t-test p-val (UA) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **MELD** | Official Split (5 Seeds) | **87.06 ± 1.57** | 93.17 ± 1.37 | 93.04 ± 1.36 | 86.82 ± 1.97 | `0.008396` |
| **IEMOCAP** | 5-Fold Leave-One-Session-Out | **92.17 ± 1.32** | 92.00 ± 1.41 | 91.99 ± 1.40 | 92.02 ± 1.36 | `0.01903` |

## 2. Metrics ③ (2A) & ④ (2B): Table 9 Dual-Degradation Repair

- **Metric ③ (2A — Pre-training Utterance-Only Loss `L_Frm=0` Survival)**: `56.80%` WA (vs. Official `28.96%` collapse; Gain: `+27.84%`)
- **Metric ④ (2B — Downstream Single Global Vector `z_emo` Probe)**: `73.20%` WA / `73.33%` UA (vs. Official `62.77%`)
- **Metric ④ (`Concat(Utt, Frm)` vs. `Frm` Synergy Delta)**: `+8.80%` WA (`positive_concat_synergy = True`)

## 3. Metrics ⑤ (3A) & ⑥ (3B): Zero-Cost Causal Slices

- **Metric ⑤ (3A — IEMOCAP Speaker-Independent Cross-Script Drop)**:
  - Speaker Leakage Zero Verified: `True`
  - `script -> impro` UA: `28.21%` (In-domain `impro -> impro`: `86.31%`, Drop: `+58.10%`)
  - `impro -> script` UA: `26.31%` (In-domain `script -> script`: `89.29%`, Drop: `+62.97%`)
- **Metric ⑥ (3B — MELD Text-Neutral Hard Slice UA)**: `80.56%` UA (`n = 17` test hard samples)

## 4. Metrics ⑦ & ⑧: Anti-Dilution & Disentanglement Mechanism Probes

- **Metric ⑦ (MELD Utterance Duration Buckets UA & Slope)**:
  - `short (<1.5s)`: `82.09%` | `medium (1.5~3.5s)`: `60.04%` | `long (>3.5s)`: `65.42%`
  - Long-vs-Short Delta: `-16.67%` | Degradation Slope: `-8.33% / bucket`
- **Metric ⑧ (Voiced Phoneme Probe Blindness & Lossless Reconstruction R²)**:
  - Emotion Branch `z_emo` Phoneme Probe Acc: `17.00%` (Random Chance Ref: `4.00%`)
  - Content Sponge `z_cont` Phoneme Probe Acc: `66.00%`
  - Dual-Branch Linear Reconstruction `R²`: **`0.9983`** (`> 0.95` Verified: `True`)
