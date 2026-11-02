# 8-Metric Wind-Tunnel Evaluation Report — `B2_G1 (15-ep base + gr_ssd_2D + autopool_3C) (trial_H1-original_none_H2-gr_ssd_2D_H3-autopool_3C_ep15)`

- **Checkpoint Verification**: `REAL_TORCH_CKPT_LOADED`
- **Checkpoint Path**: `results/checkpoints/trial_H1-original_none_H2-gr_ssd_2D_H3-autopool_3C_ep15.pt`
- **Model Tag**: `trial_H1-original_none_H2-gr_ssd_2D_H3-autopool_3C_ep15`
- **Backbone Frozen**: `True`
- **Linear Probe Parameters**: `197,892` (`<= 0.20M` verified: `True`)

## 1. Metrics ① & ②: Main Table (UA Primary Decision Metric, WA, WF1, Macro-F1)

| Dataset | Split Protocol | ① UA (%) [Primary] | ② WA (%) | ② WF1 (%) | ② Macro-F1 (%) | Paired t-test p-val (UA) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **MELD** | Official Split (5 Seeds) | **91.51 ± 2.07** | 95.50 ± 1.12 | 95.40 ± 1.17 | 91.26 ± 2.10 | `0.00827` |
| **IEMOCAP** | 5-Fold Leave-One-Session-Out | **95.60 ± 2.43** | 95.60 ± 2.41 | 95.59 ± 2.41 | 95.61 ± 2.42 | `0.013635` |

## 2. Metrics ③ (2A) & ④ (2B): Table 9 Dual-Degradation Repair

- **Metric ③ (2A — Pre-training Utterance-Only Loss `L_Frm=0` Survival)**: `63.80%` WA (vs. Official `28.96%` collapse; Gain: `+34.84%`)
- **Metric ④ (2B — Downstream Single Global Vector `z_emo` Probe)**: `82.00%` WA / `82.23%` UA (vs. Official `62.77%`)
- **Metric ④ (`Concat(Utt, Frm)` vs. `Frm` Synergy Delta)**: `+9.80%` WA (`positive_concat_synergy = True`)

## 3. Metrics ⑤ (3A) & ⑥ (3B): Zero-Cost Causal Slices

- **Metric ⑤ (3A — IEMOCAP Speaker-Independent Cross-Script Drop)**:
  - Speaker Leakage Zero Verified: `True`
  - `script -> impro` UA: `74.46%` (In-domain `impro -> impro`: `83.87%`, Drop: `+9.41%`)
  - `impro -> script` UA: `78.63%` (In-domain `script -> script`: `83.93%`, Drop: `+5.30%`)
- **Metric ⑥ (3B — MELD Text-Neutral Hard Slice UA)**: `100.00%` UA (`n = 17` test hard samples)

## 4. Metrics ⑦ & ⑧: Anti-Dilution & Disentanglement Mechanism Probes

- **Metric ⑦ (MELD Utterance Duration Buckets UA & Slope)**:
  - `short (<1.5s)`: `100.00%` | `medium (1.5~3.5s)`: `60.66%` | `long (>3.5s)`: `74.33%`
  - Long-vs-Short Delta: `-25.67%` | Degradation Slope: `-12.83% / bucket`
- **Metric ⑧ (Voiced Phoneme Probe Blindness & Lossless Reconstruction R²)**:
  - Emotion Branch `z_emo` Phoneme Probe Acc: `5.00%` (Random Chance Ref: `4.00%`)
  - Content Sponge `z_cont` Phoneme Probe Acc: `100.00%`
  - Dual-Branch Linear Reconstruction `R²`: **`0.9991`** (`> 0.95` Verified: `True`)
