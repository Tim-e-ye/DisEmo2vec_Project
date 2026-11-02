# 8-Metric Wind-Tunnel Evaluation Report — `H3-PK-3C (89h 25-ep Slot-H3 autopool_3C) (trial_H1-original_none_H2-original_single_H3-autopool_3C_ep25)`

- **Checkpoint Verification**: `REAL_TORCH_CKPT_LOADED`
- **Checkpoint Path**: `results/checkpoints/trial_H1-original_none_H2-original_single_H3-autopool_3C_ep25.pt`
- **Model Tag**: `trial_H1-original_none_H2-original_single_H3-autopool_3C_ep25`
- **Backbone Frozen**: `True`
- **Linear Probe Parameters**: `197,892` (`<= 0.20M` verified: `True`)

## 1. Metrics ① & ②: Main Table (UA Primary Decision Metric, WA, WF1, Macro-F1)

| Dataset | Split Protocol | ① UA (%) [Primary] | ② WA (%) | ② WF1 (%) | ② Macro-F1 (%) | Paired t-test p-val (UA) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **MELD** | Official Split (5 Seeds) | **86.58 ± 1.16** | 92.50 ± 1.02 | 92.43 ± 1.08 | 85.73 ± 1.79 | `0.008451` |
| **IEMOCAP** | 5-Fold Leave-One-Session-Out | **90.72 ± 3.27** | 90.60 ± 3.21 | 90.60 ± 3.19 | 90.65 ± 3.22 | `0.022732` |

## 2. Metrics ③ (2A) & ④ (2B): Table 9 Dual-Degradation Repair

- **Metric ③ (2A — Pre-training Utterance-Only Loss `L_Frm=0` Survival)**: `51.00%` WA (vs. Official `28.96%` collapse; Gain: `+22.04%`)
- **Metric ④ (2B — Downstream Single Global Vector `z_emo` Probe)**: `74.00%` WA / `74.23%` UA (vs. Official `62.77%`)
- **Metric ④ (`Concat(Utt, Frm)` vs. `Frm` Synergy Delta)**: `+8.00%` WA (`positive_concat_synergy = True`)

## 3. Metrics ⑤ (3A) & ⑥ (3B): Zero-Cost Causal Slices

- **Metric ⑤ (3A — IEMOCAP Speaker-Independent Cross-Script Drop)**:
  - Speaker Leakage Zero Verified: `True`
  - `script -> impro` UA: `27.14%` (In-domain `impro -> impro`: `86.31%`, Drop: `+59.17%`)
  - `impro -> script` UA: `25.24%` (In-domain `script -> script`: `91.37%`, Drop: `+66.13%`)
- **Metric ⑥ (3B — MELD Text-Neutral Hard Slice UA)**: `80.56%` UA (`n = 17` test hard samples)

## 4. Metrics ⑦ & ⑧: Anti-Dilution & Disentanglement Mechanism Probes

- **Metric ⑦ (MELD Utterance Duration Buckets UA & Slope)**:
  - `short (<1.5s)`: `82.71%` | `medium (1.5~3.5s)`: `60.04%` | `long (>3.5s)`: `68.94%`
  - Long-vs-Short Delta: `-13.77%` | Degradation Slope: `-6.89% / bucket`
- **Metric ⑧ (Voiced Phoneme Probe Blindness & Lossless Reconstruction R²)**:
  - Emotion Branch `z_emo` Phoneme Probe Acc: `19.00%` (Random Chance Ref: `4.00%`)
  - Content Sponge `z_cont` Phoneme Probe Acc: `73.00%`
  - Dual-Branch Linear Reconstruction `R²`: **`0.9983`** (`> 0.95` Verified: `True`)
