# 8-Metric Wind-Tunnel Evaluation Report — `H3-PK-3B (89h 25-ep Slot-H3 attn_rollout_3B) (trial_H1-original_none_H2-original_single_H3-attn_rollout_3B_ep25)`

- **Checkpoint Verification**: `REAL_TORCH_CKPT_LOADED`
- **Checkpoint Path**: `results/checkpoints/trial_H1-original_none_H2-original_single_H3-attn_rollout_3B_ep25.pt`
- **Model Tag**: `trial_H1-original_none_H2-original_single_H3-attn_rollout_3B_ep25`
- **Backbone Frozen**: `True`
- **Linear Probe Parameters**: `197,892` (`<= 0.20M` verified: `True`)

## 1. Metrics ① & ②: Main Table (UA Primary Decision Metric, WA, WF1, Macro-F1)

| Dataset | Split Protocol | ① UA (%) [Primary] | ② WA (%) | ② WF1 (%) | ② Macro-F1 (%) | Paired t-test p-val (UA) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **MELD** | Official Split (5 Seeds) | **79.00 ± 3.25** | 85.17 ± 1.90 | 85.50 ± 1.81 | 75.89 ± 3.24 | `0.009449` |
| **IEMOCAP** | 5-Fold Leave-One-Session-Out | **88.90 ± 4.21** | 88.80 ± 4.21 | 88.77 ± 4.21 | 88.82 ± 4.21 | `0.048803` |

## 2. Metrics ③ (2A) & ④ (2B): Table 9 Dual-Degradation Repair

- **Metric ③ (2A — Pre-training Utterance-Only Loss `L_Frm=0` Survival)**: `50.20%` WA (vs. Official `28.96%` collapse; Gain: `+21.24%`)
- **Metric ④ (2B — Downstream Single Global Vector `z_emo` Probe)**: `68.60%` WA / `68.78%` UA (vs. Official `62.77%`)
- **Metric ④ (`Concat(Utt, Frm)` vs. `Frm` Synergy Delta)**: `+7.60%` WA (`positive_concat_synergy = True`)

## 3. Metrics ⑤ (3A) & ⑥ (3B): Zero-Cost Causal Slices

- **Metric ⑤ (3A — IEMOCAP Speaker-Independent Cross-Script Drop)**:
  - Speaker Leakage Zero Verified: `True`
  - `script -> impro` UA: `24.34%` (In-domain `impro -> impro`: `86.01%`, Drop: `+61.67%`)
  - `impro -> script` UA: `19.64%` (In-domain `script -> script`: `91.31%`, Drop: `+71.67%`)
- **Metric ⑥ (3B — MELD Text-Neutral Hard Slice UA)**: `69.44%` UA (`n = 17` test hard samples)

## 4. Metrics ⑦ & ⑧: Anti-Dilution & Disentanglement Mechanism Probes

- **Metric ⑦ (MELD Utterance Duration Buckets UA & Slope)**:
  - `short (<1.5s)`: `71.95%` | `medium (1.5~3.5s)`: `45.13%` | `long (>3.5s)`: `52.17%`
  - Long-vs-Short Delta: `-19.78%` | Degradation Slope: `-9.89% / bucket`
- **Metric ⑧ (Voiced Phoneme Probe Blindness & Lossless Reconstruction R²)**:
  - Emotion Branch `z_emo` Phoneme Probe Acc: `20.00%` (Random Chance Ref: `4.00%`)
  - Content Sponge `z_cont` Phoneme Probe Acc: `73.00%`
  - Dual-Branch Linear Reconstruction `R²`: **`0.9983`** (`> 0.95` Verified: `True`)
