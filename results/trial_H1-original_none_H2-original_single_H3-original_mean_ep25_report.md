# 8-Metric Wind-Tunnel Evaluation Report — `Baton-0 / Anchor-2 (89h 25-ep Original Mean) (trial_H1-original_none_H2-original_single_H3-original_mean_ep25)`

- **Checkpoint Verification**: `REAL_TORCH_CKPT_LOADED`
- **Checkpoint Path**: `results/checkpoints/trial_H1-original_none_H2-original_single_H3-original_mean_ep25.pt`
- **Model Tag**: `trial_H1-original_none_H2-original_single_H3-original_mean_ep25`
- **Backbone Frozen**: `True`
- **Linear Probe Parameters**: `197,892` (`<= 0.20M` verified: `True`)

## 1. Metrics ① & ②: Main Table (UA Primary Decision Metric, WA, WF1, Macro-F1)

| Dataset | Split Protocol | ① UA (%) [Primary] | ② WA (%) | ② WF1 (%) | ② Macro-F1 (%) | Paired t-test p-val (UA) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **MELD** | Official Split (5 Seeds) | **45.77 ± 3.35** | 57.00 ± 0.95 | 60.04 ± 0.91 | 40.96 ± 2.79 | `N/A` |
| **IEMOCAP** | 5-Fold Leave-One-Session-Out | **83.39 ± 3.58** | 83.20 ± 3.63 | 83.14 ± 3.66 | 83.26 ± 3.64 | `N/A` |

## 2. Metrics ③ (2A) & ④ (2B): Table 9 Dual-Degradation Repair

- **Metric ③ (2A — Pre-training Utterance-Only Loss `L_Frm=0` Survival)**: `24.40%` WA (vs. Official `28.96%` collapse; Gain: `-4.56%`)
- **Metric ④ (2B — Downstream Single Global Vector `z_emo` Probe)**: `57.00%` WA / `57.23%` UA (vs. Official `62.77%`)
- **Metric ④ (`Concat(Utt, Frm)` vs. `Frm` Synergy Delta)**: `+4.20%` WA (`positive_concat_synergy = True`)

## 3. Metrics ⑤ (3A) & ⑥ (3B): Zero-Cost Causal Slices

- **Metric ⑤ (3A — IEMOCAP Speaker-Independent Cross-Script Drop)**:
  - Speaker Leakage Zero Verified: `True`
  - `script -> impro` UA: `7.86%` (In-domain `impro -> impro`: `85.00%`, Drop: `+77.14%`)
  - `impro -> script` UA: `8.09%` (In-domain `script -> script`: `86.01%`, Drop: `+77.92%`)
- **Metric ⑥ (3B — MELD Text-Neutral Hard Slice UA)**: `36.11%` UA (`n = 17` test hard samples)

## 4. Metrics ⑦ & ⑧: Anti-Dilution & Disentanglement Mechanism Probes

- **Metric ⑦ (MELD Utterance Duration Buckets UA & Slope)**:
  - `short (<1.5s)`: `43.37%` | `medium (1.5~3.5s)`: `34.37%` | `long (>3.5s)`: `13.25%`
  - Long-vs-Short Delta: `-30.12%` | Degradation Slope: `-15.06% / bucket`
- **Metric ⑧ (Voiced Phoneme Probe Blindness & Lossless Reconstruction R²)**:
  - Emotion Branch `z_emo` Phoneme Probe Acc: `29.00%` (Random Chance Ref: `4.00%`)
  - Content Sponge `z_cont` Phoneme Probe Acc: `61.00%`
  - Dual-Branch Linear Reconstruction `R²`: **`0.9983`** (`> 0.95` Verified: `True`)
