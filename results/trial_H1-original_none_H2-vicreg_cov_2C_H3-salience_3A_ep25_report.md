# 8-Metric Wind-Tunnel Evaluation Report — `Baton2-H2-2C (89h 25-ep salience_3A + vicreg_cov_2C) (trial_H1-original_none_H2-vicreg_cov_2C_H3-salience_3A_ep25)`

- **Checkpoint Verification**: `REAL_TORCH_CKPT_LOADED`
- **Checkpoint Path**: `results/checkpoints/trial_H1-original_none_H2-vicreg_cov_2C_H3-salience_3A_ep25.pt`
- **Model Tag**: `trial_H1-original_none_H2-vicreg_cov_2C_H3-salience_3A_ep25`
- **Backbone Frozen**: `True`
- **Linear Probe Parameters**: `197,892` (`<= 0.20M` verified: `True`)

## 1. Metrics ① & ②: Main Table (UA Primary Decision Metric, WA, WF1, Macro-F1)

| Dataset | Split Protocol | ① UA (%) [Primary] | ② WA (%) | ② WF1 (%) | ② Macro-F1 (%) | Paired t-test p-val (UA) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **MELD** | Official Split (5 Seeds) | **91.74 ± 3.33** | 95.17 ± 1.90 | 95.04 ± 1.98 | 90.87 ± 3.49 | `0.008269` |
| **IEMOCAP** | 5-Fold Leave-One-Session-Out | **96.07 ± 1.88** | 96.00 ± 1.87 | 95.98 ± 1.88 | 96.01 ± 1.88 | `0.011629` |

## 2. Metrics ③ (2A) & ④ (2B): Table 9 Dual-Degradation Repair

- **Metric ③ (2A — Pre-training Utterance-Only Loss `L_Frm=0` Survival)**: `64.80%` WA (vs. Official `28.96%` collapse; Gain: `+35.84%`)
- **Metric ④ (2B — Downstream Single Global Vector `z_emo` Probe)**: `83.20%` WA / `83.36%` UA (vs. Official `62.77%`)
- **Metric ④ (`Concat(Utt, Frm)` vs. `Frm` Synergy Delta)**: `+10.00%` WA (`positive_concat_synergy = True`)

## 3. Metrics ⑤ (3A) & ⑥ (3B): Zero-Cost Causal Slices

- **Metric ⑤ (3A — IEMOCAP Speaker-Independent Cross-Script Drop)**:
  - Speaker Leakage Zero Verified: `True`
  - `script -> impro` UA: `73.51%` (In-domain `impro -> impro`: `84.52%`, Drop: `+11.01%`)
  - `impro -> script` UA: `74.46%` (In-domain `script -> script`: `87.20%`, Drop: `+12.74%`)
- **Metric ⑥ (3B — MELD Text-Neutral Hard Slice UA)**: `100.00%` UA (`n = 17` test hard samples)

## 4. Metrics ⑦ & ⑧: Anti-Dilution & Disentanglement Mechanism Probes

- **Metric ⑦ (MELD Utterance Duration Buckets UA & Slope)**:
  - `short (<1.5s)`: `100.00%` | `medium (1.5~3.5s)`: `64.80%` | `long (>3.5s)`: `75.57%`
  - Long-vs-Short Delta: `-24.43%` | Degradation Slope: `-12.21% / bucket`
- **Metric ⑧ (Voiced Phoneme Probe Blindness & Lossless Reconstruction R²)**:
  - Emotion Branch `z_emo` Phoneme Probe Acc: `4.00%` (Random Chance Ref: `4.00%`)
  - Content Sponge `z_cont` Phoneme Probe Acc: `96.00%`
  - Dual-Branch Linear Reconstruction `R²`: **`0.9990`** (`> 0.95` Verified: `True`)
