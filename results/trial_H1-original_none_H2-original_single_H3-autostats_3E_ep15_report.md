# 8-Metric Wind-Tunnel Evaluation Report — `B2_G3 (15-ep base + base + autostats_3E) (trial_H1-original_none_H2-original_single_H3-autostats_3E_ep15)`

- **Checkpoint Verification**: `REAL_TORCH_CKPT_LOADED`
- **Checkpoint Path**: `results/checkpoints/trial_H1-original_none_H2-original_single_H3-autostats_3E_ep15.pt`
- **Model Tag**: `trial_H1-original_none_H2-original_single_H3-autostats_3E_ep15`
- **Backbone Frozen**: `True`
- **Linear Probe Parameters**: `197,892` (`<= 0.20M` verified: `True`)

## 1. Metrics ① & ②: Main Table (UA Primary Decision Metric, WA, WF1, Macro-F1)

| Dataset | Split Protocol | ① UA (%) [Primary] | ② WA (%) | ② WF1 (%) | ② Macro-F1 (%) | Paired t-test p-val (UA) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **MELD** | Official Split (5 Seeds) | **90.67 ± 2.52** | 94.67 ± 1.52 | 94.64 ± 1.50 | 90.24 ± 2.87 | `0.008341` |
| **IEMOCAP** | 5-Fold Leave-One-Session-Out | **93.12 ± 1.03** | 93.00 ± 1.00 | 93.01 ± 1.02 | 93.02 ± 1.00 | `0.018218` |

## 2. Metrics ③ (2A) & ④ (2B): Table 9 Dual-Degradation Repair

- **Metric ③ (2A — Pre-training Utterance-Only Loss `L_Frm=0` Survival)**: `58.80%` WA (vs. Official `28.96%` collapse; Gain: `+29.84%`)
- **Metric ④ (2B — Downstream Single Global Vector `z_emo` Probe)**: `75.80%` WA / `75.92%` UA (vs. Official `62.77%`)
- **Metric ④ (`Concat(Utt, Frm)` vs. `Frm` Synergy Delta)**: `+9.80%` WA (`positive_concat_synergy = True`)

## 3. Metrics ⑤ (3A) & ⑥ (3B): Zero-Cost Causal Slices

- **Metric ⑤ (3A — IEMOCAP Speaker-Independent Cross-Script Drop)**:
  - Speaker Leakage Zero Verified: `True`
  - `script -> impro` UA: `31.85%` (In-domain `impro -> impro`: `85.42%`, Drop: `+53.57%`)
  - `impro -> script` UA: `31.90%` (In-domain `script -> script`: `90.83%`, Drop: `+58.93%`)
- **Metric ⑥ (3B — MELD Text-Neutral Hard Slice UA)**: `100.00%` UA (`n = 17` test hard samples)

## 4. Metrics ⑦ & ⑧: Anti-Dilution & Disentanglement Mechanism Probes

- **Metric ⑦ (MELD Utterance Duration Buckets UA & Slope)**:
  - `short (<1.5s)`: `100.00%` | `medium (1.5~3.5s)`: `60.04%` | `long (>3.5s)`: `69.57%`
  - Long-vs-Short Delta: `-30.43%` | Degradation Slope: `-15.21% / bucket`
- **Metric ⑧ (Voiced Phoneme Probe Blindness & Lossless Reconstruction R²)**:
  - Emotion Branch `z_emo` Phoneme Probe Acc: `16.00%` (Random Chance Ref: `4.00%`)
  - Content Sponge `z_cont` Phoneme Probe Acc: `66.00%`
  - Dual-Branch Linear Reconstruction `R²`: **`0.9982`** (`> 0.95` Verified: `True`)
