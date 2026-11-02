# 8-Metric Wind-Tunnel Evaluation Report — `Baton-1 (89h 25-ep Slot-H3 salience_3A) (trial_H1-original_none_H2-original_single_H3-salience_3A_ep25)`

- **Checkpoint Verification**: `REAL_TORCH_CKPT_LOADED`
- **Checkpoint Path**: `results/checkpoints/trial_H1-original_none_H2-original_single_H3-salience_3A_ep25.pt`
- **Model Tag**: `trial_H1-original_none_H2-original_single_H3-salience_3A_ep25`
- **Backbone Frozen**: `True`
- **Linear Probe Parameters**: `197,892` (`<= 0.20M` verified: `True`)

## 1. Metrics ① & ②: Main Table (UA Primary Decision Metric, WA, WF1, Macro-F1)

| Dataset | Split Protocol | ① UA (%) [Primary] | ② WA (%) | ② WF1 (%) | ② Macro-F1 (%) | Paired t-test p-val (UA) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **MELD** | Official Split (5 Seeds) | **88.96 ± 1.41** | 93.83 ± 1.12 | 93.72 ± 1.20 | 88.17 ± 1.90 | `0.008325` |
| **IEMOCAP** | 5-Fold Leave-One-Session-Out | **92.56 ± 2.64** | 92.40 ± 2.88 | 92.35 ± 2.94 | 92.42 ± 2.85 | `0.011718` |

## 2. Metrics ③ (2A) & ④ (2B): Table 9 Dual-Degradation Repair

- **Metric ③ (2A — Pre-training Utterance-Only Loss `L_Frm=0` Survival)**: `59.60%` WA (vs. Official `28.96%` collapse; Gain: `+30.64%`)
- **Metric ④ (2B — Downstream Single Global Vector `z_emo` Probe)**: `77.80%` WA / `78.00%` UA (vs. Official `62.77%`)
- **Metric ④ (`Concat(Utt, Frm)` vs. `Frm` Synergy Delta)**: `+8.60%` WA (`positive_concat_synergy = True`)

## 3. Metrics ⑤ (3A) & ⑥ (3B): Zero-Cost Causal Slices

- **Metric ⑤ (3A — IEMOCAP Speaker-Independent Cross-Script Drop)**:
  - Speaker Leakage Zero Verified: `True`
  - `script -> impro` UA: `32.62%` (In-domain `impro -> impro`: `87.09%`, Drop: `+54.47%`)
  - `impro -> script` UA: `27.50%` (In-domain `script -> script`: `90.89%`, Drop: `+63.39%`)
- **Metric ⑥ (3B — MELD Text-Neutral Hard Slice UA)**: `100.00%` UA (`n = 17` test hard samples)

## 4. Metrics ⑦ & ⑧: Anti-Dilution & Disentanglement Mechanism Probes

- **Metric ⑦ (MELD Utterance Duration Buckets UA & Slope)**:
  - `short (<1.5s)`: `99.38%` | `medium (1.5~3.5s)`: `60.04%` | `long (>3.5s)`: `70.19%`
  - Long-vs-Short Delta: `-29.19%` | Degradation Slope: `-14.60% / bucket`
- **Metric ⑧ (Voiced Phoneme Probe Blindness & Lossless Reconstruction R²)**:
  - Emotion Branch `z_emo` Phoneme Probe Acc: `12.00%` (Random Chance Ref: `4.00%`)
  - Content Sponge `z_cont` Phoneme Probe Acc: `76.00%`
  - Dual-Branch Linear Reconstruction `R²`: **`0.9982`** (`> 0.95` Verified: `True`)
