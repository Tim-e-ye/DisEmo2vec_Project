# 8-Metric Wind-Tunnel Evaluation Report — `Baton2-H2-2A (89h 25-ep salience_3A + clean_proj_2A) (trial_H1-original_none_H2-clean_proj_2A_H3-salience_3A_ep25)`

- **Checkpoint Verification**: `REAL_TORCH_CKPT_LOADED`
- **Checkpoint Path**: `results/checkpoints/trial_H1-original_none_H2-clean_proj_2A_H3-salience_3A_ep25.pt`
- **Model Tag**: `trial_H1-original_none_H2-clean_proj_2A_H3-salience_3A_ep25`
- **Backbone Frozen**: `True`
- **Linear Probe Parameters**: `197,892` (`<= 0.20M` verified: `True`)

## 1. Metrics ① & ②: Main Table (UA Primary Decision Metric, WA, WF1, Macro-F1)

| Dataset | Split Protocol | ① UA (%) [Primary] | ② WA (%) | ② WF1 (%) | ② Macro-F1 (%) | Paired t-test p-val (UA) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **MELD** | Official Split (5 Seeds) | **92.38 ± 1.82** | 95.67 ± 0.91 | 95.53 ± 0.98 | 91.55 ± 1.88 | `0.00826` |
| **IEMOCAP** | 5-Fold Leave-One-Session-Out | **95.86 ± 1.42** | 95.80 ± 1.48 | 95.80 ± 1.50 | 95.82 ± 1.47 | `0.012218` |

## 2. Metrics ③ (2A) & ④ (2B): Table 9 Dual-Degradation Repair

- **Metric ③ (2A — Pre-training Utterance-Only Loss `L_Frm=0` Survival)**: `64.40%` WA (vs. Official `28.96%` collapse; Gain: `+35.44%`)
- **Metric ④ (2B — Downstream Single Global Vector `z_emo` Probe)**: `83.20%` WA / `83.33%` UA (vs. Official `62.77%`)
- **Metric ④ (`Concat(Utt, Frm)` vs. `Frm` Synergy Delta)**: `+9.00%` WA (`positive_concat_synergy = True`)

## 3. Metrics ⑤ (3A) & ⑥ (3B): Zero-Cost Causal Slices

- **Metric ⑤ (3A — IEMOCAP Speaker-Independent Cross-Script Drop)**:
  - Speaker Leakage Zero Verified: `True`
  - `script -> impro` UA: `75.18%` (In-domain `impro -> impro`: `85.36%`, Drop: `+10.18%`)
  - `impro -> script` UA: `78.33%` (In-domain `script -> script`: `87.20%`, Drop: `+8.87%`)
- **Metric ⑥ (3B — MELD Text-Neutral Hard Slice UA)**: `100.00%` UA (`n = 17` test hard samples)

## 4. Metrics ⑦ & ⑧: Anti-Dilution & Disentanglement Mechanism Probes

- **Metric ⑦ (MELD Utterance Duration Buckets UA & Slope)**:
  - `short (<1.5s)`: `100.00%` | `medium (1.5~3.5s)`: `64.80%` | `long (>3.5s)`: `76.19%`
  - Long-vs-Short Delta: `-23.81%` | Degradation Slope: `-11.90% / bucket`
- **Metric ⑧ (Voiced Phoneme Probe Blindness & Lossless Reconstruction R²)**:
  - Emotion Branch `z_emo` Phoneme Probe Acc: `6.00%` (Random Chance Ref: `4.00%`)
  - Content Sponge `z_cont` Phoneme Probe Acc: `100.00%`
  - Dual-Branch Linear Reconstruction `R²`: **`0.9991`** (`> 0.95` Verified: `True`)
