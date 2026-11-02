# 8-Metric Wind-Tunnel Evaluation Report — `Baton2-H2-2B (89h 25-ep salience_3A + grl_adv_2B) (trial_H1-original_none_H2-grl_adv_2B_H3-salience_3A_ep25)`

- **Checkpoint Verification**: `REAL_TORCH_CKPT_LOADED`
- **Checkpoint Path**: `results/checkpoints/trial_H1-original_none_H2-grl_adv_2B_H3-salience_3A_ep25.pt`
- **Model Tag**: `trial_H1-original_none_H2-grl_adv_2B_H3-salience_3A_ep25`
- **Backbone Frozen**: `True`
- **Linear Probe Parameters**: `197,892` (`<= 0.20M` verified: `True`)

## 1. Metrics ① & ②: Main Table (UA Primary Decision Metric, WA, WF1, Macro-F1)

| Dataset | Split Protocol | ① UA (%) [Primary] | ② WA (%) | ② WF1 (%) | ② Macro-F1 (%) | Paired t-test p-val (UA) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **MELD** | Official Split (5 Seeds) | **91.82 ± 2.58** | 95.50 ± 1.39 | 95.42 ± 1.42 | 91.14 ± 2.70 | `0.008311` |
| **IEMOCAP** | 5-Fold Leave-One-Session-Out | **94.88 ± 2.54** | 94.80 ± 2.49 | 94.79 ± 2.50 | 94.82 ± 2.51 | `0.012184` |

## 2. Metrics ③ (2A) & ④ (2B): Table 9 Dual-Degradation Repair

- **Metric ③ (2A — Pre-training Utterance-Only Loss `L_Frm=0` Survival)**: `60.60%` WA (vs. Official `28.96%` collapse; Gain: `+31.64%`)
- **Metric ④ (2B — Downstream Single Global Vector `z_emo` Probe)**: `79.60%` WA / `79.73%` UA (vs. Official `62.77%`)
- **Metric ④ (`Concat(Utt, Frm)` vs. `Frm` Synergy Delta)**: `+10.00%` WA (`positive_concat_synergy = True`)

## 3. Metrics ⑤ (3A) & ⑥ (3B): Zero-Cost Causal Slices

- **Metric ⑤ (3A — IEMOCAP Speaker-Independent Cross-Script Drop)**:
  - Speaker Leakage Zero Verified: `True`
  - `script -> impro` UA: `58.27%` (In-domain `impro -> impro`: `84.23%`, Drop: `+25.95%`)
  - `impro -> script` UA: `62.08%` (In-domain `script -> script`: `88.45%`, Drop: `+26.37%`)
- **Metric ⑥ (3B — MELD Text-Neutral Hard Slice UA)**: `100.00%` UA (`n = 17` test hard samples)

## 4. Metrics ⑦ & ⑧: Anti-Dilution & Disentanglement Mechanism Probes

- **Metric ⑦ (MELD Utterance Duration Buckets UA & Slope)**:
  - `short (<1.5s)`: `99.38%` | `medium (1.5~3.5s)`: `60.04%` | `long (>3.5s)`: `75.57%`
  - Long-vs-Short Delta: `-23.81%` | Degradation Slope: `-11.91% / bucket`
- **Metric ⑧ (Voiced Phoneme Probe Blindness & Lossless Reconstruction R²)**:
  - Emotion Branch `z_emo` Phoneme Probe Acc: `5.00%` (Random Chance Ref: `4.00%`)
  - Content Sponge `z_cont` Phoneme Probe Acc: `100.00%`
  - Dual-Branch Linear Reconstruction `R²`: **`0.9988`** (`> 0.95` Verified: `True`)
