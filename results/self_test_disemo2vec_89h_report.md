# 8-Metric Wind-Tunnel Evaluation Report — `DisEmo2vec (89h Candidate + v0 Hammers)`

- **Checkpoint Verification**: `SMOKE_PIPELINE_TEST`
- **Checkpoint Path**: `NONE (SMOKE_PIPELINE_TEST)`
- **Model Tag**: `self_test_disemo2vec_89h`
- **Backbone Frozen**: `True`
- **Linear Probe Parameters**: `197,892` (`<= 0.20M` verified: `True`)

## 1. Metrics ① & ②: Main Table (UA Primary Decision Metric, WA, WF1, Macro-F1)

| Dataset | Split Protocol | ① UA (%) [Primary] | ② WA (%) | ② WF1 (%) | ② Macro-F1 (%) | Paired t-test p-val (UA) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **MELD** | Official Split (5 Seeds) | **94.48 ± 1.81** | 97.00 ± 0.95 | 96.92 ± 1.05 | 94.01 ± 2.00 | `0.008311` |
| **IEMOCAP** | 5-Fold Leave-One-Session-Out | **97.41 ± 1.18** | 97.40 ± 1.14 | 97.40 ± 1.14 | 97.41 ± 1.14 | `0.014003` |

## 2. Metrics ③ (2A) & ④ (2B): Table 9 Dual-Degradation Repair

- **Metric ③ (2A — Pre-training Utterance-Only Loss `L_Frm=0` Survival)**: `71.00%` WA (vs. Official `28.96%` collapse; Gain: `+42.04%`)
- **Metric ④ (2B — Downstream Single Global Vector `z_emo` Probe)**: `86.00%` WA / `86.16%` UA (vs. Official `62.77%`)
- **Metric ④ (`Concat(Utt, Frm)` vs. `Frm` Synergy Delta)**: `+9.20%` WA (`positive_concat_synergy = True`)

## 3. Metrics ⑤ (3A) & ⑥ (3B): Zero-Cost Causal Slices

- **Metric ⑤ (3A — IEMOCAP Speaker-Independent Cross-Script Drop)**:
  - Speaker Leakage Zero Verified: `True`
  - `script -> impro` UA: `83.81%` (In-domain `impro -> impro`: `84.47%`, Drop: `+0.66%`)
  - `impro -> script` UA: `83.21%` (In-domain `script -> script`: `85.66%`, Drop: `+2.44%`)
- **Metric ⑥ (3B — MELD Text-Neutral Hard Slice UA)**: `100.00%` UA (`n = 17` test hard samples)

## 4. Metrics ⑦ & ⑧: Anti-Dilution & Disentanglement Mechanism Probes

- **Metric ⑦ (MELD Utterance Duration Buckets UA & Slope)**:
  - `short (<1.5s)`: `100.00%` | `medium (1.5~3.5s)`: `69.57%` | `long (>3.5s)`: `76.19%`
  - Long-vs-Short Delta: `-23.81%` | Degradation Slope: `-11.91% / bucket`
- **Metric ⑧ (Voiced Phoneme Probe Blindness & Lossless Reconstruction R²)**:
  - Emotion Branch `z_emo` Phoneme Probe Acc: `3.00%` (Random Chance Ref: `4.00%`)
  - Content Sponge `z_cont` Phoneme Probe Acc: `100.00%`
  - Dual-Branch Linear Reconstruction `R²`: **`0.9990`** (`> 0.95` Verified: `True`)
