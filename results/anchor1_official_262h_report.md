# 8-Metric Wind-Tunnel Evaluation Report — `Anchor-1: Official emotion2vec (262h Baseline)`

- **Model Tag**: `anchor1_official_262h`
- **Backbone Frozen**: `True`
- **Linear Probe Parameters**: `197,892` (`<= 0.20M` verified: `True`)

## 1. Metrics ① & ②: Main Table (UA Primary Decision Metric, WA, WF1, Macro-F1)

| Dataset | Split Protocol | ① UA (%) [Primary] | ② WA (%) | ② WF1 (%) | ② Macro-F1 (%) | Paired t-test p-val (UA) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **MELD** | Official Split (5 Seeds) | **79.24 ± 3.74** | 82.17 ± 2.25 | 83.16 ± 1.99 | 74.46 ± 3.59 | `N/A` |
| **IEMOCAP** | 5-Fold Leave-One-Session-Out | **99.20 ± 0.86** | 99.20 ± 0.84 | 99.20 ± 0.84 | 99.21 ± 0.84 | `N/A` |

## 2. Metrics ③ (2A) & ④ (2B): Table 9 Dual-Degradation Repair

- **Metric ③ (2A — Pre-training Utterance-Only Loss `L_Frm=0` Survival)**: `24.40%` WA (vs. Official `28.96%` collapse; Gain: `-4.56%`)
- **Metric ④ (2B — Downstream Single Global Vector `z_emo` Probe)**: `82.80%` WA / `82.97%` UA (vs. Official `62.77%`)
- **Metric ④ (`Concat(Utt, Frm)` vs. `Frm` Synergy Delta)**: `+0.80%` WA (`positive_concat_synergy = True`)

## 3. Metrics ⑤ (3A) & ⑥ (3B): Zero-Cost Causal Slices

- **Metric ⑤ (3A — IEMOCAP Speaker-Independent Cross-Script Drop)**:
  - Speaker Leakage Zero Verified: `True`
  - `script -> impro` UA: `9.88%` (In-domain `impro -> impro`: `98.04%`, Drop: `+88.15%`)
  - `impro -> script` UA: `7.86%` (In-domain `script -> script`: `99.23%`, Drop: `+91.37%`)
- **Metric ⑥ (3B — MELD Text-Neutral Hard Slice UA)**: `63.89%` UA (`n = 17` test hard samples)

## 4. Metrics ⑦ & ⑧: Anti-Dilution & Disentanglement Mechanism Probes

- **Metric ⑦ (MELD Utterance Duration Buckets UA & Slope)**:
  - `short (<1.5s)`: `69.05%` | `medium (1.5~3.5s)`: `56.52%` | `long (>3.5s)`: `39.54%`
  - Long-vs-Short Delta: `-29.51%` | Degradation Slope: `-14.76% / bucket`
- **Metric ⑧ (Voiced Phoneme Probe Blindness & Lossless Reconstruction R²)**:
  - Emotion Branch `z_emo` Phoneme Probe Acc: `18.00%` (Random Chance Ref: `4.00%`)
  - Content Sponge `z_cont` Phoneme Probe Acc: `53.00%`
  - Dual-Branch Linear Reconstruction `R²`: **`0.9987`** (`> 0.95` Verified: `True`)
