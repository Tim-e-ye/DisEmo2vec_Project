# 8-Metric Wind-Tunnel Evaluation Report — `Anchor-1: Official emotion2vec (262h Baseline)`

- **Checkpoint Verification**: `SMOKE_PIPELINE_TEST`
- **Checkpoint Path**: `NONE (SMOKE_PIPELINE_TEST)`
- **Model Tag**: `anchor1_official_262h`
- **Backbone Frozen**: `True`
- **Linear Probe Parameters**: `197,892` (`<= 0.20M` verified: `True`)

## 1. Metrics ① & ②: Main Table (UA Primary Decision Metric, WA, WF1, Macro-F1)

| Dataset | Split Protocol | ① UA (%) [Primary] | ② WA (%) | ② WF1 (%) | ② Macro-F1 (%) | Paired t-test p-val (UA) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **MELD** | Official Split (5 Seeds) | **48.98 ± 2.96** | 59.17 ± 1.86 | 62.00 ± 1.45 | 43.71 ± 2.65 | `N/A` |
| **IEMOCAP** | 5-Fold Leave-One-Session-Out | **85.77 ± 4.04** | 85.60 ± 4.10 | 85.49 ± 4.20 | 85.56 ± 4.13 | `N/A` |

## 2. Metrics ③ (2A) & ④ (2B): Table 9 Dual-Degradation Repair

- **Metric ③ (2A — Pre-training Utterance-Only Loss `L_Frm=0` Survival)**: `24.40%` WA (vs. Official `28.96%` collapse; Gain: `-4.56%`)
- **Metric ④ (2B — Downstream Single Global Vector `z_emo` Probe)**: `63.00%` WA / `63.18%` UA (vs. Official `62.77%`)
- **Metric ④ (`Concat(Utt, Frm)` vs. `Frm` Synergy Delta)**: `+3.20%` WA (`positive_concat_synergy = True`)

## 3. Metrics ⑤ (3A) & ⑥ (3B): Zero-Cost Causal Slices

- **Metric ⑤ (3A — IEMOCAP Speaker-Independent Cross-Script Drop)**:
  - Speaker Leakage Zero Verified: `True`
  - `script -> impro` UA: `9.29%` (In-domain `impro -> impro`: `81.01%`, Drop: `+71.73%`)
  - `impro -> script` UA: `10.18%` (In-domain `script -> script`: `88.51%`, Drop: `+78.34%`)
- **Metric ⑥ (3B — MELD Text-Neutral Hard Slice UA)**: `41.67%` UA (`n = 17` test hard samples)

## 4. Metrics ⑦ & ⑧: Anti-Dilution & Disentanglement Mechanism Probes

- **Metric ⑦ (MELD Utterance Duration Buckets UA & Slope)**:
  - `short (<1.5s)`: `48.14%` | `medium (1.5~3.5s)`: `34.99%` | `long (>3.5s)`: `13.25%`
  - Long-vs-Short Delta: `-34.89%` | Degradation Slope: `-17.44% / bucket`
- **Metric ⑧ (Voiced Phoneme Probe Blindness & Lossless Reconstruction R²)**:
  - Emotion Branch `z_emo` Phoneme Probe Acc: `22.00%` (Random Chance Ref: `4.00%`)
  - Content Sponge `z_cont` Phoneme Probe Acc: `76.00%`
  - Dual-Branch Linear Reconstruction `R²`: **`0.9983`** (`> 0.95` Verified: `True`)
