# 8-Metric Wind-Tunnel Evaluation Report — `DisEmo2vec (89h Candidate + v0 Hammers)`

- **Checkpoint Verification**: `SMOKE_PIPELINE_TEST`
- **Checkpoint Path**: `NONE (SMOKE_PIPELINE_TEST)`
- **Model Tag**: `self_test_disemo2vec_89h`
- **Backbone Frozen**: `True`
- **Linear Probe Parameters**: `197,892` (`<= 0.20M` verified: `True`)

## 1. Metrics ① & ②: Main Table (UA Primary Decision Metric, WA, WF1, Macro-F1)

| Dataset | Split Protocol | ① UA (%) [Primary] | ② WA (%) | ② WF1 (%) | ② Macro-F1 (%) | Paired t-test p-val (UA) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **MELD** | Official Split (5 Seeds) | **93.81 ± 2.06** | 96.67 ± 1.02 | 96.59 ± 1.00 | 93.32 ± 2.01 | `0.008491` |
| **IEMOCAP** | 5-Fold Leave-One-Session-Out | **95.86 ± 1.93** | 95.80 ± 1.92 | 95.77 ± 1.94 | 95.78 ± 1.96 | `0.009641` |

## 2. Metrics ③ (2A) & ④ (2B): Table 9 Dual-Degradation Repair

- **Metric ③ (2A — Pre-training Utterance-Only Loss `L_Frm=0` Survival)**: `66.40%` WA (vs. Official `28.96%` collapse; Gain: `+37.44%`)
- **Metric ④ (2B — Downstream Single Global Vector `z_emo` Probe)**: `84.40%` WA / `84.56%` UA (vs. Official `62.77%`)
- **Metric ④ (`Concat(Utt, Frm)` vs. `Frm` Synergy Delta)**: `+8.40%` WA (`positive_concat_synergy = True`)

## 3. Metrics ⑤ (3A) & ⑥ (3B): Zero-Cost Causal Slices

- **Metric ⑤ (3A — IEMOCAP Speaker-Independent Cross-Script Drop)**:
  - Speaker Leakage Zero Verified: `True`
  - `script -> impro` UA: `79.58%` (In-domain `impro -> impro`: `83.69%`, Drop: `+4.11%`)
  - `impro -> script` UA: `80.83%` (In-domain `script -> script`: `87.20%`, Drop: `+6.37%`)
- **Metric ⑥ (3B — MELD Text-Neutral Hard Slice UA)**: `100.00%` UA (`n = 17` test hard samples)

## 4. Metrics ⑦ & ⑧: Anti-Dilution & Disentanglement Mechanism Probes

- **Metric ⑦ (MELD Utterance Duration Buckets UA & Slope)**:
  - `short (<1.5s)`: `99.38%` | `medium (1.5~3.5s)`: `64.80%` | `long (>3.5s)`: `76.19%`
  - Long-vs-Short Delta: `-23.19%` | Degradation Slope: `-11.59% / bucket`
- **Metric ⑧ (Voiced Phoneme Probe Blindness & Lossless Reconstruction R²)**:
  - Emotion Branch `z_emo` Phoneme Probe Acc: `1.00%` (Random Chance Ref: `4.00%`)
  - Content Sponge `z_cont` Phoneme Probe Acc: `98.00%`
  - Dual-Branch Linear Reconstruction `R²`: **`0.9991`** (`> 0.95` Verified: `True`)
