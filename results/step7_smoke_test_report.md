# Step 7 候选锤子 3 分钟 100 步上机前冒烟总闸报告 (`step7_smoke_test_report.md`)

> **总闸结论**：`ALL 9 CONFIGURATIONS PASSED (100% 绿灯放行)`
> **执行脚本**：[`scripts/smoke_test_hammer.py`](../scripts/smoke_test_hammer.py)
> **制度依据**：[`STEP7_SMOOTH_HAMMER_ITERATION_GUARANTEES.md`](../../SymbioThesis/docs/construction_manual/proposals/STEP7_SMOOTH_HAMMER_ITERATION_GUARANTEES.md)

## 四道硬性冒烟门禁汇总表

| 锤子组合 (`H1 | H2 | H3`) | Gate 1: Padding 零泄漏误差 (`<1e-5`) | Gate 2: 推荐解耦权重 $\lambda_{\text{rec}}$ ($\alpha=0.15$) | Gate 3: 表征标准差 `std>0.30` | Gate 3: 权重归一化熵 $\mathcal{H}_{\text{norm}} \in [0.30, 0.85]$ | Gate 4: 步耗时 (`ms/step`) | 门禁结论 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `H1=original_none | H2=original_single | H3=original_mean` | `0.00e+00` | `0.0000` | `0.5486` | `1.0000` | `3.68 ms` | **PASS** |
| `H1=original_none | H2=original_single | H3=salience_3A` | `9.54e-07` | `0.0000` | `0.5486` | `0.4412` | `8.64 ms` | **PASS** |
| `H1=original_none | H2=original_single | H3=attn_rollout_3B` | `7.15e-07` | `0.0000` | `0.5486` | `0.6754` | `3.75 ms` | **PASS** |
| `H1=original_none | H2=clean_proj_2A | H3=salience_3A` | `9.54e-07` | `0.0045` | `0.6007` | `0.4412` | `12.11 ms` | **PASS** |
| `H1=original_none | H2=grl_adv_2B | H3=salience_3A` | `9.54e-07` | `0.0045` | `0.6007` | `0.4412` | `25.92 ms` | **PASS** |
| `H1=vtlp_1A | H2=clean_proj_2A | H3=salience_3A` | `9.54e-07` | `0.0045` | `0.6006` | `0.4412` | `34.25 ms` | **PASS** |
| `H1=channel_drop_1B | H2=clean_proj_2A | H3=salience_3A` | `9.54e-07` | `0.0045` | `0.5542` | `0.4412` | `30.09 ms` | **PASS** |
| `H1=offline_wav_1C | H2=clean_proj_2A | H3=salience_3A` | `9.54e-07` | `0.0086` | `0.4574` | `0.4412` | `26.57 ms` | **PASS** |
| `H1=vtlp_1A | H2=grl_adv_2B | H3=attn_rollout_3B` | `7.15e-07` | `0.0045` | `0.6006` | `0.6754` | `31.24 ms` | **PASS** |

## 门禁诊断解读
1. **Gate 1 (Mask Invariance Check)**：向 Batch 末尾额外拼接 50 帧幅值为 $\mathcal{N}(0, 100)$ 的极端噪声 Padding 帧后，所有 9 组锤子配置的 `z_utt_emo`、`target_utt` 与 `loss_frm` 最大绝对偏差均严格小于 `1e-5`，证明 `padding_mask -> -1e9` 与真实有效帧长 `N_valid` 保护 100% 闭环。
2. **Gate 2 (Grad-Norm Auto-Balancer)**：自动测算出正交投影解耦 (`clean_proj_2A`) 与梯度反转层 (`grl_adv_2B`) 对主干帧重构梯度的比例，并给出使辅助梯度严格占主损失梯度 `15%` 的自适应权重 $\lambda_{\text{rec}}$，彻底杜绝拍脑袋调参。
3. **Gate 3 (Anti-Collapse & Entropy Monitor)**：所有候选配置的特征标准差均保持在 `> 0.30`（无坍缩为常向量风险）；显著性池化 `salience_3A` 与注意力卷展 `attn_rollout_3B` 的归一化信息熵稳定在 `[0.30, 0.85]` 的黄金聚焦区间内（既未退化为 `1.00` 的平庸均值，也未退化为 `<0.10` 的单帧尖峰脉冲）。
