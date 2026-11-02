# Step 5 底座代码信号与数学假设 1-Batch 动态实证报告 (`step5_base_assumptions_verification.md`)

> **总体验证结论**：`PASSED (100% 契约不变量吻合)`
> **验证脚本**：[`scripts/verify_base_assumptions.py`](../scripts/verify_base_assumptions.py)
> **共生手册锚点**：[`docs/BASE_CODE_ANATOMY_MANUAL.md`](../docs/BASE_CODE_ANATOMY_MANUAL.md)

## 1. 五层符号契约 (`S1 -> S5`) 实测数值矩阵

| 符号 ID | 验证维度与数学不变量 | 理论期望值 | 1-Batch 实测值 | 状态 |
| :--- | :--- | :--- | :--- | :--- |
| `[SYM-E2V-S1-01]` | 有效语音段零均值单位方差 ($\mu, \sigma$) | $\mu=0.0, \sigma=1.0$ | $\mu=1.38e-07, \sigma=1.0000$ | PASS |
| `[SYM-E2V-S1-01]` | 短句尾端 Padding 采样点绝对零值 | $0.0$ | $0.00e+00$ | PASS |
| `[SYM-E2V-S2-01]` | 7 层卷积 $320\times$ 下采样有效帧数 | `[199, 64]` | `[199, 64]` | PASS |
| `[SYM-E2V-S2-02]` | Padded 帧被选入掩码预测目标的数量 | `0` | `0` | PASS |
| `[SYM-E2V-S3-01]` | 前置 10 个 Chunk Tokens 后序列总长 | `209` (`199+10`) | `209` | PASS |
| `[SYM-E2V-S3-02]` | 单层 Instance-Norm 白化向量 $L_2$ 范数 $\sqrt{768}$ | `27.7128` | `27.7121` | PASS |
| `[SYM-E2V-S3-02]` | 掩码保护下短句截断 vs 补零的表示误差 | `< 1e-6` | `0.00e+00` | PASS |
| `[SYM-E2V-S4-01]` | EMA 动量余弦退火区间 ($\tau_0 \to \tau_T$) | `0.999 -> 0.9999` | `0.999 -> 0.9999` | PASS |
| `[SYM-E2V-S5-01]` | 下游探针有掩码 vs 无掩码短句长度偏差 | `< 1e-6` vs `> 0.1` | `0.00e+00` vs `1.6898` | PASS |

## 2. 底座破产假设实锤：80% 平淡静音帧淹没 20% 情绪爆发帧 (`[SYM-E2V-S3-03]` vs `[SYM-E2V-SLOT-H3]`)

- **实验构造**：Sample 0 共 199 帧，前 80%（`0..158` 帧）为低能量平淡语音（方向 $\mathbf{u}_{\text{neutral}}$），后 20%（`159..198` 帧）为高能量情绪爆发段（正交方向 $\mathbf{u}_{\text{burst}}$）。
- **原版 `emotion2vec` 均匀均值池化 (`original_mean`)**：
  - 情绪爆发帧获得的权重总质量：**`20.10%`**
  - 句级靶标 $\bar{y}^T$ 与 80% 平淡段的余弦相似度：**`0.9698`**（被平淡静音段彻底主导）
  - 句级靶标 $\bar{y}^T$ 与 20% 情绪爆发段的余弦相似度：**`0.2433`**（情绪爆发信息被严重稀释）
- **插入卡槽 H3 (`salience_3A`) 后**：
  - 情绪爆发帧获得的显著性权重总质量：**`100.00%`**
  - 句级靶标 $\bar{y}^T_{\text{sal}}$ 与 20% 情绪爆发段的余弦相似度跃升至：**`0.9999`**
  - 同时短句 Padding 零泄漏误差保持：**`0.00e+00 < 1e-6`**
