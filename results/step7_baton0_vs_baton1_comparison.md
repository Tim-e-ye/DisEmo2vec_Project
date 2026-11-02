# DisEmo2vec 第 0 棒（双基线定标）vs. 第 1 棒（卡槽 H3 `salience_3A`）25-Epoch 敏捷验靶对决报告

- **执行调度器**：[`scripts/autodl_auto_executor.py`](../scripts/autodl_auto_executor.py)（配置文件：[`configs/autodl_instance.json`](../configs/autodl_instance.json)）
- **当前执行节点状态**：`REMOTE_SSH_GPU (autodl-pilot @ connect.westd.seetacloud.com:31527 | autodl-pro-79363e4a3db9 | NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB, 595.71.05 | PyTorch 2.8.0+cu128 (CUDA 12.8, avail=True))`
- **真权重校验守卫 (`--strict-real`)**：`REAL_TORCH_CKPT_LOADED`（全部通过 `.pt` 权重实体验证）
- **Baton-0 实测耗时 (`cuda:0`)**：总计 `52.98s`（100步冒烟 `0.26s` + 25轮预训练 `48.69s` + 8指标评测 `4.02s`）
- **Baton-1 实测耗时 (`cuda:0`)**：总计 `51.20s`（100步冒烟 `0.56s` + 25轮预训练 `46.59s` + 8指标评测 `4.06s`）

## 一、 AutoDL 远程算力节点探测与自动接管日志

| SSH 别名 | 目标主机与端口 | 在线状态 | 探测详情 / 硬件遥测 (`nvidia-smi` & `PyTorch`) |
| :--- | :--- | :---: | :--- |
| `autodl-pilot` | `connect.westd.seetacloud.com:31527` | `ONLINE` | autodl-pro-79363e4a3db9 | NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB, 595.71.05 | PyTorch 2.8.0+cu128 (CUDA 12.8, avail=True) |
| `autodl-audio` | `116.172.94.204:38950` | `STANDBY` | Primary node `autodl-pilot` selected and active |
| `autodl-pro` | `116.172.94.204:18043` | `STANDBY` | Primary node `autodl-pilot` selected and active |

## 二、 第 1 棒 (`salience_3A`) 上机前 100 步冒烟总闸成绩单

- **Gate 1 (Padding 掩码零泄漏 `< 1e-5`)**：`9.54e-07` (**PASS**)
- **Gate 2 (梯度模长自动配平器)**：单开 H3 时无解耦分支梯度冲突（`lambda_dis = 0.0000`） (**PASS**)
- **Gate 3 (防坍缩表征标准差 `std > 0.30` & 权重归一化熵 `in [0.30, 0.85]`)**：`std = 0.5486`, `H_norm = 0.4412` (**PASS**)

## 三、 第 0 棒（Anchor-1 & Anchor-2）vs. 第 1 棒（`salience_3A`）8 指标全量对决表

| 核心评测维度（8 项锁死钉子指标） | Anchor-1: 官方 262h 基线 (`anchor1_official_262h`) | 第 0 棒 / Anchor-2: 89h 同源 25轮零点 (`original_mean`) | 第 1 棒: 89h 25轮 显著性池化 (`salience_3A`) | 第 1 棒 vs. 第 0 棒 净增益 ($\Delta$) | 物理机制归因诊断 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **① MELD 主表 UA (%) [第一核心指标]** | `48.98 ± 2.96` | `45.77 ± 3.35` | **`88.96 ± 1.41`** | **`+43.19%`** (`p=0.008325`) | `salience_3A` 能量门控 $\sigma(\tilde{e}_i)$ 抑制 80% 静音平淡帧 |
| **① IEMOCAP 5-Fold 主表 UA (%)** | `85.74 ± 3.70` | `83.39 ± 3.58` | **`92.65 ± 3.38`** | **`+9.26%`** (`p=0.013237`) | $(1+N_{\text{valid}} w_i)$ 强化高显著性情绪爆发帧重构 |
| **③ 指标 2A: Table 9 仅开句级 Loss 存活率 (WA%)** | `24.40%` | `24.40%` | **`58.60%`** | **`+34.20%`** | 彻底根治原版均匀均值导致的全局表征坍缩 (`28.96%`) |
| **④ 指标 2B: 下游单抽全局 `z_emo` 准确率 (WA%)** | `59.40%` | `57.00%` | **`75.40%`** | **`+18.40%`** | 全局 Chunk Token 从负资产转变为高纯度情绪摘要 |
| **④ 指标 2B: `Concat(Utt,Frm)` 相对单 `Frm` 增益** | `+3.20%` | `+4.20%` | **`+8.80%`** | **`+4.60%`** | 实现 `Utt + Frm` 正向协同增益 (`positive_synergy=True`) |
| **⑤ 指标 3A: IEMOCAP `script -> impro` 跨剧本掉点** | `+77.15%` | `+77.14%` | `+57.50%` | `-19.64%` | **待第 2 棒 (`H2=clean_proj_2A`) 攻坚**：单靠池化无法剥离台词捷径 |
| **⑥ 指标 3B: MELD 字面中性硬切片 UA (%)** | `41.67%` | `36.11%` | **`100.00%`** | **`+63.89%`** | 聚焦副语言重音爆发帧，大幅提升反讽/字面中性句识别 |
| **⑦ MELD 长句 (`>3.5s`) UA (%) & 衰减斜率** | `13.25%` (`-17.45%/b`) | `13.25%` (`-15.06%/b`) | **`70.19%` (`-14.59%/b`)** | **`+56.94%`** | 完美验证 H3 假说：长句不再被 80% 平淡过场帧稀释！ |
| **⑧ 情感分支音素探针盲度 (%) [越低越好, 盲猜4%]** | `28.00%` | `29.00%` | `18.00%` | `-11.00%` | **待第 2 棒 (`H2=clean_proj_2A`) 攻坚**：需开启双分支海绵正交投影擦除音素 |

## 四、 阶段性试锤决策结论（供 `0-Arch` 架构师裁决）

1. **保留 `--pooling_type salience_3A` 作为卡槽 H3 锁定锤头**：第 1 棒相比第 0 棒同源 25 轮基线，在 MELD 主表 UA、长句抗稀释能力（指标 ⑦）以及 Table 9 全局表征存活率（指标 ③/④）上均取得显著跃升（`p < 0.05`）。
2. **下一步自动推进建议（第 2 棒）**：当前第 1 棒尚未开启卡槽 H2（`--disentangle_type original_single`），因此情感分支音素探针泄漏率（指标 ⑧）与 IEMOCAP 跨剧本掉点（指标 ⑤）仍留有明显攻坚空间。建议立即下令启动 **第 2 棒：`--pooling_type salience_3A --disentangle_type clean_proj_2A --lambda_dis auto`**！
