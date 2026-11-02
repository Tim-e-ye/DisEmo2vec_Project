#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DisEmo2vec Step 5 - Action 5.3: AST Bidirectional Code-Doc Symbiosis Sync Engine
================================================================================
Scans `src/base_e2v/` and `models/hammer_slots/` for `[SYM-E2V-*]` symbols and
`@bce_contract` blocks, computes AST structural hashes (`ast_hash`) and exact
line ranges (`lines="Lstart-Lend"`), and synchronizes/verifies:
1. `docs/symbol_manifest.json` (Machine-readable AST & contract registry)
2. `docs/BASE_CODE_ANATOMY_MANUAL.md` (Bidirectional Code-Doc Anatomy Manual
   wrapped in paired `<!-- [SYM-E2V-...:START] ... -->` static anchors).

CLI Usage:
  python tools/base_code_symbiosis_sync.py --scan
  python tools/base_code_symbiosis_sync.py --sync
  python tools/base_code_symbiosis_sync.py --verify
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCAN_DIRS = ("src/base_e2v", "models/hammer_slots")
MANIFEST_PATH = PROJECT_ROOT / "docs" / "symbol_manifest.json"
MANUAL_PATH = PROJECT_ROOT / "docs" / "BASE_CODE_ANATOMY_MANUAL.md"

SYM_HEADER_RE = re.compile(r"\[(SYM-E2V-[A-Z0-9\-]+)\]")
ANCHOR_START_RE = re.compile(
    r'<!--\s*\[(?P<uid>SYM-E2V-[A-Z0-9\-]+):START\]\s+file="(?P<file>[^"]+)"\s+lines="(?P<lines>\d+-\d+)"\s+ast_hash="(?P<hash>[0-9a-f]{8})"\s*-->'
)


def compute_node_ast_hash(node: ast.AST) -> str:
    """
    Computes a deterministic 8-char SHA256 hex fingerprint of a class/function AST node
    (excluding its docstring literal so documentation edits do not invalidate code AST hash).
    """
    node_copy = copy.deepcopy(node)
    if isinstance(node_copy, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node_copy.body:
        first_stmt = node_copy.body[0]
        if (
            isinstance(first_stmt, ast.Expr)
            and isinstance(getattr(first_stmt, "value", None), ast.Constant)
            and isinstance(first_stmt.value.value, str)
        ):
            node_copy.body = node_copy.body[1:]
    dumped = ast.dump(node_copy, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()[:8]


def parse_bce_contract_from_docstring(docstring: str) -> Optional[Dict[str, Any]]:
    """
    Extracts structured `@bce_contract:` fields from a symbol's docstring without
    requiring external YAML libraries.
    """
    if not docstring or "@bce_contract:" not in docstring:
        return None

    m = SYM_HEADER_RE.search(docstring)
    uid = m.group(1) if m else None

    _, contract_part = docstring.split("@bce_contract:", 1)
    lines = contract_part.splitlines()

    data: Dict[str, Any] = {
        "uid": uid or "",
        "layer": "",
        "name": "",
        "file_rel_path": "",
        "tensor_input": "",
        "tensor_output": "",
        "signal_math_assumptions": [],
        "hard_invariants": [],
        "broken_assumptions": [],
        "reserved_slot_id": "",
    }

    current_list_key: Optional[str] = None
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            continue

        if stripped.startswith("- "):
            if current_list_key is not None and isinstance(data.get(current_list_key), list):
                val = stripped[2:].strip().strip('"').strip("'")
                data[current_list_key].append(val)
            continue

        if ":" in stripped:
            key, val_part = stripped.split(":", 1)
            key = key.strip()
            val = val_part.strip().strip('"').strip("'")
            if key in ("signal_math_assumptions", "hard_invariants", "broken_assumptions"):
                current_list_key = key
                if val:
                    data[key].append(val)
            elif key in data:
                current_list_key = None
                data[key] = val

    return data


def scan_all_symbols() -> List[Dict[str, Any]]:
    """Scans `src/base_e2v` and `models/hammer_slots` and returns sorted symbol descriptors."""
    discovered: List[Dict[str, Any]] = []

    for rel_dir in SCAN_DIRS:
        abs_dir = PROJECT_ROOT / rel_dir
        if not abs_dir.exists():
            continue
        for py_file in sorted(abs_dir.rglob("*.py")):
            rel_file = py_file.relative_to(PROJECT_ROOT).as_posix()
            source_text = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source_text, filename=rel_file)

            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    doc = ast.get_docstring(node, clean=True)
                    if not doc:
                        continue
                    contract = parse_bce_contract_from_docstring(doc)
                    if contract is None:
                        continue

                    uid = contract["uid"]
                    line_start = int(node.lineno)
                    line_end = int(getattr(node, "end_lineno", node.lineno))
                    ast_hash = compute_node_ast_hash(node)

                    discovered.append(
                        {
                            "uid": uid,
                            "symbol_name": node.name,
                            "symbol_type": "class" if isinstance(node, ast.ClassDef) else "function",
                            "layer": contract["layer"],
                            "title": contract["name"],
                            "file": rel_file,
                            "line_start": line_start,
                            "line_end": line_end,
                            "lines": f"{line_start}-{line_end}",
                            "ast_hash": ast_hash,
                            "tensor_input": contract["tensor_input"],
                            "tensor_output": contract["tensor_output"],
                            "signal_math_assumptions": contract["signal_math_assumptions"],
                            "hard_invariants": contract["hard_invariants"],
                            "broken_assumptions": contract["broken_assumptions"],
                            "reserved_slot_id": contract["reserved_slot_id"],
                        }
                    )

    # Sort S1..S5 first, then SLOT-H1..H3
    def _sort_key(item: Dict[str, Any]) -> Tuple[int, str]:
        u = item["uid"]
        if "-S1-" in u:
            return (1, u)
        if "-S2-" in u:
            return (2, u)
        if "-S3-" in u:
            return (3, u)
        if "-S4-" in u:
            return (4, u)
        if "-S5-" in u:
            return (5, u)
        return (6, u)

    discovered.sort(key=_sort_key)
    return discovered


def render_symbol_anchor_block(sym: Dict[str, Any]) -> str:
    """Renders a single `<!-- [SYM-E2V-...:START] ... --> ... <!-- [SYM-E2V-...:END] -->` block."""
    uid = sym["uid"]
    file_p = sym["file"]
    lines_s = sym["lines"]
    l_start = sym["line_start"]
    l_end = sym["line_end"]
    ast_h = sym["ast_hash"]

    abs_uri = f"file:///{PROJECT_ROOT.as_posix()}/{file_p}#L{l_start}-L{l_end}"

    lines = [
        f'<!-- [{uid}:START] file="{file_p}" lines="{lines_s}" ast_hash="{ast_h}" -->',
        f"### `[{uid}]` {sym['title']} (`{sym['symbol_name']}`)",
        "",
        f"- **绑定源码坐标**：[{Path(file_p).name} (L{l_start}-L{l_end})]({abs_uri}) | **拓扑层**：`{sym['layer']}` | **AST 指纹**：`{ast_h}`",
        f"- **输入张量契约 (`tensor_input`)**：`{sym['tensor_input']}`",
        f"- **输出张量契约 (`tensor_output`)**：`{sym['tensor_output']}`",
        f"- **绑定/预留卡槽 (`reserved_slot_id`)**：`{sym['reserved_slot_id']}`",
        "",
        "#### 📐 数学与信号处理假设 (`signal_math_assumptions`)",
    ]
    for item in sym["signal_math_assumptions"]:
        lines.append(f"- {item}")

    lines.extend(["", "#### 🛡️ 绝对不可破坏的保命护栏 (`hard_invariants`)"])
    for item in sym["hard_invariants"]:
        lines.append(f"- {item}")

    lines.extend(["", "#### 🎯 原版失效假设 / 钉子精确代码位点 (`broken_assumptions`)"])
    for item in sym["broken_assumptions"]:
        lines.append(f"- {item}")

    lines.append(f"<!-- [{uid}:END] -->")
    return "\n".join(lines)


def generate_full_anatomy_manual(symbols: List[Dict[str, Any]]) -> str:
    """Generates the complete `docs/BASE_CODE_ANATOMY_MANUAL.md` document."""
    base_syms = [s for s in symbols if not s["uid"].startswith("SYM-E2V-SLOT-")]
    slot_syms = [s for s in symbols if s["uid"].startswith("SYM-E2V-SLOT-")]

    header = [
        "# 《emotion2vec / DisEmo2vec》基础代码双向静态共生解剖手册（Step 5 & Step 7 权威基线）",
        "",
        "> **自动同步引擎**：`tools/base_code_symbiosis_sync.py`（基于 AST 语法树指纹 `ast_hash` 与 `@bce_contract` 双向锁死）",
        "> **配套机读清单**：`docs/symbol_manifest.json`",
        "> **验证状态**：运行 `python tools/base_code_symbiosis_sync.py --verify` 100% 通过静态双射校验",
        "",
        "### 📌 三源认知标签体系（5% 反思防混淆增量补丁）",
        "为防止在拆解底座代码与设计新锤子时混淆「原论文宣称的机制」、「Fairseq 源码实际干的事」与「本试验田新增的传感器」，本手册所有条目严格区分以下三源标签：",
        "1. **`[论文原貌 Paper Claim]`**：ACL 2024 `emotion2vec` 原论文正文宣称的数学公式与设计意图（例如宣称 Utterance Loss 与 Frame Loss 协同学到全局与局部情感）；",
        "2. **`[源码客观事实 Code Reality]`**：原版 `emotion2vec` Fairseq 源码中真实执行的张量运算与隐式假设（例如 `s2_conv_frontend.py` 首层 GroupNorm 抹平绝对声压级、`s3_dual_stream_backbone.py` 顶部 $k=8$ 层必须做无仿射参数 InstanceNorm 白化使 $\\Vert y^T_i \\Vert_2 \\approx 27.71$、以及 Table 9 暴露的单抽 `Utt` 仅 `62.77%` 负资产退化）；",
        "3. **`[外挂诊断传感器 Sensor Patch]`**：原版 `emotion2vec` 源码中**原本不存在**、由本试验田在不改动原版主干拓扑的前提下旁路提取的无监督物理传感器或离线探针（例如 `[SYM-E2V-S1-01]` 旁路提取的 50Hz 句内标准化短时能量 `frame_energy_std` $\\tilde{e}_i$ 专为卡槽 `[SLOT-H3]` 提供有声门控，以及离线提取的 HuBERT $K=500$ 软分布 $P_{\\text{hubert}}$ 专为卡槽 `[SLOT-H2]` 提供无监督海绵靶标）。",
        "",
        "---",
        "",
        "## 一、 S1 ~ S5 五层基础代码拓扑与 H1 ~ H3 卡槽精确咬合全景图",
        "",
        "```mermaid",
        "flowchart TD",
        '    subgraph S1 ["S1 数据与信号层 (src/base_e2v/s1_audio_dataset.py)"]',
        '        SYM_S1_01["[SYM-E2V-S1-01] BaseAudioCollaterAndNormalizer<br/>• 补零前单句零均值单位方差归一化 (护栏 I0)<br/>• 采样点 wav_padding_mask & 50Hz 短时能量 e_tilde"]',
        "    end",
        "",
        '    subgraph S2 ["S2 声学前端与掩码层 (src/base_e2v/s2_conv_frontend.py)"]',
        '        SYM_S2_01["[SYM-E2V-S2-01] ConvFeatureExtractionModel<br/>• 7层 1D 卷积 320倍下采样 (16kHz -> 50Hz [B,N,768])<br/>• 采样点掩码整数除法对齐 frame_padding_mask (护栏 I1-S2)"]',
        '        SYM_S2_02["[SYM-E2V-S2-02] MaskingAndPositionalConv<br/>• Student 跨度掩码 (p=0.5, l=5, 护栏 I2-S2)<br/>• k=128 分组卷积相对位置编码 pos_conv"]',
        "        SYM_S2_01 --> SYM_S2_02",
        "    end",
        "",
        '    subgraph S3 ["S3 双流主干与联合损失层 — 核心病灶与卡槽宿主区"]',
        '        SYM_S3_01["[SYM-E2V-S3-01] PrependChunkTokens<br/>• pos_conv 后拼接前置 C=10 全局 Chunk Tokens<br/>• 掩码前端补 C 个 False -> [B, C+N]"]',
        '        SYM_S3_02["[SYM-E2V-S3-02] EMATeacherTargetExtractor<br/>• Teacher 无掩码前向 & 剥离前 C 个 Token (护栏 I2)<br/>• 顶部 k=8 层独立无仿射白化防坍缩 ||y^T||_2≈27.71 (护栏 I1)"]',
        '        SYM_S3_03["[SYM-E2V-S3-03] BaseEmotion2VecCriterion<br/>• 🎯 失效假设 A: 1/N_valid 均匀平均被 80% 平淡帧稀释<br/>• 🎯 失效假设 B: 单分支 10 Token 混同导致词汇捷径泄漏<br/>• 🛡️ 保命护栏 I3/I4: padding_mask 清零与 FP32/detach"]',
        "        SYM_S3_01 --> SYM_S3_03",
        "        SYM_S3_02 --> SYM_S3_03",
        "    end",
        "",
        '    subgraph SLOTS ["Phase II 可插拔锤子卡槽层 (models/hammer_slots/)"]',
        '        SLOT_H1["🔨 [SYM-E2V-SLOT-H1] HammerSlotH1_ViewTransform<br/>(--view_type: original_none | vtlp_1A | channel_drop_1B | offline_wav_1C)"]',
        '        SLOT_H2["🔨 [SYM-E2V-SLOT-H2] HammerSlotH2_Disentangler<br/>(--disentangle_type: original_single | clean_proj_2A | grl_adv_2B)"]',
        '        SLOT_H3["🔨 [SYM-E2V-SLOT-H3] HammerSlotH3_TemporalAggregator<br/>(--pooling_type: original_mean | salience_3A | attn_rollout_3B)"]',
        "    end",
        "",
        '    subgraph S4_S5 ["S4 优化动量层 & S5 下游探针层"]',
        '        SYM_S4_01["[SYM-E2V-S4-01] EMAApplierAndFp16Guard<br/>• EMA 动量 tau in [0.999, 0.9999] 余弦退火<br/>• 梯度模长配平器 ||grad(L_hammer)|| / ||grad(L_Frm)|| in [0.05, 0.50]"]',
        '        SYM_S5_01["[SYM-E2V-S5-01] FrozenFeatureExtractorAndLinearProbe<br/>• 冻结主干切片 [:5] 导出 z_utt_emo & [C:] 导出 x_frm_emo<br/>• 0.20M 轻量线性探针对接 eval/eval_all_8_metrics.py"]',
        "    end",
        "",
        "    SYM_S1_01 --> SYM_S2_01",
        "    SYM_S2_02 -.-> SLOT_H1 -.-> SYM_S3_01",
        "    SYM_S3_02 -.-> SLOT_H3 -.-> SYM_S3_03",
        "    SYM_S3_01 -.-> SLOT_H2 -.-> SYM_S3_03",
        "    SYM_S3_03 --> SYM_S4_01 --> SYM_S5_01",
        "```",
        "",
        "---",
        "",
        "## 二、 基础代码五层静态符号契约台账（`[SYM-E2V-S1-01]` ~ `[SYM-E2V-S5-01]`）",
        "",
    ]

    for sym in base_syms:
        header.append(render_symbol_anchor_block(sym))
        header.append("")
        header.append("---")
        header.append("")

    header.extend(
        [
            "## 三、 第 6 步可插拔锤子卡槽双向静态契约（`[SYM-E2V-SLOT-H1]` ~ `[SYM-E2V-SLOT-H3]`）",
            "",
        ]
    )

    for sym in slot_syms:
        header.append(render_symbol_anchor_block(sym))
        header.append("")
        header.append("---")
        header.append("")

    header.extend(
        [
            "## 四、 第 7 步顺畅搜锤与试锤四大闭环保障机制（内嵌规范）",
            "",
            "### 1. 跨领域搜锤地图（按钉子数学本质借锤）",
            "- **针对 `[SYM-E2V-SLOT-H3]`（80% 平淡帧稀释 20% 情绪爆发帧 —— 弱监督稀疏事件汇聚）**：",
            "  - **矿区 A（DCASE / TASLP 弱监督音频事件检测）**：`AutoPool`（自适应指数软池化）、`Gated Attention MIL`、`Power-Mean Pooling`；",
            "  - **矿区 B（CVPR / ICLR 视觉动态 Token 显著性汇聚）**：`Attention Rollout Salience`、`Token Merging (ToMe)`。",
            "- **针对 `[SYM-E2V-SLOT-H2]`（零人工标签剥离词汇与情绪 —— 无监督子空间去相关与信息擦除）**：",
            "  - **矿区 A（ICML / NeurIPS / TASLP 零样本语音解耦）**：`ContentVec`、`SpeechSplit`、`Prosody2Vec`；",
            "  - **矿区 B（ICLR / ICML 自监督非对比去相关）**：`Barlow Twins`（跨分支互相关矩阵非对角归零）、`VICReg`、`vCLUB` 互信息上界最小化。",
            "- **针对 `[SYM-E2V-SLOT-H1]`（切断固定音素共振峰死记硬背 —— 保韵律声学信息瓶颈摄动）**：",
            "  - **矿区**：`ContentVec / NANSY` 保基频频谱包络扭曲（VTLP）、保能量频带通道掩码（`channel_drop_1B`）。",
            "",
            "### 2. 文献锤子本土化「三步阻抗匹配转译协议」",
            "1. **Step A（无监督信号替换）**：严禁引入外部 ASR 文本/音素人工标签，统一替换为 `[SYM-E2V-S3-02]` 的白化 Teacher 特征 $Y^T$、`[SYM-E2V-S1-01]` 的短时能量 $\\tilde{e}_i$（`[外挂诊断传感器 Sensor Patch]`）或离线 K=500 无监督音素簇分布 $P_{\\text{hubert}}$（`[外挂诊断传感器 Sensor Patch]`）；",
            "2. **Step B（量纲与范数阻抗匹配）**：注意 `[SYM-E2V-S3-02]` 输出的白化目标帧二范数恒为 $\\Vert y^T_i \\Vert_2 \\approx \\sqrt{768} = 27.71$（`[源码客观事实 Code Reality]`，而非单位球面 $\\Vert z \\Vert_2 = 1$），若文献公式基于余弦/球面假设，需在卡槽内先做局部 `F.normalize` 或温度缩放；",
            "3. **Step C（`padding_mask` 掩码安全改写）**：一切求和 $\\sum_{i=1}^N$ 或 `Softmax` 必须严格接入 `padding_mask`（打分置 `-1e9`、有效帧长取 $N_{\\text{valid}}$），并通过 `scripts/smoke_test_hammer.py` 的 Padding 零泄漏断言（误差 $< 10^{-6}$）。",
            "",
            "---",
            "",
            "## 五、 `S1 ~ S5` 开放母板全链路 6 大候选切口表（3 个已激活卡槽 + 3 个储备扩展切口）",
            "",
            "> **母板开放性原则**：`S1 ~ S5` 的 8 个基础符号均作为开放式实验母板。除当前在 `models/hammer_modules.py` 中已通过命令行开关激活的 **3 大主力卡槽（`SLOT-H1 / H2 / H3`）** 外，若后续文献阅读发现针对 Teacher 层间权重或掩码策略的强力新锤子，可直接沿以下预留坐标切入：",
            "",
            "| 切口编号与状态 | 宿主符号与源码坐标 | 针对的原版失效假设 (`[Code Reality]`) | 可插拔算法机制与信号来源 |",
            "| :--- | :--- | :--- | :--- |",
            "| **`SLOT-H1`**<br/>`[ACTIVE 已激活]` | `[SYM-E2V-S2-02]` $\\to$ `[SYM-E2V-SLOT-H1]`<br/>(`models/hammer_slots/slot_h1_view.py`) | Student 与 Teacher 共享完全相同的共振峰包络，模型通过死记特定说话人/音素共振峰走捷径 | `--view_type`: `vtlp_1A` (在线线性频谱扭曲) / `channel_drop_1B` (能量自适应通道掩码) / `offline_wav_1C` (离线变声副本 `(B,1,1)` 独立采样) |",
            "| **`SLOT-H2`**<br/>`[ACTIVE 已激活]` | `[SYM-E2V-S3-01]` & `[SYM-E2V-S3-03]` $\\to$ `[SYM-E2V-SLOT-H2]`<br/>(`models/hammer_slots/slot_h2_disentangle.py`) | 全部 $C=10$ 个前置 Chunk Tokens 混为单分支回归含音素的 Teacher 目标，导致严重音素泄漏与跨剧本掉点 | `--disentangle_type`: `clean_proj_2A` (前5情绪+后5海绵双分支正交投影 + `[Sensor Patch]` $P_{\\text{hubert}}$ KL 吸附) / `grl_adv_2B` (梯度反转层) |",
            "| **`SLOT-H3`**<br/>`[ACTIVE 已激活]` | `[SYM-E2V-S3-02]` & `[SYM-E2V-S3-03]` $\\to$ `[SYM-E2V-SLOT-H3]`<br/>(`models/hammer_slots/slot_h3_pooling.py`) | `1/N_valid` 无差别时间平均被自然对话中 80% 的平淡/静音帧淹没，导致 Table 9 全局表征坍缩 (`28.96%`) 与长句严重稀释 | `--pooling_type`: `salience_3A` (`[Sensor Patch]` 短时能量软门控 $\\sigma(\\tilde{e}_i)$ $\\times$ 局部背景偏离度 + $(1+N_{\\text{valid}} w_i)$ 帧重加权) / `attn_rollout_3B` |",
            "| **`Candidate-H4`**<br/>`[RESERVE 储备切口]` | `[SYM-E2V-S3-02]` `EMATeacherTargetExtractor`<br/>(`src/base_e2v/s3_dual_stream_backbone.py#L130-L142`) | 原版对 Teacher 顶部 $k=8$ 层 (`Layer 5..12`) 赋予完全相等的 `1/8` 算术平均权重，未区分中层副语言韵律层与高层语义/音素层 | **Teacher 顶部 8 层跨层非均匀加权 / 音素主导层降权切口**：将 `np.mean(top_k_whitened, axis=0)` 替换为可学习或基于层间音素探针盲度加权的 $\\sum_{l=5}^{12} \\beta_l \\tilde{h}^{(l)}$ |",
            "| **`Candidate-H5`**<br/>`[RESERVE 储备切口]` | `[SYM-E2V-S2-02]` `MaskingAndPositionalConv`<br/>(`src/base_e2v/s2_conv_frontend.py#L148-L165`) | 原版时间掩码 `compute_mask_indices` 在有效帧上均匀随机采样 (`p=0.5, l=5`)，大量掩码落在无信息的平淡静音段，情绪爆发段掩码不足 | **显著性引导时间掩码切口 (Salience-Guided Span Masking)**：利用 `[SYM-E2V-S1-01]` 的 `[Sensor Patch]` `frame_energy_std` 或基频变化率提高情绪爆发帧的掩码采样概率 |",
            "| **`Candidate-H6`**<br/>`[RESERVE 储备切口]` | `[SYM-E2V-S4-01]` `EMAApplierAndFp16Guard`<br/>(`src/base_e2v/s4_ema_and_fp16.py#L52-L85`) | 原版对所有共享层使用单一固定轨迹的 EMA 动量 $\\tau(t)$ 与固定辅助损失权重，未随解耦程度动态调节 | **动态梯度模长自平衡与分层 EMA 退火切口**：在预训练过程中每 $K$ 步根据 $\\Vert \\nabla \\mathcal{L}_{\\text{dis}} \\Vert / \\Vert \\nabla \\mathcal{L}_{\\text{Frm}} \\Vert$ 在线自适应微调 $\\lambda_{\\text{dis}}(t)$ |",
            "",
        ]
    )
    return "\n".join(header)


def sync_files(symbols: List[Dict[str, Any]]) -> None:
    """Writes `docs/symbol_manifest.json` and `docs/BASE_CODE_ANATOMY_MANUAL.md`."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest_payload = {
        "project": "DisEmo2vec",
        "step": "Step 5 & Step 6 Bidirectional Code-Doc Symbiosis",
        "num_total_symbols": len(symbols),
        "num_base_symbols": sum(1 for s in symbols if not s["uid"].startswith("SYM-E2V-SLOT-")),
        "num_slot_symbols": sum(1 for s in symbols if s["uid"].startswith("SYM-E2V-SLOT-")),
        "symbols": {s["uid"]: s for s in symbols},
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    MANUAL_PATH.write_text(generate_full_anatomy_manual(symbols), encoding="utf-8")


def verify_bidirectional_alignment(symbols: List[Dict[str, Any]]) -> bool:
    """
    Verifies 100% bijection between source code AST (`symbols`),
    `docs/symbol_manifest.json`, and `docs/BASE_CODE_ANATOMY_MANUAL.md`.
    """
    if not MANIFEST_PATH.exists() or not MANUAL_PATH.exists():
        print("[FAIL] Manifest or Manual file missing! Run with --sync first.")
        return False

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest_syms = manifest.get("symbols", {})
    manual_text = MANUAL_PATH.read_text(encoding="utf-8")

    manual_anchors: Dict[str, Dict[str, str]] = {}
    for match in ANCHOR_START_RE.finditer(manual_text):
        manual_anchors[match.group("uid")] = {
            "file": match.group("file"),
            "lines": match.group("lines"),
            "ast_hash": match.group("hash"),
        }

    all_passed = True
    for sym in symbols:
        uid = sym["uid"]
        # 1. Check manifest
        if uid not in manifest_syms:
            print(f"[FAIL] Symbol {uid} missing from symbol_manifest.json")
            all_passed = False
            continue
        m_entry = manifest_syms[uid]
        if (
            m_entry.get("ast_hash") != sym["ast_hash"]
            or m_entry.get("lines") != sym["lines"]
            or m_entry.get("file") != sym["file"]
        ):
            print(
                f"[FAIL] Manifest drift for {uid}: code=({sym['file']}:{sym['lines']}@{sym['ast_hash']}) "
                f"vs manifest=({m_entry.get('file')}:{m_entry.get('lines')}@{m_entry.get('ast_hash')})"
            )
            all_passed = False

        # 2. Check manual anchor
        if uid not in manual_anchors:
            print(f"[FAIL] Anchor [{uid}:START] missing from BASE_CODE_ANATOMY_MANUAL.md")
            all_passed = False
            continue
        a_entry = manual_anchors[uid]
        if (
            a_entry["ast_hash"] != sym["ast_hash"]
            or a_entry["lines"] != sym["lines"]
            or a_entry["file"] != sym["file"]
        ):
            print(
                f"[FAIL] Manual anchor drift for {uid}: code=({sym['file']}:{sym['lines']}@{sym['ast_hash']}) "
                f"vs manual=({a_entry['file']}:{a_entry['lines']}@{a_entry['ast_hash']})"
            )
            all_passed = False

        # 3. Check closing anchor
        if f"<!-- [{uid}:END] -->" not in manual_text:
            print(f"[FAIL] Closing anchor <!-- [{uid}:END] --> missing from BASE_CODE_ANATOMY_MANUAL.md")
            all_passed = False

    if all_passed:
        print("======================================================================")
        print(f"[VERIFY PASSED 100%] All {len(symbols)} [SYM-E2V-*] symbols are bidirectionally locked!")
        for s in symbols:
            print(
                f"  - [{s['uid']:<15}] {s['file']:<38} L{s['lines']:<9} ast_hash={s['ast_hash']}"
            )
        print("======================================================================")
    return all_passed


def main() -> None:
    parser = argparse.ArgumentParser(description="AST Bidirectional Code-Doc Symbiosis Sync Engine")
    parser.add_argument("--scan", action="store_true", help="Scan source files and print discovered symbols")
    parser.add_argument("--sync", action="store_true", help="Synchronize symbol_manifest.json & BASE_CODE_ANATOMY_MANUAL.md")
    parser.add_argument("--verify", action="store_true", help="Verify 100% AST hash & line range alignment")
    args = parser.parse_args()

    symbols = scan_all_symbols()
    if args.scan:
        print(f"[SCAN] Discovered {len(symbols)} [SYM-E2V-*] symbols:")
        for s in symbols:
            print(f"  * [{s['uid']}] {s['symbol_name']} ({s['file']}:{s['lines']}) ast_hash={s['ast_hash']}")

    if args.sync or (not args.scan and not args.verify):
        sync_files(symbols)
        print(f"[SYNC] Updated {MANIFEST_PATH} and {MANUAL_PATH} ({len(symbols)} symbols).")

    if args.verify or args.sync:
        ok = verify_bidirectional_alignment(symbols)
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
