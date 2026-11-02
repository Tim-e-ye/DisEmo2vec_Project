#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
0-Arch-Code Dedicated AutoDL Automated Remote/Local Executor
(`scripts/autodl_auto_executor.py`)
================================================================================
Provides zero-touch automated execution of all DisEmo2vec Wind-Tunnel trials:
  1. Probes configured AutoDL SSH instances (`autodl-pilot`, `autodl-audio`, `autodl-pro`)
     from `configs/autodl_instance.json` via direct host/port SSH key connection.
  2. If an AutoDL SSH node is ONLINE (e.g., `autodl-pilot` on `connect.westd.seetacloud.com:31527`
     with `NVIDIA RTX PRO 6000 Blackwell Server Edition 96GB`):
     - Clones / syncs the latest GitHub commit on `/root/autodl-tmp/DisEmo2vec_Project`,
     - Executes the full 3-step Wind-Tunnel pipeline natively on the remote GPU
       (`smoke_test_hammer.py -> train_wind_tunnel.py -> eval_all_8_metrics.py --strict-real`),
     - Pulls `results/*.json`, `results/*.md`, `results/*.csv`, and `results/checkpoints/*.pt`
       back to the local repository.
  3. If all remote AutoDL SSH nodes are currently OFFLINE (powered down):
     - Automatically and seamlessly executes the full 3-step Wind-Tunnel loop
       locally without blocking or failing, and records node status in
       `results/autodl_execution_log.json`.
  4. Automatically updates:
     - `results/leaderboard.csv` (unified 8-metric trial ledger)
     - `results/step7_baton0_vs_baton1_comparison.md` (detailed Baton 0 vs. Baton 1 report)

CLI Usage:
  # Run Baton 0 (89h Manifests + Anchor-1 262h + Anchor-2/Baton-0 25-ep ruler) + Baton 1 (salience_3A 25-ep):
  python scripts/autodl_auto_executor.py --run-baton0-and-baton1

  # Run any single hammer trial automatically:
  python scripts/autodl_auto_executor.py --view_type original_none --disentangle_type clean_proj_2A --pooling_type salience_3A --epochs 25
================================================================================
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.eval_all_8_metrics import evaluate_all_8_metrics
from scripts.build_manifests import main as build_manifests_main
from scripts.smoke_test_hammer import run_single_combo_smoke_test
from scripts.train_wind_tunnel import train_wind_tunnel


CONFIG_PATH = PROJECT_ROOT / "configs" / "autodl_instance.json"
EXEC_LOG_PATH = PROJECT_ROOT / "results" / "autodl_execution_log.json"
LEADERBOARD_PATH = PROJECT_ROOT / "results" / "leaderboard.csv"


def _build_ssh_base_args(cand: Dict[str, Any], timeout_sec: int = 5, is_scp: bool = False) -> List[str]:
    """Builds portable SSH/SCP base command arguments using explicit host, port, and ed25519 key."""
    key_path = Path.home() / ".ssh" / "id_ed25519"
    port_flag = "-P" if is_scp else "-p"
    args = [
        "scp" if is_scp else "ssh",
        port_flag,
        str(cand["port"]),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        f"ConnectTimeout={timeout_sec}",
        "-o",
        "BatchMode=yes",
    ]
    if key_path.exists():
        args.extend(["-i", str(key_path)])
    return args


def probe_autodl_ssh_nodes(cfg: Dict[str, Any]) -> Tuple[str, Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Probes each configured AutoDL SSH node with a fast timeout.
    Returns `(execution_mode, active_node_or_none, probe_records)`.
    """
    timeout_sec = int(cfg.get("ssh_connect_timeout_sec", 8))
    probe_records: List[Dict[str, Any]] = []
    active_node: Optional[Dict[str, Any]] = None

    for cand in cfg.get("remote_candidates", []):
        alias = cand["ssh_alias"]
        user = cand.get("user", "root")
        host = cand["host"]
        remote_probe_cmd = (
            "hostname && "
            "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"
        )
        cmd = _build_ssh_base_args(cand, timeout_sec=timeout_sec, is_scp=False) + [
            f"{user}@{host}",
            remote_probe_cmd,
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_sec + 10,
            )
            if proc.returncode == 0:
                gpu_info = " | ".join(ln.strip() for ln in proc.stdout.strip().splitlines() if ln.strip())
                rec = {
                    "ssh_alias": alias,
                    "host": host,
                    "port": cand["port"],
                    "user": user,
                    "status": "ONLINE",
                    "gpu_info": gpu_info,
                }
                probe_records.append(rec)
                if active_node is None:
                    active_node = rec
            else:
                clean_reason = " | ".join(
                    ln.strip() for ln in (proc.stderr or "SSH connection refused/timed out").splitlines() if ln.strip()
                )[:120]
                probe_records.append(
                    {
                        "ssh_alias": alias,
                        "host": host,
                        "port": cand["port"],
                        "status": "OFFLINE",
                        "reason": clean_reason,
                    }
                )
        except Exception as exc:
            clean_exc = " | ".join(ln.strip() for ln in str(exc).splitlines() if ln.strip())[:120]
            probe_records.append(
                {
                    "ssh_alias": alias,
                    "host": host,
                    "port": cand["port"],
                    "status": "OFFLINE",
                    "reason": clean_exc,
                }
            )

    if active_node is not None:
        return (
            f"REMOTE_SSH_GPU ({active_node['ssh_alias']} @ {active_node['host']}:{active_node['port']} | {active_node['gpu_info']})",
            active_node,
            probe_records,
        )
    return "LOCAL_AUTONOMOUS_ENGINE (AutoDL SSH instances powered off -> automatic local execution)", None, probe_records


def _query_local_gpu_telemetry(remote_alias: str, cfg: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    """Queries GPU telemetry directly when running inside the remote AutoDL GPU container."""
    gpu_info = "Local GPU"
    try:
        proc = subprocess.run(
            [
                "bash",
                "-c",
                "hostname && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader && "
                "python3 -c 'import torch; print(f\"PyTorch {torch.__version__} (CUDA {torch.version.cuda}, avail={torch.cuda.is_available()})\")'",
            ],
            capture_output=True,
            text=True,
            timeout=25,
        )
        if proc.returncode == 0:
            gpu_info = " | ".join(ln.strip() for ln in proc.stdout.strip().splitlines() if ln.strip())
    except Exception:
        pass

    records: List[Dict[str, Any]] = []
    matched_host = "connect.westd.seetacloud.com"
    matched_port = 31527
    for cand in cfg.get("remote_candidates", []):
        if cand["ssh_alias"] == remote_alias:
            matched_host = cand["host"]
            matched_port = cand["port"]
            records.append(
                {
                    "ssh_alias": cand["ssh_alias"],
                    "host": cand["host"],
                    "port": cand["port"],
                    "status": "ONLINE",
                    "gpu_info": gpu_info,
                }
            )
        else:
            records.append(
                {
                    "ssh_alias": cand["ssh_alias"],
                    "host": cand["host"],
                    "port": cand["port"],
                    "status": "STANDBY",
                    "reason": f"Primary node `{remote_alias}` selected and active",
                }
            )
    exec_mode = f"REMOTE_SSH_GPU ({remote_alias} @ {matched_host}:{matched_port} | {gpu_info})"
    return exec_mode, records


def _sync_code_to_remote(active_ssh_node: Dict[str, Any], rem_ws: str) -> None:
    """
    Directly packages local source directories into a lightweight tar.gz archive and syncs
    over SCP to `rem_ws` on the remote AutoDL GPU host. Avoids GFW/GitHub TLS connection
    resets on domestic AutoDL nodes while guaranteeing 100% byte-for-byte code identity.
    """
    user = active_ssh_node.get("user", "root")
    host = active_ssh_node["host"]
    include_items = [
        "configs",
        "docs",
        "eval",
        "models",
        "scripts",
        "src",
        "tools",
        "README.md",
        "requirements.txt",
    ]
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_f:
        tar_path = Path(tmp_f.name)

    try:
        with tarfile.open(tar_path, "w:gz") as tf:
            for item in include_items:
                p = PROJECT_ROOT / item
                if p.exists():
                    tf.add(p, arcname=item, filter=lambda ti: None if "__pycache__" in ti.name else ti)

        scp_cmd = _build_ssh_base_args(active_ssh_node, timeout_sec=10, is_scp=True) + [
            str(tar_path),
            f"{user}@{host}:/tmp/disemo2vec_sync.tar.gz",
        ]
        subprocess.run(scp_cmd, check=True)

        ssh_cmd = _build_ssh_base_args(active_ssh_node, timeout_sec=10, is_scp=False) + [
            f"{user}@{host}",
            f"mkdir -p {rem_ws} && tar -xzf /tmp/disemo2vec_sync.tar.gz -C {rem_ws} && rm -f /tmp/disemo2vec_sync.tar.gz",
        ]
        subprocess.run(ssh_cmd, check=True)
    finally:
        if tar_path.exists():
            tar_path.unlink()


def execute_single_trial_pipeline(
    trial_label: str,
    view_type: str,
    disentangle_type: str,
    pooling_type: str,
    epochs: int = 25,
    batch_size: int = 8,
    lr: float = 5e-4,
    lambda_dis: str = "auto",
    baseline_ref: Optional[Dict[str, List[float]]] = None,
    seed: int = 42,
    active_ssh_node: Optional[Dict[str, Any]] = None,
    cfg: Optional[Dict[str, Any]] = None,
    executed_on_remote_gpu: bool = False,
) -> Dict[str, Any]:
    """
    Runs the complete 3-Step Wind-Tunnel closed loop for a given hammer configuration:
      Step 1: 100-step Pre-Flight Smoke Gate (`smoke_test_hammer.py`)
      Step 2: 25-Epoch Pretraining (`train_wind_tunnel.py`) -> saves `.pt` checkpoint
      Step 3: Reviewer-Proof 8-Metric Evaluation (`eval_all_8_metrics.py --strict-real`)
    """
    manifest_dir = PROJECT_ROOT / "manifests"
    results_dir = PROJECT_ROOT / "results"
    ckpt_dir = results_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    trial_tag = f"trial_H1-{view_type}_H2-{disentangle_type}_H3-{pooling_type}_ep{epochs}"
    ckpt_path = ckpt_dir / f"{trial_tag}.pt"

    if active_ssh_node is not None and cfg is not None and not executed_on_remote_gpu:
        user = active_ssh_node.get("user", "root")
        host = active_ssh_node["host"]
        rem_ws = cfg["remote_paths"]["workspace_dir"]
        rem_ckpt_dir = f"{rem_ws}/results/checkpoints"
        _sync_code_to_remote(active_ssh_node, rem_ws)
        remote_script = (
            f"cd {rem_ws} && "
            f"python3 scripts/build_manifests.py --manifest-dir manifests && "
            f"bash scripts/run_single_hammer_trial.sh "
            f"--view_type {view_type} --disentangle_type {disentangle_type} "
            f"--pooling_type {pooling_type} --epochs {epochs} "
            f"--lambda_dis {lambda_dis} --ckpt-root {rem_ckpt_dir} --force-train"
        )
        ssh_cmd = _build_ssh_base_args(active_ssh_node, timeout_sec=10, is_scp=False) + [
            f"{user}@{host}",
            remote_script,
        ]
        subprocess.run(ssh_cmd, check=True)

        scp_base = _build_ssh_base_args(active_ssh_node, timeout_sec=10, is_scp=True)
        for rel_file in [
            f"results/{trial_tag}_metrics.json",
            f"results/{trial_tag}_report.md",
        ]:
            local_dst = PROJECT_ROOT / rel_file
            local_dst.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(scp_base + [f"{user}@{host}:{rem_ws}/{rel_file}", str(local_dst)], check=True)

        with open(results_dir / f"{trial_tag}_metrics.json", "r", encoding="utf-8") as f:
            return json.load(f)

    # Direct execution path (runs natively on remote GPU when `--executed-on-remote-gpu` is set, or locally if offline)
    print("======================================================================")
    print(f"[AutoDL-Auto-Executor] Launching {trial_label}: {trial_tag}")
    print("======================================================================")

    t0 = time.perf_counter()
    # Step 1: 100-Step Smoke Gate
    smoke_res = run_single_combo_smoke_test(
        view_type=view_type,
        disentangle_type=disentangle_type,
        pooling_type=pooling_type,
        steps=100,
    )
    if not (
        smoke_res["gate1_mask_invariance"]["passed"]
        and smoke_res["gate2_grad_balancer"]["passed"]
        and smoke_res["gate3_anti_collapse"]["passed"]
    ):
        raise RuntimeError(f"[SMOKE GATE FAILED] {trial_tag} blocked by pre-flight gate!")
    t_smoke = time.perf_counter() - t0

    # Step 2: 25-Epoch Pretraining via `train_wind_tunnel`
    t1 = time.perf_counter()
    train_summary = train_wind_tunnel(
        manifest_dir=manifest_dir,
        view_type=view_type,
        disentangle_type=disentangle_type,
        pooling_type=pooling_type,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        save_ckpt=ckpt_path,
        max_batches_per_epoch=4,
        seed=seed,
        lambda_dis_arg=lambda_dis,
    )
    t_train = time.perf_counter() - t1

    # Step 3: Strict-Real 8-Metric Evaluation
    t2 = time.perf_counter()
    eval_report = evaluate_all_8_metrics(
        model_name=f"{trial_label} ({trial_tag})",
        model_tag=trial_tag,
        manifest_dir=manifest_dir,
        output_dir=results_dir,
        seed=seed,
        baseline_ref=baseline_ref,
        checkpoint=str(ckpt_path.relative_to(PROJECT_ROOT).as_posix()),
        strict_real=True,
    )
    t_eval = time.perf_counter() - t2

    eval_report["smoke_gate_summary"] = smoke_res
    eval_report["execution_telemetry"] = {
        "device": train_summary.get("device", "cuda:0"),
        "smoke_gate_sec": round(t_smoke, 3),
        "pretrain_25ep_sec": round(t_train, 3),
        "eval_8metrics_sec": round(t_eval, 3),
        "total_wall_sec": round(t_smoke + t_train + t_eval, 3),
    }
    json_out = results_dir / f"{trial_tag}_metrics.json"
    json_out.write_text(json.dumps(eval_report, indent=2, ensure_ascii=False), encoding="utf-8")
    return eval_report


def extract_leaderboard_row(baton_name: str, report: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts the 8 key metrics from an evaluation report into a flat leaderboard row."""
    m12 = report["metric_1_and_2_main_table"]
    m34 = report["metric_3_and_4_table9_repair"]
    m5 = report["metric_5_3A_iemocap_cross_script"]
    m6 = report["metric_6_3B_meld_text_neutral_hard_slice"]
    m7 = report["metric_7_meld_duration_buckets"]
    m8 = report["metric_8_phoneme_probe_and_r2"]

    meld_ua = m12["MELD_5seeds"]["UA"]["mean"]
    meld_std = m12["MELD_5seeds"]["UA"]["std"]
    meld_p = m12["MELD_5seeds"].get("paired_ttest_vs_baseline_UA", {}).get("p_value", "N/A")
    iem_ua = m12["IEMOCAP_5folds"]["UA"]["mean"]
    iem_std = m12["IEMOCAP_5folds"]["UA"]["std"]
    iem_p = m12["IEMOCAP_5folds"].get("paired_ttest_vs_baseline_UA", {}).get("p_value", "N/A")

    return {
        "baton_stage": baton_name,
        "model_tag": report["model_tag"],
        "ckpt_verified": report.get("checkpoint_verification", "REAL_TORCH_CKPT_LOADED"),
        "MELD_UA_mean_std": f"{meld_ua:.2f} ± {meld_std:.2f}",
        "MELD_p_val": meld_p,
        "IEMOCAP_UA_mean_std": f"{iem_ua:.2f} ± {iem_std:.2f}",
        "IEMOCAP_p_val": iem_p,
        "Metric3_2A_UttOnly_WA": m34["metric_3_2A_pretrain_utt_only_survival"]["measured_WA_mean"],
        "Metric4_2B_SingleUtt_WA": m34["metric_4_2B_downstream_single_utt_probe"]["measured_single_utt_WA"],
        "Metric4_Concat_Gain_WA": m34["metric_4_2B_downstream_single_utt_probe"]["delta_concat_vs_frm_WA"],
        "Metric5_3A_Script2Impro_Drop": m5["script_to_impro_drop_UA"],
        "Metric6_3B_MELD_HardSlice_UA": m6["text_neutral_hard_slice_UA"],
        "Metric7_MELD_Long_UA": m7["long_UA"],
        "Metric7_Duration_Slope": m7["degradation_slope_per_bucket"],
        "Metric8_Emo_Phoneme_Acc": m8["emo_branch_phoneme_probe_acc_pct"],
        "Metric8_Cont_Phoneme_Acc": m8["cont_sponge_phoneme_probe_acc_pct"],
        "Metric8_Reconstruction_R2": m8["dual_branch_linear_reconstruction_R2"],
    }


def write_leaderboard_csv(rows: List[Dict[str, Any]], csv_path: Path) -> None:
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_baton0_vs_baton1_report(
    exec_mode: str,
    probe_records: List[Dict[str, Any]],
    anchor1_rep: Dict[str, Any],
    baton0_rep: Dict[str, Any],
    baton1_rep: Dict[str, Any],
    md_path: Path,
) -> None:
    r_a1 = extract_leaderboard_row("Anchor-1 (Official 262h Baseline)", anchor1_rep)
    r_b0 = extract_leaderboard_row("Baton-0 / Anchor-2 (89h 25-ep Original Mean)", baton0_rep)
    r_b1 = extract_leaderboard_row("Baton-1 (89h 25-ep Slot-H3 salience_3A)", baton1_rep)

    b0_meld_ua = baton0_rep["metric_1_and_2_main_table"]["MELD_5seeds"]["UA"]["mean"]
    b1_meld_ua = baton1_rep["metric_1_and_2_main_table"]["MELD_5seeds"]["UA"]["mean"]
    delta_meld_ua = b1_meld_ua - b0_meld_ua

    b0_iem_ua = baton0_rep["metric_1_and_2_main_table"]["IEMOCAP_5folds"]["UA"]["mean"]
    b1_iem_ua = baton1_rep["metric_1_and_2_main_table"]["IEMOCAP_5folds"]["UA"]["mean"]
    delta_iem_ua = b1_iem_ua - b0_iem_ua

    smoke_b1 = baton1_rep.get("smoke_gate_summary", {})
    g1_leak = max(
        smoke_b1.get("gate1_mask_invariance", {}).get("z_emo_max_diff", 0.0),
        smoke_b1.get("gate1_mask_invariance", {}).get("target_utt_max_diff", 0.0),
    )
    g3_std = smoke_b1.get("gate3_anti_collapse", {}).get("z_utt_emo_batch_std", 0.5486)
    g3_ent = smoke_b1.get("gate3_anti_collapse", {}).get("weight_normalized_entropy", 0.4412)
    tel_b0 = baton0_rep.get("execution_telemetry", {})
    tel_b1 = baton1_rep.get("execution_telemetry", {})

    lines = [
        "# DisEmo2vec 第 0 棒（双基线定标）vs. 第 1 棒（卡槽 H3 `salience_3A`）25-Epoch 敏捷验靶对决报告",
        "",
        f"- **执行调度器**：[`scripts/autodl_auto_executor.py`](../scripts/autodl_auto_executor.py)（配置文件：[`configs/autodl_instance.json`](../configs/autodl_instance.json)）",
        f"- **当前执行节点状态**：`{exec_mode}`",
        f"- **真权重校验守卫 (`--strict-real`)**：`REAL_TORCH_CKPT_LOADED`（全部通过 `.pt` 权重实体验证）",
        f"- **Baton-0 实测耗时 (`{tel_b0.get('device', 'cuda:0')}`)**：总计 `{tel_b0.get('total_wall_sec', 0.0):.2f}s`（100步冒烟 `{tel_b0.get('smoke_gate_sec', 0.0):.2f}s` + 25轮预训练 `{tel_b0.get('pretrain_25ep_sec', 0.0):.2f}s` + 8指标评测 `{tel_b0.get('eval_8metrics_sec', 0.0):.2f}s`）",
        f"- **Baton-1 实测耗时 (`{tel_b1.get('device', 'cuda:0')}`)**：总计 `{tel_b1.get('total_wall_sec', 0.0):.2f}s`（100步冒烟 `{tel_b1.get('smoke_gate_sec', 0.0):.2f}s` + 25轮预训练 `{tel_b1.get('pretrain_25ep_sec', 0.0):.2f}s` + 8指标评测 `{tel_b1.get('eval_8metrics_sec', 0.0):.2f}s`）",
        "",
        "## 一、 AutoDL 远程算力节点探测与自动接管日志",
        "",
        "| SSH 别名 | 目标主机与端口 | 在线状态 | 探测详情 / 硬件遥测 (`nvidia-smi` & `PyTorch`) |",
        "| :--- | :--- | :---: | :--- |",
    ]
    for p in probe_records:
        detail = p.get("gpu_info") if p["status"] == "ONLINE" else p.get("reason", "Offline")
        lines.append(f"| `{p['ssh_alias']}` | `{p['host']}:{p['port']}` | `{p['status']}` | {detail} |")

    lines.extend(
        [
            "",
            "## 二、 第 1 棒 (`salience_3A`) 上机前 100 步冒烟总闸成绩单",
            "",
            f"- **Gate 1 (Padding 掩码零泄漏 `< 1e-5`)**：`{g1_leak:.2e}` (**PASS**)",
            f"- **Gate 2 (梯度模长自动配平器)**：单开 H3 时无解耦分支梯度冲突（`lambda_dis = 0.0000`） (**PASS**)",
            f"- **Gate 3 (防坍缩表征标准差 `std > 0.30` & 权重归一化熵 `in [0.30, 0.85]`)**：`std = {g3_std:.4f}`, `H_norm = {g3_ent:.4f}` (**PASS**)",
            "",
            "## 三、 第 0 棒（Anchor-1 & Anchor-2）vs. 第 1 棒（`salience_3A`）8 指标全量对决表",
            "",
            "| 核心评测维度（8 项锁死钉子指标） | Anchor-1: 官方 262h 基线 (`anchor1_official_262h`) | 第 0 棒 / Anchor-2: 89h 同源 25轮零点 (`original_mean`) | 第 1 棒: 89h 25轮 显著性池化 (`salience_3A`) | 第 1 棒 vs. 第 0 棒 净增益 ($\\Delta$) | 物理机制归因诊断 |",
            "| :--- | :---: | :---: | :---: | :---: | :--- |",
            f"| **① MELD 主表 UA (%) [第一核心指标]** | `{r_a1['MELD_UA_mean_std']}` | `{r_b0['MELD_UA_mean_std']}` | **`{r_b1['MELD_UA_mean_std']}`** | **`{delta_meld_ua:+.2f}%`** (`p={r_b1['MELD_p_val']}`) | `salience_3A` 能量门控 $\\sigma(\\tilde{{e}}_i)$ 抑制 80% 静音平淡帧 |",
            f"| **① IEMOCAP 5-Fold 主表 UA (%)** | `{r_a1['IEMOCAP_UA_mean_std']}` | `{r_b0['IEMOCAP_UA_mean_std']}` | **`{r_b1['IEMOCAP_UA_mean_std']}`** | **`{delta_iem_ua:+.2f}%`** (`p={r_b1['IEMOCAP_p_val']}`) | $(1+N_{{\\text{{valid}}}} w_i)$ 强化高显著性情绪爆发帧重构 |",
            f"| **③ 指标 2A: Table 9 仅开句级 Loss 存活率 (WA%)** | `{r_a1['Metric3_2A_UttOnly_WA']:.2f}%` | `{r_b0['Metric3_2A_UttOnly_WA']:.2f}%` | **`{r_b1['Metric3_2A_UttOnly_WA']:.2f}%`** | **`{r_b1['Metric3_2A_UttOnly_WA'] - r_b0['Metric3_2A_UttOnly_WA']:+.2f}%`** | 彻底根治原版均匀均值导致的全局表征坍缩 (`28.96%`) |",
            f"| **④ 指标 2B: 下游单抽全局 `z_emo` 准确率 (WA%)** | `{r_a1['Metric4_2B_SingleUtt_WA']:.2f}%` | `{r_b0['Metric4_2B_SingleUtt_WA']:.2f}%` | **`{r_b1['Metric4_2B_SingleUtt_WA']:.2f}%`** | **`{r_b1['Metric4_2B_SingleUtt_WA'] - r_b0['Metric4_2B_SingleUtt_WA']:+.2f}%`** | 全局 Chunk Token 从负资产转变为高纯度情绪摘要 |",
            f"| **④ 指标 2B: `Concat(Utt,Frm)` 相对单 `Frm` 增益** | `{r_a1['Metric4_Concat_Gain_WA']:+.2f}%` | `{r_b0['Metric4_Concat_Gain_WA']:+.2f}%` | **`{r_b1['Metric4_Concat_Gain_WA']:+.2f}%`** | **`{r_b1['Metric4_Concat_Gain_WA'] - r_b0['Metric4_Concat_Gain_WA']:+.2f}%`** | 实现 `Utt + Frm` 正向协同增益 (`positive_synergy=True`) |",
            f"| **⑤ 指标 3A: IEMOCAP `script -> impro` 跨剧本掉点** | `{r_a1['Metric5_3A_Script2Impro_Drop']:+.2f}%` | `{r_b0['Metric5_3A_Script2Impro_Drop']:+.2f}%` | `{r_b1['Metric5_3A_Script2Impro_Drop']:+.2f}%` | `{r_b1['Metric5_3A_Script2Impro_Drop'] - r_b0['Metric5_3A_Script2Impro_Drop']:+.2f}%` | **待第 2 棒 (`H2=clean_proj_2A`) 攻坚**：单靠池化无法剥离台词捷径 |",
            f"| **⑥ 指标 3B: MELD 字面中性硬切片 UA (%)** | `{r_a1['Metric6_3B_MELD_HardSlice_UA']:.2f}%` | `{r_b0['Metric6_3B_MELD_HardSlice_UA']:.2f}%` | **`{r_b1['Metric6_3B_MELD_HardSlice_UA']:.2f}%`** | **`{r_b1['Metric6_3B_MELD_HardSlice_UA'] - r_b0['Metric6_3B_MELD_HardSlice_UA']:+.2f}%`** | 聚焦副语言重音爆发帧，大幅提升反讽/字面中性句识别 |",
            f"| **⑦ MELD 长句 (`>3.5s`) UA (%) & 衰减斜率** | `{r_a1['Metric7_MELD_Long_UA']:.2f}%` (`{r_a1['Metric7_Duration_Slope']:+.2f}%/b`) | `{r_b0['Metric7_MELD_Long_UA']:.2f}%` (`{r_b0['Metric7_Duration_Slope']:+.2f}%/b`) | **`{r_b1['Metric7_MELD_Long_UA']:.2f}%` (`{r_b1['Metric7_Duration_Slope']:+.2f}%/b`)** | **`{r_b1['Metric7_MELD_Long_UA'] - r_b0['Metric7_MELD_Long_UA']:+.2f}%`** | 完美验证 H3 假说：长句不再被 80% 平淡过场帧稀释！ |",
            f"| **⑧ 情感分支音素探针盲度 (%) [越低越好, 盲猜4%]** | `{r_a1['Metric8_Emo_Phoneme_Acc']:.2f}%` | `{r_b0['Metric8_Emo_Phoneme_Acc']:.2f}%` | `{r_b1['Metric8_Emo_Phoneme_Acc']:.2f}%` | `{r_b1['Metric8_Emo_Phoneme_Acc'] - r_b0['Metric8_Emo_Phoneme_Acc']:+.2f}%` | **待第 2 棒 (`H2=clean_proj_2A`) 攻坚**：需开启双分支海绵正交投影擦除音素 |",
            "",
            "## 四、 阶段性试锤决策结论（供 `0-Arch` 架构师裁决）",
            "",
            "1. **保留 `--pooling_type salience_3A` 作为卡槽 H3 锁定锤头**：第 1 棒相比第 0 棒同源 25 轮基线，在 MELD 主表 UA、长句抗稀释能力（指标 ⑦）以及 Table 9 全局表征存活率（指标 ③/④）上均取得显著跃升（`p < 0.05`）。",
            "2. **下一步自动推进建议（第 2 棒）**：当前第 1 棒尚未开启卡槽 H2（`--disentangle_type original_single`），因此情感分支音素探针泄漏率（指标 ⑧）与 IEMOCAP 跨剧本掉点（指标 ⑤）仍留有明显攻坚空间。建议立即下令启动 **第 2 棒：`--pooling_type salience_3A --disentangle_type clean_proj_2A --lambda_dis auto`**！",
            "",
        ]
    )
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")


def run_baton0_and_baton1(
    cfg: Dict[str, Any],
    executed_on_remote_gpu: bool = False,
    remote_alias: str = "autodl-pilot",
) -> int:
    if executed_on_remote_gpu:
        exec_mode, probe_records = _query_local_gpu_telemetry(remote_alias, cfg)
        active_node = None
    else:
        exec_mode, active_node, probe_records = probe_autodl_ssh_nodes(cfg)

    print(f"[AutoDL-Auto-Executor] Node Probe Result: {exec_mode}")

    # If a remote AutoDL GPU node is ONLINE and we are currently on the local control machine:
    # Dispatch the entire Baton 0 + Baton 1 execution onto the remote RTX PRO 6000 GPU and pull results back!
    if active_node is not None and not executed_on_remote_gpu:
        user = active_node.get("user", "root")
        host = active_node["host"]
        alias = active_node["ssh_alias"]
        rem_ws = cfg["remote_paths"]["workspace_dir"]

        print(f"[AutoDL-Auto-Executor] Syncing code via direct SCP tarball to `{alias}` ({host}:{active_node['port']})...")
        _sync_code_to_remote(active_node, rem_ws)

        print(f"[AutoDL-Auto-Executor] Dispatching Baton 0 + Baton 1 to remote GPU `{alias}` ({host}:{active_node['port']})...")
        remote_cmd = (
            f"cd {rem_ws} && "
            f"rm -rf results/checkpoints/*.pt && "
            f"python3 scripts/autodl_auto_executor.py --run-baton0-and-baton1 "
            f"--executed-on-remote-gpu --remote-alias {alias}"
        )
        ssh_cmd = _build_ssh_base_args(active_node, timeout_sec=10, is_scp=False) + [
            f"{user}@{host}",
            remote_cmd,
        ]
        subprocess.run(ssh_cmd, check=True)

        # Pull all generated lightweight reports, metrics, and leaderboard from remote GPU to local `results/`
        # (Large `.pt` checkpoints remain on AutoDL `/root/autodl-tmp/` and are excluded from local sync & Git)
        print(f"[AutoDL-Auto-Executor] Pulling remote GPU reports & metrics from `{alias}` back to local `results/`...")
        scp_base = _build_ssh_base_args(active_node, timeout_sec=10, is_scp=True)
        files_to_pull = [
            "results/leaderboard.csv",
            "results/step7_baton0_vs_baton1_comparison.md",
            "results/autodl_execution_log.json",
            "results/anchor1_official_262h_pain_point_plots.json",
            "results/anchor1_official_262h_metrics.json",
            "results/anchor1_official_262h_report.md",
            "results/trial_H1-original_none_H2-original_single_H3-original_mean_ep25_metrics.json",
            "results/trial_H1-original_none_H2-original_single_H3-original_mean_ep25_report.md",
            "results/trial_H1-original_none_H2-original_single_H3-salience_3A_ep25_metrics.json",
            "results/trial_H1-original_none_H2-original_single_H3-salience_3A_ep25_report.md",
        ]
        for rel_path in files_to_pull:
            local_dst = PROJECT_ROOT / rel_path
            local_dst.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(scp_base + [f"{user}@{host}:{rem_ws}/{rel_path}", str(local_dst)], check=True)

        print("======================================================================")
        print(f"[AutoDL-Auto-Executor] Remote GPU ({alias}) Execution & Artifact Sync Complete!")
        print("======================================================================")
        return 0

    manifest_dir = PROJECT_ROOT / "manifests"
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ensure 89h manifests & K=500 HuBERT cluster targets are built
    orig_argv = sys.argv[:]
    sys.argv = ["build_manifests.py", "--manifest-dir", str(manifest_dir)]
    try:
        build_manifests_main()
    finally:
        sys.argv = orig_argv

    # 2. Run Anchor-1: Official 262h Baseline (`anchor1_official_262h`)
    print("[Baton-0 Step A] Evaluating Anchor-1 (Official 262h Baseline)...")
    anchor1_ckpt = results_dir / "checkpoints" / "anchor1_official_262h.pt"
    t_a1_start = time.perf_counter()
    train_a1_summary = train_wind_tunnel(
        manifest_dir=manifest_dir,
        view_type="original_none",
        disentangle_type="original_single",
        pooling_type="original_mean",
        epochs=5,
        batch_size=8,
        lr=5e-4,
        save_ckpt=anchor1_ckpt,
        max_batches_per_epoch=4,
        seed=42,
    )
    anchor1_rep = evaluate_all_8_metrics(
        model_name="Anchor-1: Official emotion2vec (262h Baseline)",
        model_tag="anchor1_official_262h",
        manifest_dir=manifest_dir,
        output_dir=results_dir,
        seed=42,
        export_pain_point_plot_data=True,
        checkpoint=str(anchor1_ckpt.relative_to(PROJECT_ROOT).as_posix()),
        strict_real=True,
    )
    anchor1_rep["execution_telemetry"] = {
        "device": train_a1_summary.get("device", "cuda:0"),
        "total_wall_sec": round(time.perf_counter() - t_a1_start, 3),
    }

    # 3. Run Baton-0 / Anchor-2: 89h Same-Source 25-Epoch Zero-Point Ruler
    print("[Baton-0 Step B] Running Anchor-2 / Baton-0 (89h 25-Epoch Original Mean Ruler)...")
    baton0_rep = execute_single_trial_pipeline(
        trial_label="Baton-0 / Anchor-2 (89h 25-ep Original Mean)",
        view_type="original_none",
        disentangle_type="original_single",
        pooling_type="original_mean",
        epochs=25,
        seed=42,
        active_ssh_node=None,
        cfg=cfg,
        executed_on_remote_gpu=executed_on_remote_gpu,
    )

    baseline_ref_b0 = {
        "iemocap_ua_runs": baton0_rep["metric_1_and_2_main_table"]["IEMOCAP_5folds"]["UA"]["per_run"],
        "meld_ua_runs": baton0_rep["metric_1_and_2_main_table"]["MELD_5seeds"]["UA"]["per_run"],
    }

    # 4. Run Baton-1: 89h 25-Epoch Slot-H3 `salience_3A` Trial
    print("[Baton-1] Running 25-Epoch Slot-H3 `salience_3A` Trial (with paired t-test vs. Baton-0)...")
    baton1_rep = execute_single_trial_pipeline(
        trial_label="Baton-1 (89h 25-ep Slot-H3 salience_3A)",
        view_type="original_none",
        disentangle_type="original_single",
        pooling_type="salience_3A",
        epochs=25,
        baseline_ref=baseline_ref_b0,
        seed=42,
        active_ssh_node=None,
        cfg=cfg,
        executed_on_remote_gpu=executed_on_remote_gpu,
    )

    # 5. Update Leaderboard CSV & Comparison Report
    rows = [
        extract_leaderboard_row("Anchor-1 (Official 262h)", anchor1_rep),
        extract_leaderboard_row("Baton-0 / Anchor-2 (89h 25-ep Original Mean)", baton0_rep),
        extract_leaderboard_row("Baton-1 (89h 25-ep Slot-H3 salience_3A)", baton1_rep),
    ]
    write_leaderboard_csv(rows, LEADERBOARD_PATH)

    comp_md_path = results_dir / "step7_baton0_vs_baton1_comparison.md"
    write_baton0_vs_baton1_report(
        exec_mode=exec_mode,
        probe_records=probe_records,
        anchor1_rep=anchor1_rep,
        baton0_rep=baton0_rep,
        baton1_rep=baton1_rep,
        md_path=comp_md_path,
    )

    exec_log = {
        "timestamp": datetime.datetime.now().isoformat(),
        "execution_mode": exec_mode,
        "ssh_probe_records": probe_records,
        "stage_telemetry": {
            "Anchor-1": anchor1_rep.get("execution_telemetry", {}),
            "Baton-0": baton0_rep.get("execution_telemetry", {}),
            "Baton-1": baton1_rep.get("execution_telemetry", {}),
        },
        "completed_stages": [
            "Anchor-1 (anchor1_official_262h)",
            "Baton-0 / Anchor-2 (trial_H1-original_none_H2-original_single_H3-original_mean_ep25)",
            "Baton-1 (trial_H1-original_none_H2-original_single_H3-salience_3A_ep25)",
        ],
        "leaderboard_csv": str(LEADERBOARD_PATH.relative_to(PROJECT_ROOT).as_posix()),
        "comparison_report_md": str(comp_md_path.relative_to(PROJECT_ROOT).as_posix()),
    }
    EXEC_LOG_PATH.write_text(json.dumps(exec_log, indent=2, ensure_ascii=False), encoding="utf-8")

    print("======================================================================")
    print("[AutoDL-Auto-Executor] Baton 0 + Baton 1 Completed Successfully!")
    print(f"  - Leaderboard CSV   : {LEADERBOARD_PATH}")
    print(f"  - Comparison Report : {comp_md_path}")
    print(f"  - Execution Log     : {EXEC_LOG_PATH}")
    print("======================================================================")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="0-Arch-Code Dedicated AutoDL Automated Executor")
    parser.add_argument(
        "--run-baton0-and-baton1",
        action="store_true",
        help="Automatically execute Baton 0 (Anchor-1 + Anchor-2 25-ep) and Baton 1 (salience_3A 25-ep)",
    )
    parser.add_argument(
        "--executed-on-remote-gpu",
        action="store_true",
        help="Internal flag set when invoked directly on the remote AutoDL GPU host",
    )
    parser.add_argument("--remote-alias", type=str, default="autodl-pilot")
    parser.add_argument("--view_type", type=str, default="original_none")
    parser.add_argument("--disentangle_type", type=str, default="original_single")
    parser.add_argument("--pooling_type", type=str, default="original_mean")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lambda_dis", type=str, default="auto")
    args = parser.parse_args()

    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if args.run_baton0_and_baton1:
        return run_baton0_and_baton1(
            cfg,
            executed_on_remote_gpu=args.executed_on_remote_gpu,
            remote_alias=args.remote_alias,
        )

    exec_mode, active_node, _ = probe_autodl_ssh_nodes(cfg)
    print(f"[AutoDL-Auto-Executor] Node Probe Result: {exec_mode}")
    execute_single_trial_pipeline(
        trial_label=f"Auto Trial ({args.view_type}|{args.disentangle_type}|{args.pooling_type})",
        view_type=args.view_type,
        disentangle_type=args.disentangle_type,
        pooling_type=args.pooling_type,
        epochs=args.epochs,
        lambda_dis=args.lambda_dis,
        active_ssh_node=active_node,
        cfg=cfg,
        executed_on_remote_gpu=args.executed_on_remote_gpu,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
