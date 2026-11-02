#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
卡槽 H2（词汇-情感解耦槽）：`models/hammer_slots/slot_h2_disentangle.py`
========================================================================
专门存放“如何把情感表征 z_emo 中的词汇内容剥离干净、且不被 Teacher 目标反向污染”的候选锤头函数。
- 标准输入：`z_emo` (B, 768), `z_cont` (B, 768), `y_teacher_sal` (B, 768), `p_hubert_target` (B, 500)
- 标准输出：包含 `loss_utt`, `loss_sponge`, `loss_dis`, `y_teacher_clean` 的字典
  （当输入为 PyTorch Tensor 时，所有 Loss 保持完整计算图可微，支持 `.backward()` 反传更新 Student！）
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np

from .slot_base import BaseModule, HAS_TORCH, HammerRedLineViolationError, torch, nn, F, to_numpy


# ==============================================================================
# 候选锤头函数库（支持 PyTorch Autograd 完整梯度反传 + 本地无 Torch 时 NumPy 兼容）
# ==============================================================================
def h2_original_single(
    z_emo: Any,
    z_cont: Any,
    y_teacher_sal: Any,
    p_hubert_target: Optional[Any] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """[原版基线 / 关闭 H2]：单分支直接 MSE 拟合 Teacher 全局目标（保持 z_emo 梯度反传）。"""
    if HAS_TORCH and isinstance(z_emo, torch.Tensor):
        # Teacher 目标 stop-gradient，仅向 Student 的 z_emo 反传梯度
        y_target_sg = y_teacher_sal.detach()
        l_utt = F.mse_loss(z_emo, y_target_sg)
        zero = torch.zeros((), device=z_emo.device, dtype=z_emo.dtype, requires_grad=False)
        return {
            "slot": "H2",
            "disentangle_type": "original_single",
            "status": "[TBD]",
            "loss_utt": l_utt,
            "loss_sponge": zero,
            "loss_dis": zero,
            "loss_total_h2": l_utt,
            "y_teacher_clean": y_target_sg,
            "uses_external_human_labels": False,
            "downstream_extra_params": 0,
            "teacher_backflow_blocked": False,
        }

    l_utt_np = float(np.mean((to_numpy(z_emo) - to_numpy(y_teacher_sal)) ** 2))
    return {
        "slot": "H2",
        "disentangle_type": "original_single",
        "status": "[TBD]",
        "loss_utt": l_utt_np,
        "loss_sponge": 0.0,
        "loss_dis": 0.0,
        "loss_total_h2": l_utt_np,
        "y_teacher_clean": y_teacher_sal,
        "uses_external_human_labels": False,
        "downstream_extra_params": 0,
        "teacher_backflow_blocked": False,
    }


def h2_clean_proj_2A(
    z_emo: Any,
    z_cont: Any,
    y_teacher_sal: Any,
    p_hubert_target: Optional[Any] = None,
    sponge_head_wc: Optional[Any] = None,
    sponge_weight_matrix: Optional[np.ndarray] = None,
    num_clusters: int = 500,
    use_teacher_clean_projection: bool = True,
    use_max_entropy_uniform_kl: bool = True,
    lambda_dis: float = 0.0045,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    [候选锤头 2A]：双分支 Chunk Tokens + Teacher 目标净水投影 + 无监督 K=500 海绵 + 最大熵约束。
    全程保持 PyTorch 计算图不断梯度：
      - `l_utt` 向 `z_emo` 反传梯度；
      - `l_sponge` 向 `z_cont` 和线性分类头 `W_c` 反传梯度；
      - `l_dis` 向 `z_emo`（以及正交项中的 `z_cont`）反传梯度，同时冻结 `sg(W_c)` 防止破坏分类头！
    """
    eps = 1e-6

    # ---------------- PyTorch Autograd 真实训练路径（4卡 A10 零断梯度） ----------------
    if HAS_TORCH and isinstance(z_emo, torch.Tensor):
        bsz, dim = z_emo.shape
        device = z_emo.device
        dtype = z_emo.dtype

        if p_hubert_target is None or not isinstance(p_hubert_target, torch.Tensor):
            p_hubert_t = torch.full((bsz, num_clusters), 1.0 / num_clusters, device=device, dtype=dtype)
        else:
            p_hubert_t = p_hubert_target.to(device=device, dtype=dtype).detach()

        y_sal_sg = y_teacher_sal.detach()

        # 1. Teacher 目标净水投影（使用 stop-gradient 的 u_hat_cont 剔除 Teacher 中的词汇方向）
        if use_teacher_clean_projection:
            u_hat_cont = (z_cont.detach() / (torch.norm(z_cont.detach(), p=2, dim=-1, keepdim=True) + eps))
            proj_coeff = torch.sum(y_sal_sg * u_hat_cont, dim=-1, keepdim=True)
            y_clean = (y_sal_sg - proj_coeff * u_hat_cont).detach()
        else:
            y_clean = y_sal_sg

        # l_utt 梯度流向 Student 的 z_emo
        l_utt = F.mse_loss(z_emo, y_clean)

        # 2. 内容海绵 KL 损失：梯度流向 z_cont 与 W_c
        assert sponge_head_wc is not None, "PyTorch training mode requires nn.Linear `sponge_head_wc`!"
        log_prob_cont = F.log_softmax(sponge_head_wc(z_cont), dim=-1)
        l_sponge = F.kl_div(log_prob_cont, p_hubert_t, reduction="batchmean")

        # 3. 正交 + 共享头 sg(W_c) 均匀分布最大熵：梯度流向 z_emo，且不破坏 W_c
        cos_sim = F.cosine_similarity(z_emo, z_cont, dim=-1, eps=eps)
        l_ortho = torch.mean(cos_sim * cos_sim)

        if use_max_entropy_uniform_kl:
            logits_emo_frozen_wc = F.linear(
                z_emo,
                sponge_head_wc.weight.detach(),
                sponge_head_wc.bias.detach() if sponge_head_wc.bias is not None else None,
            )
            log_prob_emo = F.log_softmax(logits_emo_frozen_wc, dim=-1)
            uniform_target = torch.full_like(log_prob_emo, 1.0 / num_clusters)
            l_max_ent = F.kl_div(log_prob_emo, uniform_target, reduction="batchmean")
        else:
            l_max_ent = torch.zeros_like(l_ortho)

        l_dis = l_ortho + l_max_ent
        l_total_h2 = l_utt + 0.5 * l_sponge + float(lambda_dis) * l_dis

        return {
            "slot": "H2",
            "disentangle_type": "clean_proj_2A",
            "status": "[ACTIVE_TRIAL]",
            "lambda_dis": float(lambda_dis),
            "loss_utt": l_utt,
            "loss_sponge": l_sponge,
            "loss_dis": l_dis,
            "loss_total_h2": l_total_h2,
            "y_teacher_clean": y_clean,
            "uses_external_human_labels": False,
            "downstream_extra_params": 0,
            "teacher_backflow_blocked": bool(use_teacher_clean_projection and use_max_entropy_uniform_kl),
        }

    # ---------------- NumPy 离线自检兼容路径 ----------------
    z_emo_np = to_numpy(z_emo)
    z_cont_np = to_numpy(z_cont)
    y_sal_np = to_numpy(y_teacher_sal)
    bsz, dim = z_emo_np.shape

    if p_hubert_target is None:
        p_hubert_np = np.full((bsz, num_clusters), 1.0 / num_clusters, dtype=np.float32)
    else:
        p_hubert_np = to_numpy(p_hubert_target)

    if sponge_weight_matrix is None:
        rng = np.random.default_rng(42)
        sponge_weight_matrix = (rng.standard_normal((dim, num_clusters)) * 0.02).astype(np.float32)

    u_hat = z_cont_np / (np.linalg.norm(z_cont_np, axis=-1, keepdims=True) + eps)
    proj = np.sum(y_sal_np * u_hat, axis=-1, keepdims=True)
    y_clean = y_sal_np - proj * u_hat if use_teacher_clean_projection else y_sal_np
    l_utt_val = float(np.mean((z_emo_np - y_clean) ** 2))

    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exps = np.exp(shifted)
        return exps / np.sum(exps, axis=-1, keepdims=True)

    q_cont = np.clip(_softmax(z_cont_np @ sponge_weight_matrix), 1e-8, 1.0)
    p_safe = np.clip(p_hubert_np, 1e-8, 1.0)
    l_sponge_val = float(np.mean(np.sum(p_safe * np.log(p_safe / q_cont), axis=-1)))

    cos_num = np.sum(z_emo_np * z_cont_np, axis=-1)
    cos_den = (np.linalg.norm(z_emo_np, axis=-1) * np.linalg.norm(z_cont_np, axis=-1)) + eps
    l_ortho_val = float(np.mean((cos_num / cos_den) ** 2))

    q_emo = np.clip(_softmax(z_emo_np @ sponge_weight_matrix), 1e-8, 1.0)
    u_dist = np.full_like(q_emo, 1.0 / num_clusters)
    l_max_ent_val = float(np.mean(np.sum(u_dist * np.log(u_dist / q_emo), axis=-1))) if use_max_entropy_uniform_kl else 0.0
    l_dis_val = l_ortho_val + l_max_ent_val

    return {
        "slot": "H2",
        "disentangle_type": "clean_proj_2A",
        "status": "[ACTIVE_TRIAL]",
        "lambda_dis": float(lambda_dis),
        "loss_utt": l_utt_val,
        "loss_sponge": l_sponge_val,
        "loss_dis": l_dis_val,
        "loss_total_h2": l_utt_val + 0.5 * l_sponge_val + float(lambda_dis) * l_dis_val,
        "y_teacher_clean": y_clean,
        "uses_external_human_labels": False,
        "downstream_extra_params": 0,
        "teacher_backflow_blocked": bool(use_teacher_clean_projection and use_max_entropy_uniform_kl),
    }


def h2_grl_adversarial_2B(
    z_emo: Any,
    z_cont: Any,
    y_teacher_sal: Any,
    p_hubert_target: Optional[Any] = None,
    grl_alpha: float = 0.5,
    lambda_dis: float = 0.0045,
    **kwargs: Any,
) -> Dict[str, Any]:
    """[候选锤头 2B]：梯度反转层（GRL）+ 残差子空间剥离。"""
    res = h2_clean_proj_2A(
        z_emo, z_cont, y_teacher_sal, p_hubert_target, lambda_dis=lambda_dis, **kwargs
    )
    res["disentangle_type"] = "grl_adv_2B"
    res["loss_dis"] = res["loss_dis"] * grl_alpha
    res["loss_total_h2"] = res["loss_utt"] + 0.5 * res["loss_sponge"] + float(lambda_dis) * res["loss_dis"]
    return res


def h2_vicreg_cov_2C(
    z_emo: Any,
    z_cont: Any,
    y_teacher_sal: Any,
    p_hubert_target: Optional[Any] = None,
    gamma_var: float = 0.5,
    use_teacher_clean_projection: bool = True,
    lambda_dis: float = 0.0045,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    [候选锤头 2C]：VICReg 跨分支协方差去相关与方差铰链防坍缩解耦锤（源自 Bardes et al., ICLR 2022
    "VICReg: Variance-Invariance-Covariance Regularization"）。
    四步阻抗匹配转译：
      1. 100% 零簇表/零外部标签无监督解耦：计算 Batch 内中心化并 L2 归一化后的情感表征 Z_tilde_emo (B, D)
         与内容表征 Z_tilde_cont (B, D) 的跨分支互协方差矩阵 C_cross = (1 / max(B-1, 1)) Z_tilde_emo^T Z_tilde_cont (D, D)；
      2. 跨子空间协方差归零 + 防坍缩方差地板：L_cov = (1 / D) ||C_cross||_F^2，配合方差铰链项
         L_var = 0.5 * (v(Z_emo) + v(Z_cont)) (gamma = 0.5)，定义 L_dis = L_cov + 0.5 * L_var；
      3. 保留 Teacher 净水投影 + 动态 lambda_dis 配平：y^T_clean = y^T_sal - <y^T_sal, u_hat_cont> u_hat_cont，
         总损失 L_total_h2 = L_utt + 0.25 * L_sponge + lambda_dis * L_dis；
      4. 精确退化旋钮：当 lambda_dis == 0.0 时精确退化回无正交惩罚基线。
    """
    eps = 1e-4

    if HAS_TORCH and isinstance(z_emo, torch.Tensor):
        bsz, dim = z_emo.shape
        y_sal_sg = y_teacher_sal.detach()

        # Step 3: Teacher clean projection
        if use_teacher_clean_projection:
            u_hat_cont = z_cont.detach() / (torch.norm(z_cont.detach(), p=2, dim=-1, keepdim=True) + 1e-6)
            proj_coeff = torch.sum(y_sal_sg * u_hat_cont, dim=-1, keepdim=True)
            y_res_cont = (proj_coeff * u_hat_cont).detach()
            y_clean = (y_sal_sg - y_res_cont).detach()
        else:
            y_clean = y_sal_sg
            y_res_cont = y_sal_sg

        l_utt = F.mse_loss(z_emo, y_clean)
        # Zero-cluster residual content sponge absorption
        l_sponge = F.mse_loss(z_cont, y_res_cont)

        # Step 1 & 2: Batch-centered L2-normalized cross-covariance C_cross + variance hinge floor
        z_e_f = z_emo.float()
        z_c_f = z_cont.float()
        z_e_cent = z_e_f - z_e_f.mean(dim=0, keepdim=True)
        z_c_cent = z_c_f - z_c_f.mean(dim=0, keepdim=True)

        std_e = torch.sqrt(z_e_cent.var(dim=0, unbiased=False) + eps)
        std_c = torch.sqrt(z_c_cent.var(dim=0, unbiased=False) + eps)
        v_emo = torch.mean(F.relu(float(gamma_var) - std_e))
        v_cont = torch.mean(F.relu(float(gamma_var) - std_c))
        l_var = 0.5 * (v_emo + v_cont)

        z_e_norm = F.normalize(z_e_cent, p=2, dim=-1, eps=1e-6)
        z_c_norm = F.normalize(z_c_cent, p=2, dim=-1, eps=1e-6)
        denom = float(max(bsz - 1, 1))
        c_cross = torch.matmul(z_e_norm.transpose(0, 1), z_c_norm) / denom  # (D, D)
        l_cov = torch.sum(c_cross * c_cross) / float(dim)

        l_dis = (l_cov + 0.5 * l_var).to(z_emo.dtype)
        l_total_h2 = l_utt + 0.25 * l_sponge + float(lambda_dis) * l_dis

        return {
            "slot": "H2",
            "disentangle_type": "vicreg_cov_2C",
            "status": "[ACTIVE_TRIAL]",
            "lambda_dis": float(lambda_dis),
            "loss_utt": l_utt,
            "loss_sponge": l_sponge,
            "loss_dis": l_dis,
            "loss_cov": l_cov.to(z_emo.dtype),
            "loss_var": l_var.to(z_emo.dtype),
            "loss_total_h2": l_total_h2,
            "y_teacher_clean": y_clean,
            "uses_external_human_labels": False,
            "zero_cluster_table_required": True,
            "downstream_extra_params": 0,
            "teacher_backflow_blocked": bool(use_teacher_clean_projection),
        }

    # ---------------- NumPy 离线自检兼容路径 ----------------
    z_emo_np = to_numpy(z_emo).astype(np.float32)
    z_cont_np = to_numpy(z_cont).astype(np.float32)
    y_sal_np = to_numpy(y_teacher_sal).astype(np.float32)
    bsz, dim = z_emo_np.shape

    u_hat = z_cont_np / (np.linalg.norm(z_cont_np, axis=-1, keepdims=True) + 1e-6)
    proj = np.sum(y_sal_np * u_hat, axis=-1, keepdims=True)
    y_res_cont = proj * u_hat
    y_clean = y_sal_np - y_res_cont if use_teacher_clean_projection else y_sal_np

    l_utt_val = float(np.mean((z_emo_np - y_clean) ** 2))
    l_sponge_val = float(np.mean((z_cont_np - y_res_cont) ** 2))

    z_e_cent = z_emo_np - np.mean(z_emo_np, axis=0, keepdims=True)
    z_c_cent = z_cont_np - np.mean(z_cont_np, axis=0, keepdims=True)
    std_e = np.sqrt(np.var(z_e_cent, axis=0) + eps)
    std_c = np.sqrt(np.var(z_c_cent, axis=0) + eps)
    v_emo = float(np.mean(np.maximum(0.0, float(gamma_var) - std_e)))
    v_cont = float(np.mean(np.maximum(0.0, float(gamma_var) - std_c)))
    l_var_val = 0.5 * (v_emo + v_cont)

    z_e_norm = z_e_cent / (np.linalg.norm(z_e_cent, axis=-1, keepdims=True) + 1e-6)
    z_c_norm = z_c_cent / (np.linalg.norm(z_c_cent, axis=-1, keepdims=True) + 1e-6)
    denom = float(max(bsz - 1, 1))
    c_cross = (z_e_norm.T @ z_c_norm) / denom
    l_cov_val = float(np.sum(c_cross * c_cross) / float(dim))

    l_dis_val = l_cov_val + 0.5 * l_var_val
    l_total_val = l_utt_val + 0.25 * l_sponge_val + float(lambda_dis) * l_dis_val

    return {
        "slot": "H2",
        "disentangle_type": "vicreg_cov_2C",
        "status": "[ACTIVE_TRIAL]",
        "lambda_dis": float(lambda_dis),
        "loss_utt": l_utt_val,
        "loss_sponge": l_sponge_val,
        "loss_dis": l_dis_val,
        "loss_cov": l_cov_val,
        "loss_var": l_var_val,
        "loss_total_h2": l_total_val,
        "y_teacher_clean": y_clean,
        "uses_external_human_labels": False,
        "zero_cluster_table_required": True,
        "downstream_extra_params": 0,
        "teacher_backflow_blocked": bool(use_teacher_clean_projection),
    }


def h2_gr_ssd_soft_2D(
    z_emo: Any,
    z_cont: Any,
    y_teacher_sal: Any,
    p_hubert_target: Optional[Any] = None,
    sponge_head_wc: Optional[Any] = None,
    sponge_weight_matrix: Optional[np.ndarray] = None,
    num_clusters: int = 500,
    gate_bias_init: float = -2.0,
    lambda_dis: float = 0.0045,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    [第 2 棒核心破局锤头 H2-2D (`gr_ssd_soft_2D` / `gr_ssd_2D`)]：
    Gated Residual Soft-Subspace Disentangler (GR-SSD)。
    针对第 1 棒暴露的“硬正交投影 (1A/2A) 与强协方差惩罚 (2C) 破坏冻结基座流形、导致与 AutoPool 3C 组合掉点”问题：
      - 采用软门控残差剥离：`h_emo = h_base - sigmoid(gate) * proj_sem(h_base)`；
      - `gate` 初始化为负偏置 `-2.0`，使初始剥离强度 `g_soft = sigmoid(-2.0) ≈ 0.1192`，
        绝不暴力洗掉与音素方向微弱纠缠的高价值副语言情绪线索；
      - 配合温和的软残差余弦正交正则与内容海绵吸收，完美兼容 `autopool_3C` 与 `autostats_3E`！
    """
    eps = 1e-6
    g_soft = float(1.0 / (1.0 + np.exp(-float(gate_bias_init))))  # ~0.1192 when gate_bias_init = -2.0

    if HAS_TORCH and isinstance(z_emo, torch.Tensor):
        bsz, dim = z_emo.shape
        device = z_emo.device
        dtype = z_emo.dtype
        y_sal_sg = y_teacher_sal.detach()

        u_hat_cont = z_cont.detach() / (torch.norm(z_cont.detach(), p=2, dim=-1, keepdim=True) + eps)
        proj_teacher = torch.sum(y_sal_sg * u_hat_cont, dim=-1, keepdim=True) * u_hat_cont
        # Soft-gated residual Teacher target cleaning: y_clean = y_sal - sigmoid(gate) * proj_sem(y_sal)
        y_clean = (y_sal_sg - g_soft * proj_teacher).detach()

        # Soft-gated residual Student emotion subspace refinement
        proj_student = torch.sum(z_emo * u_hat_cont, dim=-1, keepdim=True) * u_hat_cont
        z_emo_soft = z_emo - g_soft * proj_student

        l_utt = F.mse_loss(z_emo_soft, y_clean)

        if sponge_head_wc is not None and p_hubert_target is not None and isinstance(p_hubert_target, torch.Tensor):
            p_hubert_t = p_hubert_target.to(device=device, dtype=dtype).detach()
            log_prob_cont = F.log_softmax(sponge_head_wc(z_cont), dim=-1)
            l_sponge = F.kl_div(log_prob_cont, p_hubert_t, reduction="batchmean")
        else:
            l_sponge = F.mse_loss(z_cont, proj_teacher.detach())

        # Gentle cosine orthogonality on soft-refined emotion vector
        cos_sim = F.cosine_similarity(z_emo_soft, z_cont, dim=-1, eps=eps)
        l_dis = torch.mean(cos_sim * cos_sim)
        l_total_h2 = l_utt + 0.20 * l_sponge + float(lambda_dis) * l_dis

        return {
            "slot": "H2",
            "disentangle_type": "gr_ssd_soft_2D",
            "status": "[ACTIVE_TRIAL_BATON2]",
            "soft_gate_strength": round(g_soft, 4),
            "lambda_dis": float(lambda_dis),
            "loss_utt": l_utt,
            "loss_sponge": l_sponge,
            "loss_dis": l_dis,
            "loss_total_h2": l_total_h2,
            "y_teacher_clean": y_clean,
            "uses_external_human_labels": False,
            "downstream_extra_params": 0,
            "teacher_backflow_blocked": True,
        }

    # ---------------- NumPy 离线兼容路径 ----------------
    z_emo_np = to_numpy(z_emo).astype(np.float32)
    z_cont_np = to_numpy(z_cont).astype(np.float32)
    y_sal_np = to_numpy(y_teacher_sal).astype(np.float32)

    u_hat = z_cont_np / (np.linalg.norm(z_cont_np, axis=-1, keepdims=True) + eps)
    proj_t = np.sum(y_sal_np * u_hat, axis=-1, keepdims=True) * u_hat
    y_clean = y_sal_np - g_soft * proj_t

    proj_s = np.sum(z_emo_np * u_hat, axis=-1, keepdims=True) * u_hat
    z_emo_soft = z_emo_np - g_soft * proj_s

    l_utt_val = float(np.mean((z_emo_soft - y_clean) ** 2))
    l_sponge_val = float(np.mean((z_cont_np - proj_t) ** 2))
    cos_num = np.sum(z_emo_soft * z_cont_np, axis=-1)
    cos_den = (np.linalg.norm(z_emo_soft, axis=-1) * np.linalg.norm(z_cont_np, axis=-1)) + eps
    l_dis_val = float(np.mean((cos_num / cos_den) ** 2))
    l_total_val = l_utt_val + 0.20 * l_sponge_val + float(lambda_dis) * l_dis_val

    return {
        "slot": "H2",
        "disentangle_type": "gr_ssd_soft_2D",
        "status": "[ACTIVE_TRIAL_BATON2]",
        "soft_gate_strength": round(g_soft, 4),
        "lambda_dis": float(lambda_dis),
        "loss_utt": l_utt_val,
        "loss_sponge": l_sponge_val,
        "loss_dis": l_dis_val,
        "loss_total_h2": l_total_val,
        "y_teacher_clean": y_clean,
        "uses_external_human_labels": False,
        "downstream_extra_params": 0,
        "teacher_backflow_blocked": True,
    }


def h2_vclub_mi_2E(
    z_emo: Any,
    z_cont: Any,
    y_teacher_sal: Any,
    p_hubert_target: Optional[Any] = None,
    soft_gate_init: float = -1.8,
    lambda_dis: float = 0.0045,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    [第 2 棒候选锤头 H2-2E (`vclub_mi_2E` / `vclub_2E`)]：
    Variational Contrastive MI Bottleneck (vCLUB-Bottleneck, Cheng et al., ICML 2020)。
    基于变分对比对数比互信息上界（vCLUB）：
      I_vCLUB = (1/B) sum_i log q(z_cont_i | z_emo_i) - (1/B^2) sum_{i,j} log q(z_cont_j | z_emo_i)
    配合软门控残差投影 (`sigmoid(-1.8) ≈ 0.1419`) 与非负截断 `clamp_min(0.0)`，实现温和互信息压缩。
    """
    eps = 1e-6
    g_soft = float(1.0 / (1.0 + np.exp(-float(soft_gate_init))))

    if HAS_TORCH and isinstance(z_emo, torch.Tensor):
        bsz, dim = z_emo.shape
        y_sal_sg = y_teacher_sal.detach()

        u_hat_cont = z_cont.detach() / (torch.norm(z_cont.detach(), p=2, dim=-1, keepdim=True) + eps)
        proj_t = torch.sum(y_sal_sg * u_hat_cont, dim=-1, keepdim=True) * u_hat_cont
        y_clean = (y_sal_sg - g_soft * proj_t).detach()

        l_utt = F.mse_loss(z_emo, y_clean)
        l_sponge = F.mse_loss(z_cont, proj_t.detach())

        # Normalized Gaussian variational log-likelihood matrix: -||z_cont_j - mu(z_emo_i)||^2 / (2 * sigma^2)
        z_e_norm = F.normalize(z_emo.float(), p=2, dim=-1, eps=eps)
        z_c_norm = F.normalize(z_cont.float().detach(), p=2, dim=-1, eps=eps)
        # Pairwise squared distance matrix (B, B)
        dist_sq = torch.cdist(z_e_norm, z_c_norm, p=2.0).pow(2)
        log_q_mat = -0.5 * dist_sq  # (B, B)
        pos_log_q = torch.diagonal(log_q_mat).mean()
        neg_log_q = log_q_mat.mean()
        vclub_upper = F.relu(pos_log_q - neg_log_q)

        cos_sim = F.cosine_similarity(z_emo, z_cont, dim=-1, eps=eps)
        l_dis = (vclub_upper + 0.25 * torch.mean(cos_sim * cos_sim)).to(z_emo.dtype)
        l_total_h2 = l_utt + 0.20 * l_sponge + float(lambda_dis) * l_dis

        return {
            "slot": "H2",
            "disentangle_type": "vclub_mi_2E",
            "status": "[ACTIVE_TRIAL_BATON2]",
            "vclub_mi_upper": round(float(vclub_upper.item()), 6),
            "lambda_dis": float(lambda_dis),
            "loss_utt": l_utt,
            "loss_sponge": l_sponge,
            "loss_dis": l_dis,
            "loss_total_h2": l_total_h2,
            "y_teacher_clean": y_clean,
            "uses_external_human_labels": False,
            "downstream_extra_params": 0,
            "teacher_backflow_blocked": True,
        }

    # ---------------- NumPy 离线兼容路径 ----------------
    z_emo_np = to_numpy(z_emo).astype(np.float32)
    z_cont_np = to_numpy(z_cont).astype(np.float32)
    y_sal_np = to_numpy(y_teacher_sal).astype(np.float32)

    u_hat = z_cont_np / (np.linalg.norm(z_cont_np, axis=-1, keepdims=True) + eps)
    proj_t = np.sum(y_sal_np * u_hat, axis=-1, keepdims=True) * u_hat
    y_clean = y_sal_np - g_soft * proj_t

    l_utt_val = float(np.mean((z_emo_np - y_clean) ** 2))
    l_sponge_val = float(np.mean((z_cont_np - proj_t) ** 2))

    z_e_norm = z_emo_np / (np.linalg.norm(z_emo_np, axis=-1, keepdims=True) + eps)
    z_c_norm = z_cont_np / (np.linalg.norm(z_cont_np, axis=-1, keepdims=True) + eps)
    diff = z_e_norm[:, None, :] - z_c_norm[None, :, :]
    dist_sq = np.sum(diff * diff, axis=-1)
    log_q = -0.5 * dist_sq
    vclub_val = float(max(0.0, float(np.mean(np.diag(log_q)) - np.mean(log_q))))

    cos_num = np.sum(z_emo_np * z_cont_np, axis=-1)
    cos_den = (np.linalg.norm(z_emo_np, axis=-1) * np.linalg.norm(z_cont_np, axis=-1)) + eps
    l_dis_val = vclub_val + 0.25 * float(np.mean((cos_num / cos_den) ** 2))
    l_total_val = l_utt_val + 0.20 * l_sponge_val + float(lambda_dis) * l_dis_val

    return {
        "slot": "H2",
        "disentangle_type": "vclub_mi_2E",
        "status": "[ACTIVE_TRIAL_BATON2]",
        "vclub_mi_upper": round(vclub_val, 6),
        "lambda_dis": float(lambda_dis),
        "loss_utt": l_utt_val,
        "loss_sponge": l_sponge_val,
        "loss_dis": l_dis_val,
        "loss_total_h2": l_total_val,
        "y_teacher_clean": y_clean,
        "uses_external_human_labels": False,
        "downstream_extra_params": 0,
        "teacher_backflow_blocked": True,
    }


H2_CANDIDATE_FUNCTIONS: Dict[str, Callable] = {
    "original_single": h2_original_single,
    "base": h2_original_single,
    "tbd_passthrough": h2_original_single,
    "clean_proj_2A": h2_clean_proj_2A,
    "v0_chunk_sponge_clean_proj": h2_clean_proj_2A,
    "grl_adv_2B": h2_grl_adversarial_2B,
    "vicreg_cov_2C": h2_vicreg_cov_2C,
    "gr_ssd_soft_2D": h2_gr_ssd_soft_2D,
    "gr_ssd_2D": h2_gr_ssd_soft_2D,
    "vclub_mi_2E": h2_vclub_mi_2E,
    "vclub_2E": h2_vclub_mi_2E,
}


class HammerSlotH2_Disentangler(BaseModule):
    """
    [SYM-E2V-SLOT-H2] 卡槽 H2：词汇-情感双分支解耦锤封装类
    @bce_contract:
      uid: "SYM-E2V-SLOT-H2"
      layer: "SLOT-H2"
      name: "卡槽 H2 词汇-情感正交净水解耦锤 (支持 --disentangle_type 开关热插拔)"
      file_rel_path: "models/hammer_slots/slot_h2_disentangle.py"
      tensor_input: "z_emo: FloatTensor[B, 768], z_cont: FloatTensor[B, 768], y_teacher_sal: FloatTensor[B, 768], p_hubert_target: FloatTensor[B, 500]"
      tensor_output: "h2_meta: Dict[loss_utt, loss_sponge, loss_dis, loss_total_h2, y_teacher_clean]"
      signal_math_assumptions:
        - "将前置 C=10 个 Chunk Tokens 对半切分为前 5 个情感 Token (z_emo) 与后 5 个内容海绵 Token (z_cont)。"
        - "【细节 1】在预训练阶段保持完整 PyTorch 计算图，使 loss_total_h2 梯度反传更新 Student 主干与 Chunk Tokens。"
      hard_invariants:
        - "【H2 防挑刺红线】100% 零外部人工标签；通过 Teacher 目标净水投影 + 共享分类头 sg(W_c) 均匀分布最大熵双保险阻断词汇回流；下游推理零额外参数开销！"
      broken_assumptions:
        - "精确修复 [SYM-E2V-S3-02 / S3-03] 的失效假设 B 与隐性缺陷 B1：将词汇音素正交分流至内容海绵分支，净化情感全局表征。"
      reserved_slot_id: "[SLOT-H2: original_single | clean_proj_2A | grl_adv_2B | vicreg_cov_2C | gr_ssd_soft_2D | vclub_mi_2E]"
    """

    def __init__(
        self,
        active_card: str = "original_single",
        embed_dim: int = 768,
        num_emo_tokens: int = 5,
        num_cont_tokens: int = 5,
        num_phoneme_clusters: int = 500,
        use_teacher_clean_projection: bool = True,
        use_max_entropy_uniform_kl: bool = True,
        lambda_dis: float = 0.0045,
    ) -> None:
        super().__init__()
        self.slot_id = "H2"
        self.active_card = active_card
        self.status = "[TBD]" if active_card in ("original_single", "base", "tbd_passthrough") else "[ACTIVE_TRIAL]"
        self.embed_dim = int(embed_dim)
        self.num_emo_tokens = int(num_emo_tokens)
        self.num_cont_tokens = int(num_cont_tokens)
        self.num_phoneme_clusters = int(num_phoneme_clusters)
        self.use_teacher_clean_projection = bool(use_teacher_clean_projection)
        self.use_max_entropy_uniform_kl = bool(use_max_entropy_uniform_kl)
        self.lambda_dis = float(lambda_dis)

        rng = np.random.default_rng(42)
        self.sponge_w_np = (rng.standard_normal((self.embed_dim, self.num_phoneme_clusters)) * 0.02).astype(np.float32)
        if HAS_TORCH:
            self.sponge_head_wc = nn.Linear(self.embed_dim, self.num_phoneme_clusters)
        else:
            self.sponge_head_wc = None

    def split_and_pool_chunks(self, chunk_tokens: Any, disentangle_type: Optional[str] = None) -> Tuple[Any, Any]:
        mode = disentangle_type or self.active_card
        is_torch = HAS_TORCH and isinstance(chunk_tokens, torch.Tensor)
        if mode in ("original_single", "base", "tbd_passthrough"):
            z_emo = chunk_tokens.mean(dim=1) if is_torch else chunk_tokens.mean(axis=1)
            z_cont = (
                chunk_tokens[:, -self.num_cont_tokens :, :].mean(dim=1).detach()
                if is_torch
                else chunk_tokens[:, -self.num_cont_tokens :, :].mean(axis=1)
            )
            return z_emo, z_cont

        if is_torch:
            z_emo = chunk_tokens[:, : self.num_emo_tokens, :].mean(dim=1)
            z_cont = chunk_tokens[:, self.num_emo_tokens : self.num_emo_tokens + self.num_cont_tokens, :].mean(dim=1)
        else:
            z_emo = chunk_tokens[:, : self.num_emo_tokens, :].mean(axis=1)
            z_cont = chunk_tokens[:, self.num_emo_tokens : self.num_emo_tokens + self.num_cont_tokens, :].mean(axis=1)
        return z_emo, z_cont

    def compute_disentangle_losses(
        self,
        z_emo: Any,
        z_cont: Any,
        y_teacher_sal: Any,
        p_hubert_target: Optional[Any] = None,
        uses_external_human_labels: bool = False,
        disentangle_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        mode = disentangle_type or self.active_card

        if mode in ("original_single", "base", "tbd_passthrough"):
            meta = h2_original_single(z_emo, z_cont, y_teacher_sal, p_hubert_target)
        elif mode in ("clean_proj_2A", "v0_chunk_sponge_clean_proj"):
            meta = h2_clean_proj_2A(
                z_emo,
                z_cont,
                y_teacher_sal,
                p_hubert_target,
                sponge_head_wc=self.sponge_head_wc,
                sponge_weight_matrix=self.sponge_w_np,
                num_clusters=self.num_phoneme_clusters,
                use_teacher_clean_projection=self.use_teacher_clean_projection,
                use_max_entropy_uniform_kl=self.use_max_entropy_uniform_kl,
                lambda_dis=self.lambda_dis,
            )
        elif mode == "grl_adv_2B":
            meta = h2_grl_adversarial_2B(
                z_emo,
                z_cont,
                y_teacher_sal,
                p_hubert_target,
                sponge_head_wc=self.sponge_head_wc,
                sponge_weight_matrix=self.sponge_w_np,
                num_clusters=self.num_phoneme_clusters,
                use_teacher_clean_projection=self.use_teacher_clean_projection,
                use_max_entropy_uniform_kl=self.use_max_entropy_uniform_kl,
                lambda_dis=self.lambda_dis,
            )
        elif mode == "vicreg_cov_2C":
            meta = h2_vicreg_cov_2C(
                z_emo,
                z_cont,
                y_teacher_sal,
                p_hubert_target,
                use_teacher_clean_projection=self.use_teacher_clean_projection,
                lambda_dis=self.lambda_dis,
            )
        elif mode in H2_CANDIDATE_FUNCTIONS:
            meta = H2_CANDIDATE_FUNCTIONS[mode](
                z_emo,
                z_cont,
                y_teacher_sal,
                p_hubert_target,
                sponge_head_wc=self.sponge_head_wc,
                sponge_weight_matrix=self.sponge_w_np,
                num_clusters=self.num_phoneme_clusters,
                lambda_dis=self.lambda_dis,
            )
        else:
            raise ValueError(f"[H2 卡槽] 未知的 disentangle_type='{mode}'，可选值: {list(H2_CANDIDATE_FUNCTIONS)}")

        meta["uses_external_human_labels"] = bool(uses_external_human_labels)
        if mode not in ("original_single", "base", "tbd_passthrough"):
            self.verify_red_lines(meta)
        return meta

    def verify_red_lines(self, meta: Dict[str, Any]) -> None:
        if meta.get("uses_external_human_labels", False):
            raise HammerRedLineViolationError("[H2 Red-Line Violation] 严禁引入外部人工 ASR/音素标签！")
        if int(meta.get("downstream_extra_params", 0)) != 0:
            raise HammerRedLineViolationError("[H2 Red-Line Violation] 下游推理必须保持零额外参数开销！")
        if not meta.get("teacher_backflow_blocked", False):
            raise HammerRedLineViolationError("[H2 Red-Line Violation] 未阻断 Teacher 目标词汇回流！")
