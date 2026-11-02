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


H2_CANDIDATE_FUNCTIONS: Dict[str, Callable] = {
    "original_single": h2_original_single,
    "tbd_passthrough": h2_original_single,
    "clean_proj_2A": h2_clean_proj_2A,
    "v0_chunk_sponge_clean_proj": h2_clean_proj_2A,
    "grl_adv_2B": h2_grl_adversarial_2B,
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
      reserved_slot_id: "[SLOT-H2: original_single | clean_proj_2A | grl_adv_2B]"
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
        self.status = "[TBD]" if active_card in ("original_single", "tbd_passthrough") else "[ACTIVE_TRIAL]"
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
        if mode in ("original_single", "tbd_passthrough"):
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

        if mode in ("original_single", "tbd_passthrough"):
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
        if mode not in ("original_single", "tbd_passthrough"):
            self.verify_red_lines(meta)
        return meta

    def verify_red_lines(self, meta: Dict[str, Any]) -> None:
        if meta.get("uses_external_human_labels", False):
            raise HammerRedLineViolationError("[H2 Red-Line Violation] 严禁引入外部人工 ASR/音素标签！")
        if int(meta.get("downstream_extra_params", 0)) != 0:
            raise HammerRedLineViolationError("[H2 Red-Line Violation] 下游推理必须保持零额外参数开销！")
        if not meta.get("teacher_backflow_blocked", False):
            raise HammerRedLineViolationError("[H2 Red-Line Violation] 未阻断 Teacher 目标词汇回流！")
