#!/usr/bin/env bash
# ==============================================================================
# DisEmo2vec 第三步：单变量小步快跑试锤流水线 (`scripts/run_single_hammer_trial.sh`)
# ==============================================================================
# 完整串联 3 步真实闭环（绝不静默跳过训练或用假数据冒充真权重）：
#   Step 1: 上机前 3 分钟 100 步冒烟总闸 (`scripts/smoke_test_hammer.py`)
#           检验 4 道硬性门禁（Padding 零泄漏 <1e-5、梯度模长自动配平器、防坍缩与吞吐）
#   Step 2: 真实 25-Epoch / 100-Epoch 预训练引擎 (`scripts/train_wind_tunnel.py`)
#           串联 S1->S2->S3->S4 与 H1/H2/H3 卡槽，生成真实权重 `${TRIAL_CKPT}`
#   Step 3: 8 指标防挑刺钉子检测器 (`eval/eval_all_8_metrics.py --strict-real`)
#           开启 `--strict-real` 真权重校验守卫，输出 JSON + Markdown 成绩单
#
# 用法示例：
#   1. 单测卡槽 H3 候选锤头 3A (salience_3A)：
#      bash scripts/run_single_hammer_trial.sh --pooling_type salience_3A
#   2. 若 3A 没涨分，直接切 3B (attn_rollout_3B)：
#      bash scripts/run_single_hammer_trial.sh --pooling_type attn_rollout_3B
#   3. H3 锁定 salience_3A 后，叠加卡槽 H2 候选锤头 2A (clean_proj_2A)：
#      bash scripts/run_single_hammer_trial.sh --pooling_type salience_3A --disentangle_type clean_proj_2A
#   4. 再叠加卡槽 H1 候选锤头 1A (vtlp_1A)：
#      bash scripts/run_single_hammer_trial.sh --pooling_type salience_3A --disentangle_type clean_proj_2A --view_type vtlp_1A
# ==============================================================================

set -euo pipefail

VIEW_TYPE="original_none"           # 卡槽 H1 开关: original_none | vtlp_1A | channel_drop_1B | offline_wav_1C
DISENTANGLE_TYPE="original_single"  # 卡槽 H2 开关: original_single | clean_proj_2A | grl_adv_2B | vicreg_cov_2C
POOLING_TYPE="original_mean"        # 卡槽 H3 开关: original_mean | salience_3A | attn_rollout_3B | autopool_3C
EPOCHS=25
BATCH_SIZE=8
LR="5e-4"
LAMBDA_DIS="auto"
FORCE_TRAIN=0
CKPT_ROOT="/root/autodl-tmp/checkpoints"
MANIFEST_DIR="manifests"
RESULTS_DIR="results"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --view_type)
            VIEW_TYPE="$2"
            shift 2
            ;;
        --disentangle_type)
            DISENTANGLE_TYPE="$2"
            shift 2
            ;;
        --pooling_type)
            POOLING_TYPE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --lambda_dis|--lambda-dis)
            LAMBDA_DIS="$2"
            shift 2
            ;;
        --ckpt-root)
            CKPT_ROOT="$2"
            shift 2
            ;;
        --manifest-dir)
            MANIFEST_DIR="$2"
            shift 2
            ;;
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --force-train)
            FORCE_TRAIN=1
            shift 1
            ;;
        *)
            echo "[ERROR] Unknown switch: $1"
            exit 1
            ;;
    esac
done

mkdir -p "${CKPT_ROOT}" "${RESULTS_DIR}"
TRIAL_TAG="trial_H1-${VIEW_TYPE}_H2-${DISENTANGLE_TYPE}_H3-${POOLING_TYPE}_ep${EPOCHS}"
TRIAL_CKPT="${CKPT_ROOT}/${TRIAL_TAG}.pt"

echo "======================================================================"
echo "[Step 3 单变量试锤流水线] 启动 ${EPOCHS}-Epoch 敏捷验靶 (3步真实闭环)"
echo "  🔨 卡槽 H1 (--view_type)        : ${VIEW_TYPE}"
echo "  🔨 卡槽 H2 (--disentangle_type) : ${DISENTANGLE_TYPE}"
echo "  🔨 卡槽 H3 (--pooling_type)     : ${POOLING_TYPE}"
echo "  🏷️ 实验标识 (TRIAL_TAG)         : ${TRIAL_TAG}"
echo "  💾 权重路径 (TRIAL_CKPT)        : ${TRIAL_CKPT}"
echo "======================================================================"

# ------------------------------------------------------------------------------
# Step 1: 上机前 100 步冒烟总闸（若 4 道门禁任一未过则立即中断报警）
# ------------------------------------------------------------------------------
echo "[Step 1/3] 运行 scripts/smoke_test_hammer.py 四道门禁预检..."
python3 scripts/smoke_test_hammer.py \
    --view_type "${VIEW_TYPE}" \
    --disentangle_type "${DISENTANGLE_TYPE}" \
    --pooling_type "${POOLING_TYPE}" \
    --steps 100

# ------------------------------------------------------------------------------
# Step 2: 真实预训练引擎（若权重尚未生成或指定 --force-train，则立即训练落盘）
# ------------------------------------------------------------------------------
if [[ ! -f "${TRIAL_CKPT}" || "${FORCE_TRAIN}" -eq 1 ]]; then
    echo "[Step 2/3] 启动真实预训练 scripts/train_wind_tunnel.py (${EPOCHS} Epochs, lambda_dis=${LAMBDA_DIS})..."
    python3 scripts/train_wind_tunnel.py \
        --manifest-dir "${MANIFEST_DIR}" \
        --view_type "${VIEW_TYPE}" \
        --disentangle_type "${DISENTANGLE_TYPE}" \
        --pooling_type "${POOLING_TYPE}" \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --lr "${LR}" \
        --lambda_dis "${LAMBDA_DIS}" \
        --save-ckpt "${TRIAL_CKPT}"
else
    echo "[Step 2/3] 检测到已存在真实训练权重 ${TRIAL_CKPT}，直接复用（如需重训请加 --force-train）。"
fi

# ------------------------------------------------------------------------------
# Step 3: 送入钉子检测器（开启 --strict-real 真权重校验守卫）输出 8 项核心指标
# ------------------------------------------------------------------------------
echo "[Step 3/3] 启动 eval/eval_all_8_metrics.py --strict-real 冻结主干评测..."
python3 eval/eval_all_8_metrics.py \
    --model-name "Hammer Trial (${TRIAL_TAG})" \
    --model-tag "${TRIAL_TAG}" \
    --checkpoint "${TRIAL_CKPT}" \
    --strict-real \
    --manifest-dir "${MANIFEST_DIR}" \
    --output-dir "${RESULTS_DIR}"

echo "======================================================================"
echo "[DONE] 3 步真实试锤闭环完成！请查看电子秤成绩单："
echo "  - 训练权重文件  : ${TRIAL_CKPT}"
echo "  - Markdown 成绩单: ${RESULTS_DIR}/${TRIAL_TAG}_report.md"
echo "  - JSON 详细指标  : ${RESULTS_DIR}/${TRIAL_TAG}_metrics.json"
echo "======================================================================"
