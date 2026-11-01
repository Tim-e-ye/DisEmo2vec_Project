#!/usr/bin/env bash
# ==============================================================================
# DisEmo2vec 第三步：单变量小步快跑试锤流水线 (`scripts/run_single_hammer_trial.sh`)
# ==============================================================================
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

VIEW_TYPE="original_none"           # 卡槽 H1 开关: original_none | vtlp_1A | channel_drop_1B
DISENTANGLE_TYPE="original_single"  # 卡槽 H2 开关: original_single | clean_proj_2A | grl_adv_2B
POOLING_TYPE="original_mean"        # 卡槽 H3 开关: original_mean | salience_3A | attn_rollout_3B
EPOCHS=25
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
        --ckpt-root)
            CKPT_ROOT="$2"
            shift 2
            ;;
        *)
            echo "[ERROR] Unknown switch: $1"
            exit 1
            ;;
    esac
done

TRIAL_TAG="trial_H1-${VIEW_TYPE}_H2-${DISENTANGLE_TYPE}_H3-${POOLING_TYPE}_ep${EPOCHS}"
TRIAL_CKPT="${CKPT_ROOT}/${TRIAL_TAG}.pt"

echo "======================================================================"
echo "[Step 3 单变量试锤流水线] 启动 25-Epoch 敏捷验靶"
echo "  🔨 卡槽 H1 (--view_type)        : ${VIEW_TYPE}"
echo "  🔨 卡槽 H2 (--disentangle_type) : ${DISENTANGLE_TYPE}"
echo "  🔨 卡槽 H3 (--pooling_type)     : ${POOLING_TYPE}"
echo "  🏷️ 实验标识 (TRIAL_TAG)         : ${TRIAL_TAG}"
echo "======================================================================"

# 1. 先执行卡槽防挑刺红线快速预检
python3 - <<EOF
from models.hammer_modules import build_hammer_suite_from_config
import numpy as np

suite = build_hammer_suite_from_config({
    "view_type": "${VIEW_TYPE}",
    "disentangle_type": "${DISENTANGLE_TYPE}",
    "pooling_type": "${POOLING_TYPE}",
})
suite.train()
dummy_x = np.random.randn(2, 40, 768).astype("float32")
dummy_e = np.random.randn(2, 40).astype("float32")
out = suite(dummy_x, dummy_x, frame_energy=dummy_e)
print("[PRE-CHECK PASSED] H1/H2/H3 开关组合通过防挑刺红线检查！")
EOF

# 2. 送入钉子检测器（eval/eval_all_8_metrics.py）一键称重输出 8 项核心指标
python3 eval/eval_all_8_metrics.py \
    --model-name "Hammer Trial (${TRIAL_TAG})" \
    --model-tag "${TRIAL_TAG}" \
    --checkpoint "${TRIAL_CKPT}" \
    --manifest-dir "${MANIFEST_DIR}" \
    --output-dir "${RESULTS_DIR}"

echo "======================================================================"
echo "[DONE] 试锤完成！请查看电子秤成绩单："
echo "  - Markdown 成绩单: ${RESULTS_DIR}/${TRIAL_TAG}_report.md"
echo "  - JSON 详细指标  : ${RESULTS_DIR}/${TRIAL_TAG}_metrics.json"
echo "======================================================================"
