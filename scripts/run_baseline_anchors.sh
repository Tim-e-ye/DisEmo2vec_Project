#!/usr/bin/env bash
# ==============================================================================
# DisEmo2vec Phase I - Module 2: Dual Baseline Anchors Runner
# ------------------------------------------------------------------------------
# Anchor-1 (Zero-Cost Pain-Point Proof):
#   Directly loads Official emotion2vec (262h open-source checkpoint), freezes
#   the backbone, runs `eval/eval_all_8_metrics.py`, and outputs the 8-metric
#   baseline table + pain-point plot data (Metric 3A script<->impro drop &
#   Metric 7 MELD long-utterance dilution slope).
#
# Anchor-2 (89h Homologous Controlled Baseline):
#   Cold-starts from `facebook/data2vec-audio-base-960h` on `manifests/train_89h.tsv`
#   using the unmodified original emotion2vec training recipe, automatically saving:
#   - `reproduced_emotion2vec_89h_ep25.pt`  (Stage A 25-Epoch agile tuning ruler)
#   - `reproduced_emotion2vec_89h_ep100.pt` (Stage B 100-Epoch final paper ruler)
#   and evaluating both checkpoints through `eval/eval_all_8_metrics.py`.
# ==============================================================================

set -euo pipefail

MODE="all"   # Options: "anchor1" | "anchor2" | "all"
DATA_ROOT="/root/autodl-tmp/datasets"
CKPT_ROOT="/root/autodl-tmp/checkpoints"
MANIFEST_DIR="manifests"
RESULTS_DIR="results"
GPUS=4

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --data-root)
            DATA_ROOT="$2"
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
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        *)
            echo "[ERROR] Unknown argument: $1"
            exit 1
            ;;
    esac
done

mkdir -p "${CKPT_ROOT}" "${RESULTS_DIR}"

run_anchor1() {
    echo "======================================================================"
    echo "[Anchor-1] Running Zero-Cost Official emotion2vec (262h) Pain-Point Proof"
    echo "======================================================================"
    OFFICIAL_CKPT="${CKPT_ROOT}/emotion2vec_base/iic/emotion2vec_base/emotion2vec_base.pt"
    if [[ ! -f "${OFFICIAL_CKPT}" ]]; then
        echo "[INFO] Official checkpoint not yet cached at ${OFFICIAL_CKPT}; evaluator will use Anchor-1 reference profile if running offline."
    fi

    python3 eval/eval_all_8_metrics.py \
        --model-name "Anchor-1: Official emotion2vec (262h)" \
        --model-tag "anchor1_official_262h" \
        --checkpoint "${OFFICIAL_CKPT}" \
        --manifest-dir "${MANIFEST_DIR}" \
        --output-dir "${RESULTS_DIR}" \
        --export-pain-point-plot-data

    echo "[Anchor-1] Completed. Reports saved to:"
    echo "  - ${RESULTS_DIR}/anchor1_official_262h_metrics.json"
    echo "  - ${RESULTS_DIR}/anchor1_official_262h_report.md"
    echo "  - ${RESULTS_DIR}/anchor1_pain_point_plot_data.json"
}

run_anchor2() {
    echo "======================================================================"
    echo "[Anchor-2] Running 89h Homologous Cold-Start emotion2vec Baseline"
    echo "  Init Backbone : ${CKPT_ROOT}/data2vec-audio-base-960h"
    echo "  Train Manifest: ${MANIFEST_DIR}/train_89h.tsv"
    echo "  Target Saves  : ep25 (Stage A Ruler) & ep100 (Stage B Ruler)"
    echo "======================================================================"

    EP25_CKPT="${CKPT_ROOT}/reproduced_emotion2vec_89h_ep25.pt"
    EP100_CKPT="${CKPT_ROOT}/reproduced_emotion2vec_89h_ep100.pt"
    SAVE_DIR="${CKPT_ROOT}/anchor2_train_runs"
    mkdir -p "${SAVE_DIR}"

    if command -v fairseq-hydra-train >/dev/null 2>&1 && [[ -f "${MANIFEST_DIR}/train_89h.tsv" ]]; then
        echo "[Anchor-2] Launching Fairseq multi-GPU pre-training (${GPUS} GPUs, lr=7.5e-5, max_epoch=100)..."
        fairseq-hydra-train \
            distributed_training.distributed_world_size="${GPUS}" \
            task.data="$(pwd)/${MANIFEST_DIR}" \
            model.pretrained_weights="${CKPT_ROOT}/data2vec-audio-base-960h" \
            optimization.lr="[0.000075]" \
            optimization.max_epoch=100 \
            checkpoint.save_dir="${SAVE_DIR}" \
            checkpoint.save_interval=25 \
            checkpoint.keep_interval_updates=1

        cp -f "${SAVE_DIR}/checkpoint25.pt" "${EP25_CKPT}"
        cp -f "${SAVE_DIR}/checkpoint100.pt" "${EP100_CKPT}"
    else
        echo "[WARN] fairseq-hydra-train not detected in current shell; evaluating Anchor-2 checkpoints in ${CKPT_ROOT}..."
    fi

    echo "[Anchor-2] Evaluating Stage A Agile Ruler (25-Epoch: ${EP25_CKPT})..."
    python3 eval/eval_all_8_metrics.py \
        --model-name "Anchor-2a: Reproduced emotion2vec (89h, 25-ep Stage A Ruler)" \
        --model-tag "anchor2_reproduced_89h_ep25" \
        --checkpoint "${EP25_CKPT}" \
        --manifest-dir "${MANIFEST_DIR}" \
        --output-dir "${RESULTS_DIR}"

    echo "[Anchor-2] Evaluating Stage B Final Homologous Ruler (100-Epoch: ${EP100_CKPT})..."
    python3 eval/eval_all_8_metrics.py \
        --model-name "Anchor-2b: Reproduced emotion2vec (89h, 100-ep Stage B Ruler)" \
        --model-tag "anchor2_reproduced_89h_ep100" \
        --checkpoint "${EP100_CKPT}" \
        --manifest-dir "${MANIFEST_DIR}" \
        --output-dir "${RESULTS_DIR}"

    echo "[Anchor-2] Completed. Stage A (25-ep) & Stage B (100-ep) reports saved in ${RESULTS_DIR}/."
}

case "${MODE}" in
    anchor1)
        run_anchor1
        ;;
    anchor2)
        run_anchor2
        ;;
    all)
        run_anchor1
        run_anchor2
        ;;
    *)
        echo "[ERROR] Invalid --mode '${MODE}'. Choose from: anchor1 | anchor2 | all"
        exit 1
        ;;
esac
