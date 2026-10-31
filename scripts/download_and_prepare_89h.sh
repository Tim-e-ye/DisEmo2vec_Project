#!/usr/bin/env bash
# ==============================================================================
# DisEmo2vec Phase I - Module 1: 89h Public Homologous Dataset & Weight Downloader
# Target Datasets: IEMOCAP (12h) + MELD (12h) + CMU-MOSEI (65h) = 89h Unlabeled/Eval
# Domestic High-Speed Mirrors: AutoDL network_turbo + HF-Mirror + ModelScope
# ==============================================================================

set -euo pipefail

DATA_ROOT="/root/autodl-tmp/datasets"
CKPT_ROOT="/root/autodl-tmp/checkpoints"
MANIFEST_DIR="manifests"
NUM_WORKERS=8

while [[ $# -gt 0 ]]; do
    case "$1" in
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
        --workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        *)
            echo "[ERROR] Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo "======================================================================"
echo "[DisEmo2vec] Phase I Data & Checkpoint Preparation Pipeline"
echo "  DATA_ROOT     : ${DATA_ROOT}"
echo "  CKPT_ROOT     : ${CKPT_ROOT}"
echo "  MANIFEST_DIR  : ${MANIFEST_DIR}"
echo "======================================================================"

# 1. Enable AutoDL Academic Network Acceleration & Domestic HF Mirror
if [[ -f "/etc/network_turbo" ]]; then
    echo "[INFO] Enabling AutoDL academic acceleration (/etc/network_turbo)..."
    # shellcheck disable=SC1091
    source /etc/network_turbo
else
    echo "[INFO] /etc/network_turbo not found (non-AutoDL host); skipping turbo."
fi

export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_ENABLE_HF_TRANSFER=1
export MODELSCOPE_CACHE="${CKPT_ROOT}/modelscope_cache"

mkdir -p "${DATA_ROOT}/IEMOCAP" "${DATA_ROOT}/MELD" "${DATA_ROOT}/CMU_MOSEI"
mkdir -p "${CKPT_ROOT}" "${MANIFEST_DIR}"

# 2. Check and install essential CLI dependencies
python3 -c "import huggingface_hub, modelscope, soundfile, librosa, sklearn" 2>/dev/null || {
    echo "[INFO] Installing required Python packages via domestic Tsinghua PyPI mirror..."
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
        "huggingface_hub[cli]" modelscope datasets soundfile librosa scikit-learn vaderSentiment scipy tqdm pyyaml
}

# 3. Download Pre-trained Base Checkpoints (data2vec-audio-base-960h & Official emotion2vec_base)
echo "[STEP 1/4] Pulling initialization backbone (facebook/data2vec-audio-base-960h) & Official emotion2vec_base..."
python3 - <<EOF
import os
from huggingface_hub import snapshot_download
from modelscope.hub.snapshot_download import snapshot_download as ms_download

ckpt_root = "${CKPT_ROOT}"

print(">> Downloading facebook/data2vec-audio-base-960h from HF-Mirror...")
snapshot_download(
    repo_id="facebook/data2vec-audio-base-960h",
    local_dir=os.path.join(ckpt_root, "data2vec-audio-base-960h"),
    local_dir_use_symlinks=False,
    resume_download=True,
)

print(">> Downloading roberta-base (for MELD Table-6 homologous text-neutral hard slice filtering)...")
snapshot_download(
    repo_id="FacebookAI/roberta-base",
    local_dir=os.path.join(ckpt_root, "roberta-base"),
    local_dir_use_symlinks=False,
    resume_download=True,
)

print(">> Downloading Official emotion2vec_base (262h Anchor-1) from ModelScope...")
ms_download(
    model_id="iic/emotion2vec_base",
    cache_dir=os.path.join(ckpt_root, "emotion2vec_base"),
)
print(">> Base checkpoints successfully prepared in:", ckpt_root)
EOF

# 4. Download & Extract 89h Public Audio Datasets (IEMOCAP 12h + MELD 12h + CMU-MOSEI 65h)
echo "[STEP 2/4] Pulling 89h Public Audio Datasets from HF-Mirror / ModelScope..."
python3 - <<EOF
import os
from huggingface_hub import snapshot_download

data_root = "${DATA_ROOT}"

datasets_to_pull = [
    ("IEMOCAP", "AbstractTTS/IEMOCAP", os.path.join(data_root, "IEMOCAP")),
    ("MELD", "declare-lab/MELD", os.path.join(data_root, "MELD")),
    ("CMU_MOSEI", "CMU-MultimodalSDK/CMU-MOSEI", os.path.join(data_root, "CMU_MOSEI")),
]

for name, repo_id, target_dir in datasets_to_pull:
    print(f">> Checking/Downloading {name} ({repo_id}) -> {target_dir} ...")
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
    except Exception as exc:
        print(f"[WARN] Snapshot download notice for {name}: {exc}. Ensure raw or mirror files are placed in {target_dir}.")
EOF

# 5. Standardize all audio files to 16000Hz, 16-bit, mono .wav
echo "[STEP 3/4] Normalizing all audio files to 16kHz, 16-bit, mono PCM .wav..."
if command -v ffmpeg >/dev/null 2>&1; then
    find "${DATA_ROOT}" -type f \( -name "*.mp4" -o -name "*.flac" -o -name "*.mp3" \) | while read -r media_file; do
        wav_out="${media_file%.*}.wav"
        if [[ ! -f "${wav_out}" ]]; then
            ffmpeg -nostdin -y -i "${media_file}" -ar 16000 -ac 1 -sample_fmt s16 "${wav_out}" -loglevel error
        fi
    done
fi

# 6. Trigger Manifest & Offline K-Means(K=500) Probe Label Generation
echo "[STEP 4/4] Building Fairseq train_89h.tsv, speaker-independent 5-fold slices, MELD hard slices, and K=500 phoneme labels..."
python3 scripts/build_manifests.py \
    --data-root "${DATA_ROOT}" \
    --ckpt-root "${CKPT_ROOT}" \
    --manifest-dir "${MANIFEST_DIR}" \
    --workers "${NUM_WORKERS}"

echo "======================================================================"
echo "[SUCCESS] 89h Data Base & Slice Metadata Ready in ${MANIFEST_DIR}/"
echo "======================================================================"
