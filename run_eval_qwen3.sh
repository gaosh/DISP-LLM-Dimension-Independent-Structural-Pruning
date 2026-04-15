#!/usr/bin/env bash
set -euo pipefail

# Optional:
# export CUDA_VISIBLE_DEVICES=0

ROOT_DIR="/home/sgao1/DISP-LLM-Dimension-Independent-Structural-Pruning"
PRUNE_SCRIPT="${ROOT_DIR}/prune_model_qwen3.py"

HN_ROOT="/orange/sgao1/sgao1/saved_hns/disp_llm"
MODEL_ROOT="/orange/sgao1/sgao1/saved_models/disp_llm"
LOG_ROOT="${ROOT_DIR}/eval_logs"

mkdir -p "${LOG_ROOT}"
mkdir -p "${MODEL_ROOT}"

TASKS="hellaswag,arc_easy,arc_challenge,piqa,winogrande"
BASE_PORT=12323
RUN_PPL=true

# Each row:
# EXP_NAME | PVAL | HF_MODEL | FAMILY_DIR | SAVE_NAME | EVAL_BATCH_SIZE
MODELS=(
    "qwen3_8b_alpaca|0.50|Qwen/Qwen3-8B|${MODEL_ROOT}/qwen3_8b_alpaca|qwen3_8b_alpaca_0.50|32"
    "qwen3_8b_alpaca|0.60|Qwen/Qwen3-8B|${MODEL_ROOT}/qwen3_8b_alpaca|qwen3_8b_alpaca_0.60|32"
    "qwen3_8b_alpaca|0.70|Qwen/Qwen3-8B|${MODEL_ROOT}/qwen3_8b_alpaca|qwen3_8b_alpaca_0.70|32"

    # "qwen3_14b_alpaca|0.50|Qwen/Qwen3-14B|${MODEL_ROOT}/qwen3_14b_alpaca|qwen3_14b_alpaca_0.50|16"
    # "qwen3_14b_alpaca|0.60|Qwen/Qwen3-14B|${MODEL_ROOT}/qwen3_14b_alpaca|qwen3_14b_alpaca_0.60|16"

    # "qwen3_32b_alpaca|0.50|Qwen/Qwen3-32B|${MODEL_ROOT}/qwen3_32b_alpaca|qwen3_32b_alpaca_0.50|4"
)

idx=0
for item in "${MODELS[@]}"; do
    IFS='|' read -r EXP_NAME PVAL HF_MODEL FAMILY_DIR SAVE_NAME EVAL_BATCH_SIZE <<< "${item}"

    HN_PATH="${HN_ROOT}/${EXP_NAME}/hn-ckpt-final-${PVAL}.pt"
    TMP_OUT_DIR="${FAMILY_DIR}/${SAVE_NAME}"
    LOG_FILE="${LOG_ROOT}/${SAVE_NAME}.txt"
    PORT=$((BASE_PORT + idx))

    mkdir -p "${FAMILY_DIR}"
    rm -rf "${TMP_OUT_DIR}"

    {
        echo "=================================================="
        echo "Processing ${SAVE_NAME}"
        echo "HF_MODEL: ${HF_MODEL}"
        echo "HN_PATH: ${HN_PATH}"
        echo "TMP_OUT_DIR: ${TMP_OUT_DIR}"
        echo "LOG_FILE: ${LOG_FILE}"
        echo "PORT: ${PORT}"
        echo "EVAL_BATCH_SIZE: ${EVAL_BATCH_SIZE}"
        echo "=================================================="
    } | tee "${LOG_FILE}"

    python "${PRUNE_SCRIPT}" \
        --hf_model "${HF_MODEL}" \
        --hn_path "${HN_PATH}" \
        --out_dir "${TMP_OUT_DIR}" \
        --p "${PVAL}" \
        --evaluate_ppl "${RUN_PPL}" 2>&1 | tee -a "${LOG_FILE}"

    accelerate launch --main_process_port "${PORT}" --num_processes 1 \
        -m lm_eval --model hf \
        --model_args pretrained="${TMP_OUT_DIR}/",dtype=bfloat16,trust_remote_code=true \
        --tasks "${TASKS}" \
        --device cuda:0 \
        --batch_size "${EVAL_BATCH_SIZE}" 2>&1 | tee -a "${LOG_FILE}"

    echo "" | tee -a "${LOG_FILE}"
    echo "Finished ${SAVE_NAME}" | tee -a "${LOG_FILE}"
    echo "" | tee -a "${LOG_FILE}"

    idx=$((idx + 1))
done