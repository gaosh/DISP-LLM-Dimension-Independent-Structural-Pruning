#!/usr/bin/env bash
# example:
# bash run_qwen3.sh 8b 0.5
# bash run_qwen3.sh "8b 14b" "0.4 0.5"
# bash run_qwen3.sh "8b 14b 32b" "0.3 0.4"

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TARGET_LIST_STR=${1:-"8b"}
P_LIST_STR=${2:-"0.5"}

read -r -a TARGET_LIST <<< "$TARGET_LIST_STR"
read -r -a P_LIST <<< "$P_LIST_STR"

run_exp () {
    local MODEL=$1
    local DATA=$2
    local OUTDIR=$3
    local LOGBASE=$4
    local P=$5

    local LOGFILE="${LOGBASE%.txt}_${P}.txt"

    echo "=================================================="
    echo "Running model: $MODEL"
    echo "Dataset: $DATA"
    echo "p: $P"
    echo "Output: $OUTDIR"
    echo "Log: $LOGFILE"
    echo "=================================================="

    torchrun --standalone --nproc_per_node=1 \
        --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29517 \
        train_hypernetwork.py \
        --hf_model "$MODEL" \
        --data_source "$DATA" \
        --use_fsdp False \
        --p "$P" \
        --lam 16.0 \
        --batch_size 1 \
        --use_bf16 True \
        --total_n_step 10000 \
        --hn_lr 1e-3 \
        --min_hn_lr 1e-3 \
        --use_sch False \
        --num_workers 2 \
        --out_dir "$OUTDIR" \
        --exp_name "PruneQwen3_p${P}" \
        > "$LOGFILE" 2>&1

    echo "Finished $MODEL $DATA p=$P"
    echo
}

run_target () {
    local TARGET=$1

    case "$TARGET" in
        8b)
            for P in "${P_LIST[@]}"; do
                # run_exp \
                # Qwen/Qwen3-8B \
                # wiki \
                # /orange/sgao1/sgao1/saved_hns/disp_llm/qwen3_8b_wiki \
                # qwen3_8b_wiki.txt \
                # "$P"

                run_exp \
                Qwen/Qwen3-8B \
                alpaca \
                /orange/sgao1/sgao1/saved_hns/disp_llm/qwen3_8b_alpaca \
                qwen3_8b_alpaca.txt \
                "$P"
            done
            ;;

        14b)
            for P in "${P_LIST[@]}"; do
                # run_exp \
                # Qwen/Qwen3-14B \
                # wiki \
                # /orange/sgao1/sgao1/saved_hns/disp_llm/qwen3_14b_wiki \
                # qwen3_14b_wiki.txt \
                # "$P"

                run_exp \
                Qwen/Qwen3-14B \
                alpaca \
                /orange/sgao1/sgao1/saved_hns/disp_llm/qwen3_14b_alpaca \
                qwen3_14b_alpaca.txt \
                "$P"
            done
            ;;

        32b)
            for P in "${P_LIST[@]}"; do
                # run_exp \
                # Qwen/Qwen3-32B \
                # wiki \
                # /orange/sgao1/sgao1/saved_hns/disp_llm/qwen3_32b_wiki \
                # qwen3_32b_wiki.txt \
                # "$P"

                run_exp \
                Qwen/Qwen3-32B \
                alpaca \
                /orange/sgao1/sgao1/saved_hns/disp_llm/qwen3_32b_alpaca \
                qwen3_32b_alpaca.txt \
                "$P"
            done
            ;;

        *)
            echo "Unsupported target: $TARGET"
            echo "Usage: $0 \"{8b|14b|32b} ...\" [p|\"p1 p2 p3\"]"
            exit 1
            ;;
    esac
}

for TARGET in "${TARGET_LIST[@]}"; do
    run_target "$TARGET"
done

echo "Qwen3 experiments finished."
echo "Models: ${TARGET_LIST[*]}"
echo "p values: ${P_LIST[*]}"