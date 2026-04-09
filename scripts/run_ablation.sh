#!/usr/bin/env bash
# Run ablation experiments for top_k, entropy_coeff, index_refresh_interval, relevance_model.
# Each run gets its own checkpoint_dir and log_dir so TensorBoard curves stay separate.
#
# Usage:
#   bash scripts/run_ablation.sh              # run all groups
#   bash scripts/run_ablation.sh top_k        # run only top_k group
#   bash scripts/run_ablation.sh entropy      # run only entropy group
#   bash scripts/run_ablation.sh refresh      # run only refresh group
#   bash scripts/run_ablation.sh relevance    # run only relevance_model group
#
# Logs (stdout + stderr) are saved under outputs/ablation/<run_name>/train.log

set -euo pipefail

GROUP="${1:-all}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

run_experiment() {
    local name="$1"
    shift  # remaining args are key=value overrides

    local ckpt_dir="outputs/ablation/${name}"
    local log_dir="outputs/logs/ablation/${name}"
    local log_file="${ckpt_dir}/train.log"

    mkdir -p "$ckpt_dir"

    echo ""
    echo "=========================================="
    echo " RUN: ${name}"
    echo " overrides: $*"
    echo " checkpoint → ${ckpt_dir}"
    echo " tensorboard → ${log_dir}"
    echo " log → ${log_file}"
    echo "=========================================="

    python scripts/train.py \
        output.checkpoint_dir="${ckpt_dir}" \
        logging.log_dir="${log_dir}" \
        "$@" \
        2>&1 | tee "${log_file}"

    echo "[DONE] ${name}"
}

# ---------------------------------------------------------------------------
# Group: top_k  (default=3, skip 3)
# ---------------------------------------------------------------------------
run_top_k() {
    echo ""
    echo "########## Ablation: top_k ##########"
    run_experiment "top_k_2" retriever.top_k=2
    run_experiment "top_k_5" retriever.top_k=5
}

# ---------------------------------------------------------------------------
# Group: entropy_coeff  (default=0.01, skip 0.01)
# ---------------------------------------------------------------------------
run_entropy() {
    echo ""
    echo "########## Ablation: entropy_coeff ##########"
    run_experiment "entropy_0"    rl.entropy_coeff=0.0
    run_experiment "entropy_0001" rl.entropy_coeff=0.001
    run_experiment "entropy_01"   rl.entropy_coeff=0.1
}

# ---------------------------------------------------------------------------
# Group: index_refresh_interval  (default=500, skip 500)
# ---------------------------------------------------------------------------
run_refresh() {
    echo ""
    echo "########## Ablation: index_refresh_interval ##########"
    run_experiment "refresh_50"    rl.index_refresh_interval=50
    run_experiment "refresh_200"   rl.index_refresh_interval=200
    run_experiment "refresh_never" rl.index_refresh_interval=99999
}

# ---------------------------------------------------------------------------
# Group: relevance_model  (default=ollama/qwen2.5-coder:14b, skip 14b)
# ---------------------------------------------------------------------------
run_relevance() {
    echo ""
    echo "########## Ablation: relevance_model ##########"
    run_experiment "relevance_3b" "reward.relevance_model=ollama/qwen2.5-coder:3b"
    run_experiment "relevance_7b" "reward.relevance_model=ollama/qwen2.5-coder:7b"
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
case "$GROUP" in
    top_k)     run_top_k ;;
    entropy)   run_entropy ;;
    refresh)   run_refresh ;;
    relevance) run_relevance ;;
    all)
        run_top_k
        run_entropy
        run_refresh
        run_relevance
        ;;
    *)
        echo "Unknown group: $GROUP"
        echo "Usage: $0 [top_k|entropy|refresh|relevance|all]"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo " All runs complete."
echo " View results:"
echo "   tensorboard --logdir outputs/logs/ablation"
echo "=========================================="
