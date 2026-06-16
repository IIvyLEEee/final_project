#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON="${PYTHON:-/home/liyixuan23/miniforge3/envs/final_project/bin/python}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data/30h_data}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$PROJECT_DIR/data/spm1000/spm_unigram1000.model}"
PRETRAIN_PATH="${PRETRAIN_PATH:-$PROJECT_DIR/ckpt/pretrained_model.pth}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_DIR/exp/5_1_loss_sweep_single}"
MAX_TOKENS="${MAX_TOKENS:-2000}"

mkdir -p "$PROJECT_DIR/logs" "$RUN_ROOT"

launch_one() {
  local session="$1"
  local gpu="$2"
  local name="$3"
  local focal_gamma="$4"
  local confidence_penalty="$5"
  local port="$6"
  local run_dir="$RUN_ROOT/$name"
  local log_file="$PROJECT_DIR/logs/$session.log"

  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session already exists, skipping: $session"
    return 0
  fi

  tmux new-session -d -s "$session" \
    "cd '$PROJECT_DIR' && \
     CUDA_VISIBLE_DEVICES='$gpu' '$PYTHON' -u main.py \
       --config-dir '$PROJECT_DIR/configs/' \
       --config-name video2text.yaml \
       task.data='$DATA_DIR' \
       task.label_dir='$DATA_DIR' \
       task.tokenizer_bpe_model='$TOKENIZER_PATH' \
       model.pretrained_path='$PRETRAIN_PATH' \
       criterion.focal_gamma='$focal_gamma' \
       criterion.confidence_penalty='$confidence_penalty' \
       dataset.max_tokens='$MAX_TOKENS' \
       distributed_training.distributed_port='$port' \
       hydra.run.dir='$run_dir' \
       common.user_dir='$PROJECT_DIR' \
       > '$log_file' 2>&1"
  echo "launched $session on GPU $gpu: $name"
}

launch_one "f51_focal05" 0 "focal_gamma_0_5" "0.5" "0.0" 29710
launch_one "f51_focal10" 1 "focal_gamma_1_0" "1.0" "0.0" 29711
launch_one "f51_focal20" 2 "focal_gamma_2_0" "2.0" "0.0" 29712
launch_one "f51_cp0001" 3 "confidence_penalty_0_001" "0.0" "0.001" 29713
launch_one "f51_cp0005" 0 "confidence_penalty_0_005" "0.0" "0.005" 29714
launch_one "f51_cp001" 1 "confidence_penalty_0_01" "0.0" "0.01" 29715
launch_one "f51_cp005" 2 "confidence_penalty_0_05" "0.0" "0.05" 29716
launch_one "f51_g05_cp0005" 3 "gamma_0_5_cp_0_005" "0.5" "0.005" 29717
launch_one "f51_g05_cp001" 0 "gamma_0_5_cp_0_01" "0.5" "0.01" 29718
launch_one "f51_g10_cp0005" 1 "gamma_1_0_cp_0_005" "1.0" "0.005" 29719
launch_one "f51_g10_cp001" 2 "gamma_1_0_cp_0_01" "1.0" "0.01" 29720
launch_one "f51_g20_cp0005" 3 "gamma_2_0_cp_0_005" "2.0" "0.005" 29721
launch_one "f51_g20_cp001" 0 "gamma_2_0_cp_0_01" "2.0" "0.01" 29722
