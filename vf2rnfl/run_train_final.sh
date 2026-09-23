#!/bin/bash
set -euo pipefail

# Data locations (see config/env.example.sh)
: "${VF_DATA_ROOT:?set VF_DATA_ROOT first - see config/env.example.sh}"

# ==============================================================================
# MASTER TRAINING SCRIPT: VF SKIP CONNECTIONS - PARALLEL EDITION
# Goal: Test skip connection architecture across all denoising methods
# ==============================================================================

# ---------------- GPU SETTINGS ----------------

GPUS=(0 1 2 3 4 5 6 7)
MAX_PARALLEL=5
gpu_index=0
running=0

next_gpu () {
  local g="${GPUS[$gpu_index]}"
  gpu_index=$(( (gpu_index + 1) % ${#GPUS[@]} ))
  echo "$g"
}

launch () {  # launch <gpu> <logfile> <command...>
  local gpu="$1"
  local log="$2"
  shift 2
  echo "Launching on GPU $gpu → $log"
  CUDA_VISIBLE_DEVICES="$gpu" "$@" > "$log" 2>&1 &
}

throttle () {
  running=$((running + 1))
  if (( running >= MAX_PARALLEL )); then
    wait -n || true
    running=$((running - 1))
  fi
}

mkdir -p logs

# ---------------- Common Settings ----------------
AE_CKPT="checkpoints/masked30_ae_z32_autoencoder_best.pth"
Z_DIM=32
BATCH_SIZE=64
EPOCHS=100
SPLIT_RATIO=0.2

# --- UNIFIED SETTINGS WITH SKIP CONNECTIONS ---
LR=3e-4
PIXEL_LAMBDA=10.0
LATENT_LAMBDA=2.0
DROPOUT=0.05
FREEZE_STRATEGY="half"
MASK_RADIUS=30
SKIP_CHANNELS=24

echo "Starting Experiment: Skip Connection Architecture Comparison"
echo "   Skip Channels=$SKIP_CHANNELS, Freeze=$FREEZE_STRATEGY"
echo "   LR=$LR, Pixel=$PIXEL_LAMBDA, Latent=$LATENT_LAMBDA, Dropout=$DROPOUT"
echo "   GPUs=${GPUS[*]} | MAX_PARALLEL=$MAX_PARALLEL"
echo "---------------------------------------------------"

# Helper function to launch experiments
run_model () {
  local folder="$1"
  local run_name="$2"
  local log_file="$3"
  
  # Get next GPU
  local gpu="${GPUS[$gpu_index]}"
  gpu_index=$(( (gpu_index + 1) % ${#GPUS[@]} ))
  
  echo "Assigning GPU $gpu for $run_name"

  launch "$gpu" "$log_file" \
    python vf_to_rnfl_skip.py \
      --folder "$folder" \
      --autoencoder_ckpt "$AE_CKPT" \
      --run_name "$run_name" \
      --z_dim "$Z_DIM" \
      --lr "$LR" \
      --batch_size "$BATCH_SIZE" \
      --epochs "$EPOCHS" \
      --split_ratio "$SPLIT_RATIO" \
      --pixel_lambda "$PIXEL_LAMBDA" \
      --latent_lambda "$LATENT_LAMBDA" \
      --dropout "$DROPOUT" \
      --mask_radius "$MASK_RADIUS" \
      --freeze_strategy "$FREEZE_STRATEGY" \
      --skip_channels "$SKIP_CHANNELS"

  throttle
}

# ==============================================================================
# EXPERIMENTS: SKIP CONNECTION ARCHITECTURE
# ==============================================================================

echo ""
echo "Running 9 experiments with VF skip connections..."
echo ""

# 1. Raw Data Baseline
run_model "$VF_DATA_ROOT/original_below350" \
          "Skip24_Raw_All" \
          "logs/skip_raw_half.log"

# 2-9. All Denoising Methods
run_model "$VF_DATA_ROOT/denoised_n2n_FINAL" \
          "Skip24_N2N_All" \
          "logs/skip_n2n_half.log"

run_model "$VF_DATA_ROOT/denoised_NAFNet_models/denoised_NAFNet_Advanced" \
          "Skip24_NAFNet_All" \
          "logs/skip_nafnet_half.log"

run_model "$VF_DATA_ROOT/denoised_n2v" \
          "Skip24_N2V_All" \
          "logs/skip_n2v_half.log"

run_model "$VF_DATA_ROOT/denoised_TwoStage_Final/denoised_cnn_ae" \
          "Skip24_CNN_AE_All" \
          "logs/skip_cnnae_half.log"

run_model "$VF_DATA_ROOT/denoised_TwoStage_Final/denoised_cnn_vae" \
          "Skip24_CNN_VAE_All" \
          "logs/skip_cnnvae_half.log"

run_model "$VF_DATA_ROOT/denoised_TwoStage_Final/denoised_posenc" \
          "Skip24_PosEnc_All" \
          "logs/skip_posenc_half.log"

run_model "$VF_DATA_ROOT/denoised_TwoStage_Final/denoised_mlp_vae" \
          "Skip24_VAE_All" \
          "logs/skip_vae_half.log"

run_model "$VF_DATA_ROOT/denoised_TwoStage_Final/denoised_mlp_ae" \
          "Skip24_VanillaAE_All" \
          "logs/skip_vanilla_half.log"

wait || true
echo ""
echo "All skip connection experiments completed!"
echo ""
echo "Results saved to:"
echo "   - Checkpoints: checkpoints/Skip24_*_best.pth"
echo "   - Logs: logs/skip_*.log"
echo "   - W&B Project: VF_to_RNFL_SkipConnect"
echo ""