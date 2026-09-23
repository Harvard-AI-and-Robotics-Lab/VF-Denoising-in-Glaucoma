#!/bin/bash
set -euo pipefail

# Data locations (see config/env.example.sh)
: "${VF_DATA_ROOT:?set VF_DATA_ROOT first - see config/env.example.sh}"

# ==============================================================================
# MASTER TEST SCRIPT: AvgVF SKIP CONNECTIONS - PARALLEL EDITION
# Goal: Evaluate all trained AvgVF skip connection models with unified severity
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
  echo "Launching on GPU $gpu -> $log"
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
SEVERITY_MAP="splits/severity_mapping.json"
Z_DIM=32
BATCH_SIZE=32
MASK_RADIUS=30
SKIP_CHANNELS=24

echo "Starting AvgVF Test: Skip Connection Architecture Comparison"
echo "   Skip Channels=$SKIP_CHANNELS, Mask Radius=$MASK_RADIUS"
echo "   Severity Mapping=$SEVERITY_MAP"
echo "   GPUs=${GPUS[*]} | MAX_PARALLEL=$MAX_PARALLEL"
echo "---------------------------------------------------"

# Helper function to launch test experiments
run_test () {
  local test_folder="$1"
  local checkpoint="$2"
  local log_file="$3"

  # Get next GPU
  local gpu="${GPUS[$gpu_index]}"
  gpu_index=$(( (gpu_index + 1) % ${#GPUS[@]} ))

  echo "Assigning GPU $gpu for $(basename "$checkpoint")"

  launch "$gpu" "$log_file" \
    python test_unified_severity_avg.py \
      --test_folder "$test_folder" \
      --checkpoint "$checkpoint" \
      --severity_mapping "$SEVERITY_MAP" \
      --z_dim "$Z_DIM" \
      --batch_size "$BATCH_SIZE" \
      --mask_radius "$MASK_RADIUS" \
      --skip_channels "$SKIP_CHANNELS" \
      --use_split_filter

  throttle
}

# ==============================================================================
# TEST ALL 9 MODELS
# ==============================================================================

echo ""
echo "Running 9 AvgVF test evaluations..."
echo ""

# 1. Raw Data Baseline
run_test "$VF_DATA_ROOT/original_below350" \
         "checkpoints/Skip24_Raw_Avg_best.pth" \
         "logs/test_skip_raw_avg.log"

# 2-9. All Denoising Methods
run_test "$VF_DATA_ROOT/denoised_n2n_FINAL" \
         "checkpoints/Skip24_N2N_Avg_best.pth" \
         "logs/test_skip_n2n_avg.log"

run_test "$VF_DATA_ROOT/denoised_NAFNet_models/denoised_NAFNet_Advanced" \
         "checkpoints/Skip24_NAFNet_Avg_best.pth" \
         "logs/test_skip_nafnet_avg.log"

run_test "$VF_DATA_ROOT/denoised_n2v" \
         "checkpoints/Skip24_N2V_Avg_best.pth" \
         "logs/test_skip_n2v_avg.log"

run_test "$VF_DATA_ROOT/denoised_TwoStage_Final/denoised_cnn_ae" \
         "checkpoints/Skip24_CNN_AE_Avg_best.pth" \
         "logs/test_skip_cnnae_avg.log"

run_test "$VF_DATA_ROOT/denoised_TwoStage_Final/denoised_cnn_vae" \
         "checkpoints/Skip24_CNN_VAE_Avg_best.pth" \
         "logs/test_skip_cnnvae_avg.log"

run_test "$VF_DATA_ROOT/denoised_TwoStage_Final/denoised_posenc" \
         "checkpoints/Skip24_PosEnc_Avg_best.pth" \
         "logs/test_skip_posenc_avg.log"

run_test "$VF_DATA_ROOT/denoised_TwoStage_Final/denoised_mlp_vae" \
         "checkpoints/Skip24_VAE_Avg_best.pth" \
         "logs/test_skip_vae_avg.log"

run_test "$VF_DATA_ROOT/denoised_TwoStage_Final/denoised_mlp_ae" \
         "checkpoints/Skip24_VanillaAE_Avg_best.pth" \
         "logs/test_skip_vanilla_avg.log"

wait || true
echo ""
echo "All AvgVF test evaluations completed!"
echo ""
echo "Results saved to:"
echo "   - Logs: logs/test_skip_*_avg.log"
echo ""
echo "Quick summary:"
for log in logs/test_skip_*_avg.log; do
  if [ -f "$log" ]; then
    echo "--- $(basename "$log") ---"
    grep -A 6 "^====" "$log" | tail -7 || true
    echo ""
  fi
done
