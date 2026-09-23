#!/bin/bash
set -euo pipefail

# Data locations (see config/env.example.sh)
: "${VF_DATA_ROOT:?set VF_DATA_ROOT first - see config/env.example.sh}"

# ==============================================================================
# BATCH TEST SCRIPT: Unified Severity Evaluation
# Tests all trained models with consistent severity grouping from raw data
# ==============================================================================


echo "BATCH TESTING WITH UNIFIED SEVERITY MAPPING"
echo "=============================================="
echo ""

# ---------------- SETTINGS (MATCH TRAINING) ----------------
SEVERITY_MAP="splits/severity_mapping.json"
SKIP_CHANNELS=24  # Must match training
Z_DIM=32
MASK_RADIUS=30
BATCH_SIZE=32

# Check if severity mapping exists
if [ ! -f "$SEVERITY_MAP" ]; then
    echo "ERROR: Severity mapping not found: $SEVERITY_MAP"
    echo ""
    echo "Please run this first:"
    echo "  python unified_split_severity.py \\"
    echo "    --raw_folder $VF_DATA_ROOT/original_below350 \\"
    echo "    --output_dir ./splits \\"
    echo "    --seed 42"
    echo ""
    exit 1
fi

echo "Using severity mapping: $SEVERITY_MAP"
echo "   Skip Channels: $SKIP_CHANNELS"
echo "   Z Dimension: $Z_DIM"
echo "   Mask Radius: ${MASK_RADIUS}px"
echo ""
echo "=============================================="
echo ""

# Create results directory
mkdir -p test_results

# ---------------- TEST FUNCTIONS ----------------
test_model () {
    local dataset_folder="$1"
    local checkpoint="$2"
    local dataset_name="$3"
    local result_file="$4"
    
    echo "Testing: $dataset_name"
    echo "   Dataset: $dataset_folder"
    echo "   Checkpoint: $checkpoint"
    echo ""
    
    # Check if checkpoint exists
    if [ ! -f "$checkpoint" ]; then
        echo "WARNING: Checkpoint not found: $checkpoint"
        echo "   Skipping $dataset_name..."
        echo ""
        return
    fi
    
    python test_unified_severity.py \
        --test_folder "$dataset_folder" \
        --checkpoint "$checkpoint" \
        --severity_mapping "$SEVERITY_MAP" \
        --skip_channels "$SKIP_CHANNELS" \
        --z_dim "$Z_DIM" \
        --mask_radius "$MASK_RADIUS" \
        --batch_size "$BATCH_SIZE" \
        --use_split_filter \
        | tee "$result_file"
    
    echo ""
    echo "Completed: $dataset_name"
    echo "   Results saved to: $result_file"
    echo ""
    echo "---------------------------------------------------"
    echo ""
}

# ==============================================================================
# RUN TESTS FOR ALL MODELS
# ==============================================================================

echo "Starting tests..."
echo ""

# 1. Raw Data Baseline
test_model \
    "$VF_DATA_ROOT/original_below350" \
    "checkpoints/Skip24_Raw_All_best.pth" \
    "Raw (Baseline)" \
    "test_results/raw_results.txt"

# 2. Noise2Noise
test_model \
    "$VF_DATA_ROOT/denoised_n2n_FINAL" \
    "checkpoints/Skip24_N2N_All_best.pth" \
    "Noise2Noise" \
    "test_results/n2n_results.txt"

# 3. NAFNet Advanced
test_model \
    "$VF_DATA_ROOT/denoised_NAFNet_models/denoised_NAFNet_Advanced" \
    "checkpoints/Skip24_NAFNet_All_best.pth" \
    "NAFNet Advanced" \
    "test_results/nafnet_results.txt"

# 4. Noise2Void
test_model \
    "$VF_DATA_ROOT/denoised_n2v" \
    "checkpoints/Skip24_N2V_All_best.pth" \
    "Noise2Void" \
    "test_results/n2v_results.txt"

# 5. CNN Autoencoder
test_model \
    "$VF_DATA_ROOT/denoised_TwoStage_Final/denoised_cnn_ae" \
    "checkpoints/Skip24_CNN_AE_All_best.pth" \
    "CNN Autoencoder" \
    "test_results/cnn_ae_results.txt"

# 6. CNN VAE
test_model \
    "$VF_DATA_ROOT/denoised_TwoStage_Final/denoised_cnn_vae" \
    "checkpoints/Skip24_CNN_VAE_All_best.pth" \
    "CNN VAE" \
    "test_results/cnn_vae_results.txt"

# 7. Positional Encoding
test_model \
    "$VF_DATA_ROOT/denoised_TwoStage_Final/denoised_posenc" \
    "checkpoints/Skip24_PosEnc_All_best.pth" \
    "Positional Encoding" \
    "test_results/posenc_results.txt"

# 8. MLP VAE
test_model \
    "$VF_DATA_ROOT/denoised_TwoStage_Final/denoised_mlp_vae" \
    "checkpoints/Skip24_VAE_All_best.pth" \
    "MLP VAE" \
    "test_results/vae_results.txt"

# 9. Vanilla Autoencoder
test_model \
    "$VF_DATA_ROOT/denoised_TwoStage_Final/denoised_mlp_ae" \
    "checkpoints/Skip24_VanillaAE_All_best.pth" \
    "Vanilla Autoencoder" \
    "test_results/vanilla_ae_results.txt"

# ==============================================================================
# SUMMARY
# ==============================================================================

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                   ALL TESTS COMPLETED! ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Individual results saved to: test_results/"
echo ""
echo "Quick Summary (MAE in µm):"
echo "─────────────────────────────────────────────────────────────────"

# Extract MAE and R² for "All" group from each result file
for result_file in test_results/*_results.txt; do
    if [ -f "$result_file" ]; then
        method=$(basename "$result_file" _results.txt)
        
        # Extract the "All" line and parse it properly
        # Format: "All             | 1234       |   12.34 ±  3.45    |   0.7890  |   0.7654"
        all_line=$(grep "^All " "$result_file" || echo "")
        
        if [ -n "$all_line" ]; then
            # Split by | and extract fields
            mae=$(echo "$all_line" | awk -F'|' '{print $3}' | awk '{print $1}' || echo "N/A")
            global_r2=$(echo "$all_line" | awk -F'|' '{print $5}' | awk '{print $1}' || echo "N/A")
            printf "%-20s | MAE: %8s | R²: %s\n" "$method" "$mae" "$global_r2"
        else
            printf "%-20s | MAE: %8s | R²: %s\n" "$method" "N/A" "N/A"
        fi
    fi
done

echo "─────────────────────────────────────────────────────────────────"
echo ""
echo "To view detailed results for a specific method:"
echo "   cat test_results/nafnet_results.txt"
echo ""
echo "To compare severity groups across methods:"
echo "   grep 'Severe' test_results/*.txt"
echo ""
echo "Key insight: All results use the SAME severity grouping based on"
echo "   the raw data, ensuring fair comparison across denoising methods!"
echo ""