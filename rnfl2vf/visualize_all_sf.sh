#!/bin/bash

# Data locations (see config/env.example.sh)
: "${VF_DATA_ROOT:?set VF_DATA_ROOT first - see config/env.example.sh}"


GPU_ID=7
BASE="$VF_DATA_ROOT"
NUM_SAMPLES=3

echo "Starting S-F Visualization Generation on GPU $GPU_ID"
export CUDA_VISIBLE_DEVICES=$GPU_ID

# 1. RAW Baseline
echo "Visualizing RAW..."
python visualize_sf_predictions.py \
    --folder "$BASE/original_below350" \
    --checkpoint checkpoints/sf_raw.pth \
    --output_name raw \
    --num_samples $NUM_SAMPLES

# 2. N2N
echo "Visualizing N2N..."
python visualize_sf_predictions.py \
    --folder "$BASE/denoised_n2n_FINAL" \
    --checkpoint checkpoints/sf_n2n.pth \
    --output_name n2n \
    --num_samples $NUM_SAMPLES

# 3. NAFNet
echo "Visualizing NAFNet..."
python visualize_sf_predictions.py \
    --folder "$BASE/denoised_NAFNet_models/denoised_NAFNet_Advanced" \
    --checkpoint checkpoints/sf_nafnet.pth \
    --output_name nafnet \
    --num_samples $NUM_SAMPLES

# 4. N2V
echo "Visualizing N2V..."
python visualize_sf_predictions.py \
    --folder "$BASE/denoised_n2v" \
    --checkpoint checkpoints/sf_n2v.pth \
    --output_name n2v \
    --num_samples $NUM_SAMPLES

# 5. CNN_AE
echo "Visualizing CNN_AE..."
python visualize_sf_predictions.py \
    --folder "$BASE/denoised_TwoStage_Final/denoised_cnn_ae" \
    --checkpoint checkpoints/sf_cnn_ae.pth \
    --output_name cnn_ae \
    --num_samples $NUM_SAMPLES

# 6. CNN_VAE
echo "Visualizing CNN_VAE..."
python visualize_sf_predictions.py \
    --folder "$BASE/denoised_TwoStage_Final/denoised_cnn_vae" \
    --checkpoint checkpoints/sf_cnn_vae.pth \
    --output_name cnn_vae \
    --num_samples $NUM_SAMPLES

# 7. PosEnc
echo "Visualizing PosEnc..."
python visualize_sf_predictions.py \
    --folder "$BASE/denoised_TwoStage_Final/denoised_posenc" \
    --checkpoint checkpoints/sf_pos_enc.pth \
    --output_name pos_enc \
    --num_samples $NUM_SAMPLES

# 8. MLP_VAE
echo "Visualizing MLP_VAE..."
python visualize_sf_predictions.py \
    --folder "$BASE/denoised_TwoStage_Final/denoised_mlp_vae" \
    --checkpoint checkpoints/sf_mlp_vae.pth \
    --output_name mlp_vae \
    --num_samples $NUM_SAMPLES

# 9. MLP_AE
echo "Visualizing MLP_AE..."
python visualize_sf_predictions.py \
    --folder "$BASE/denoised_TwoStage_Final/denoised_mlp_ae" \
    --checkpoint checkpoints/sf_mlp_ae.pth \
    --output_name mlp_ae \
    --num_samples $NUM_SAMPLES

echo "All visualizations complete!"
echo "Check folders: sf_visualizations_*/"