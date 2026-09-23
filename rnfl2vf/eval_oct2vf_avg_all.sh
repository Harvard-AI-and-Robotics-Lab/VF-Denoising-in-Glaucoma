#!/bin/bash

# Data locations (see config/env.example.sh)
: "${VF_DATA_ROOT:?set VF_DATA_ROOT first - see config/env.example.sh}"

BASE="$VF_DATA_ROOT"

echo "Starting S-F AVG vs Original Model Evaluation"

# 1. RAW Baseline
python eval_oct2vf_avg.py \
    --folder "$BASE/original_below350" \
    --checkpoint checkpoints/sf_avg_raw.pth \
    --baseline_checkpoint checkpoints/sf_raw.pth \
    --output_name sf_avg_raw

# 2. N2N
python eval_oct2vf_avg.py \
    --folder "$BASE/denoised_n2n_FINAL" \
    --checkpoint checkpoints/sf_avg_n2n.pth \
    --baseline_checkpoint checkpoints/sf_n2n.pth \
    --output_name sf_avg_n2n

# 3. NAFNet
python eval_oct2vf_avg.py \
    --folder "$BASE/denoised_NAFNet_models/denoised_NAFNet_Advanced" \
    --checkpoint checkpoints/sf_avg_nafnet.pth \
    --baseline_checkpoint checkpoints/sf_nafnet.pth \
    --output_name sf_avg_nafnet

# 4. N2V
python eval_oct2vf_avg.py \
    --folder "$BASE/denoised_n2v" \
    --checkpoint checkpoints/sf_avg_n2v.pth \
    --baseline_checkpoint checkpoints/sf_n2v.pth \
    --output_name sf_avg_n2v

# 5. CNN_AE
python eval_oct2vf_avg.py \
    --folder "$BASE/denoised_TwoStage_Final/denoised_cnn_ae" \
    --checkpoint checkpoints/sf_avg_cnn_ae.pth \
    --baseline_checkpoint checkpoints/sf_cnn_ae.pth \
    --output_name sf_avg_cnn_ae

# 6. CNN_VAE
python eval_oct2vf_avg.py \
    --folder "$BASE/denoised_TwoStage_Final/denoised_cnn_vae" \
    --checkpoint checkpoints/sf_avg_cnn_vae.pth \
    --baseline_checkpoint checkpoints/sf_cnn_vae.pth \
    --output_name sf_avg_cnn_vae

# 7. PosEnc
python eval_oct2vf_avg.py \
    --folder "$BASE/denoised_TwoStage_Final/denoised_posenc" \
    --checkpoint checkpoints/sf_avg_pos_enc.pth \
    --baseline_checkpoint checkpoints/sf_pos_enc.pth \
    --output_name sf_avg_pos_enc

# 8. MLP_VAE
python eval_oct2vf_avg.py \
    --folder "$BASE/denoised_TwoStage_Final/denoised_mlp_vae" \
    --checkpoint checkpoints/sf_avg_mlp_vae.pth \
    --baseline_checkpoint checkpoints/sf_mlp_vae.pth \
    --output_name sf_avg_mlp_vae

# 9. MLP_AE
python eval_oct2vf_avg.py \
    --folder "$BASE/denoised_TwoStage_Final/denoised_mlp_ae" \
    --checkpoint checkpoints/sf_avg_mlp_ae.pth \
    --baseline_checkpoint checkpoints/sf_mlp_ae.pth \
    --output_name sf_avg_mlp_ae

echo "All S-F AVG evaluations complete! Results in eval_results/"
