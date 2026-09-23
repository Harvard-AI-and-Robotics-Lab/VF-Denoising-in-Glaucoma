#!/bin/bash

# Data locations (see config/env.example.sh)
: "${VF_DENOISE_ROOT:?set VF_DENOISE_ROOT first - see config/env.example.sh}"

# =================================================================
# CONFIGURATION: dataset paths
# =================================================================

# Path for MLP/PosEnc Data (1D Vectors, 52 points)

MLP_TRAIN="$VF_DENOISE_ROOT/dataset_train_val_test/MLP/fine_tune_dataset_N3_train.npz"
MLP_VAL="$VF_DENOISE_ROOT/dataset_train_val_test/MLP/fine_tune_dataset_N3_val.npz"

# Path for CNN Data (2D Images, 12x12 grid)
CNN_TRAIN="$VF_DENOISE_ROOT/dataset_train_val_test/CNN/vf_cnn_finetune_retest.npz"
CNN_VAL="$VF_DENOISE_ROOT/dataset_train_val_test/CNN/vf_cnn_finetune_retest_val.npz"

# Output Directory
SAVE_DIR="$VF_DENOISE_ROOT/model_weight/TwoStage_Final"

# Common Settings
EPOCHS_S1=50
EPOCHS_S2=50
BS=128
LATENT=8

# =================================================================
# 1. MLP MODELS (Use MLP Data)
# =================================================================

echo "Running MLP-AE (Vanilla)..."
python train_two_stage.py \
    --model mlp_ae \
    --train_path "$MLP_TRAIN" \
    --val_path "$MLP_VAL" \
    --save_dir "$SAVE_DIR" \
    --latent_dim $LATENT \
    --stage1_epochs $EPOCHS_S1 --stage2_epochs $EPOCHS_S2 \
    --batch_size $BS

echo "Running MLP-VAE (Beta=0.01)..."
python train_two_stage.py \
    --model mlp_vae \
    --train_path "$MLP_TRAIN" \
    --val_path "$MLP_VAL" \
    --save_dir "$SAVE_DIR" \
    --latent_dim $LATENT \
    --beta 0.01 \
    --stage1_epochs $EPOCHS_S1 --stage2_epochs $EPOCHS_S2 \
    --batch_size $BS

echo "Running PosEnc-AE (Transformer-like)..."
python train_two_stage.py \
    --model posenc \
    --train_path "$MLP_TRAIN" \
    --val_path "$MLP_VAL" \
    --save_dir "$SAVE_DIR" \
    --latent_dim $LATENT \
    --stage1_epochs $EPOCHS_S1 --stage2_epochs $EPOCHS_S2 \
    --batch_size $BS

# =================================================================
# 2. CNN MODELS (Use CNN Data)
# =================================================================

echo "Running CNN-AE (Spatial)..."
python train_two_stage.py \
    --model cnn_ae \
    --train_path "$CNN_TRAIN" \
    --val_path "$CNN_VAL" \
    --save_dir "$SAVE_DIR" \
    --latent_dim $LATENT \
    --stage1_epochs $EPOCHS_S1 --stage2_epochs $EPOCHS_S2 \
    --batch_size $BS

echo "Running CNN-VAE (Beta=0.10)..."
python train_two_stage.py \
    --model cnn_vae \
    --train_path "$CNN_TRAIN" \
    --val_path "$CNN_VAL" \
    --save_dir "$SAVE_DIR" \
    --latent_dim $LATENT \
    --beta 0.10 \
    --stage1_epochs $EPOCHS_S1 --stage2_epochs $EPOCHS_S2 \
    --batch_size $BS

echo "All experiments completed!"