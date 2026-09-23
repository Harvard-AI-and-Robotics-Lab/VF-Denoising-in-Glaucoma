
# Data locations (see config/env.example.sh)
: "${VF_DATA_ROOT:?set VF_DATA_ROOT first - see config/env.example.sh}"

GPU_ID=7
BASE="$VF_DATA_ROOT"

echo "Starting S-F Sequential Training on GPU $GPU_ID"

# 1. RAW Baseline
python train_oct2vf.py --folder "$BASE/original_below350" --run_name sf_raw

# 2. N2N
python train_oct2vf.py --folder "$BASE/denoised_n2n_FINAL" --run_name sf_n2n

# 3. NAFNet
python train_oct2vf.py --folder "$BASE/denoised_NAFNet_models/denoised_NAFNet_Advanced" --run_name sf_nafnet

# 4. N2V
python train_oct2vf.py --folder "$BASE/denoised_n2v" --run_name sf_n2v

# 5. CNN_AE
python train_oct2vf.py --folder "$BASE/denoised_TwoStage_Final/denoised_cnn_ae" --run_name sf_cnn_ae

# 6. CNN_VAE
python train_oct2vf.py --folder "$BASE/denoised_TwoStage_Final/denoised_cnn_vae" --run_name sf_cnn_vae

# 7. PosEnc
python train_oct2vf.py --folder "$BASE/denoised_TwoStage_Final/denoised_posenc" --run_name sf_pos_enc

# 8. VAE
python train_oct2vf.py --folder "$BASE/denoised_TwoStage_Final/denoised_mlp_vae" --run_name sf_mlp_vae

# 9. Vanilla_AE
python train_oct2vf.py --folder "$BASE/denoised_TwoStage_Final/denoised_mlp_ae" --run_name sf_mlp_ae

echo "All S-F tasks complete!"