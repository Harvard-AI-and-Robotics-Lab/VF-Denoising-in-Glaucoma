# ---------------------------------------------------------------------------
# Dataset locations used by every script in this repository.
#
#   cp config/env.example.sh config/env.sh
#   # edit the three paths below, then:
#   source config/env.sh
#
# The code never stores absolute paths of its own; each script reads these
# variables (Python: os.environ.get("VF_DATA_ROOT"), shell: "$VF_DATA_ROOT").
# ---------------------------------------------------------------------------

# 1. Paired VF-OCT samples, one .npz per test, plus the denoised copies of them.
#
#    $VF_DATA_ROOT/
#      original_below350/                      raw (undenoised) VF-OCT pairs
#      denoised_n2n_FINAL/                     Noise2Noise
#      denoised_n2v/                           Noise2Void
#      denoised_NAFNet_models/
#        denoised_NAFNet_Advanced/             pathology-aware NAFNet
#      denoised_TwoStage_Final/
#        denoised_mlp_ae/  denoised_mlp_vae/   compression models
#        denoised_cnn_ae/  denoised_cnn_vae/
#        denoised_posenc/
#
#    Each .npz holds:  td (52,) float32   - total deviation, dB
#                      rnflt (200,200) float32 - OCT RNFL thickness map, um
#                      meta - JSON with id, eye, vf_time, oct_time, delta_days, md, vfi
export VF_DATA_ROOT=/path/to/vf_oct_pairs

# 2. Working directory for the denoising models: VF-only training sets and
#    model weights.
#
#    $VF_DENOISE_ROOT/
#      dataset_train_val_test/{MLP,CNN,VAE}/*.npz
#      model_weight/TwoStage_Final/            compression-model checkpoints
#      model_weight/trained_NAFNet_Advanced/   NAFNet checkpoints
export VF_DENOISE_ROOT=/path/to/denoising_workdir

# 3. Raw clinical exports used only by data_preparation/ (not redistributable).
#      boland190822_SS24.20.20.33.RData
#      hfa_ongoing_merged_24-2_subset_cleaned_all_in.csv
#      metadata_OpticDiscCube_ongoing_cleaned.csv
export VF_SOURCE_ROOT=/path/to/clinical_exports

# Optional: weights & biases is used for training curves only.
# export WANDB_MODE=disabled
