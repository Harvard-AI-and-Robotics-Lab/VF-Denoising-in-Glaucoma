import numpy as np

# ==============================================================================
# 1. NUMPY COMPATIBILITY PATCH
# ==============================================================================
try:
    if not hasattr(np, 'object'): np.object = object
    if not hasattr(np, 'bool'):   np.bool = bool
    if not hasattr(np, 'int'):    np.int = int
    if not hasattr(np, 'float'):  np.float = float
    if not hasattr(np, 'complex'):np.complex = complex
except Exception:
    pass

import os
import sys
import pickle
from pathlib import Path
from tqdm import tqdm
import tensorflow.compat.v1 as tf

# ================= CONFIGURATION =================
SNAPSHOT_PATH = "results/vf-mlp-n2n-dynamic/network_final.pickle" 
# Data locations come from the environment (see config/env.example.sh)
_DATA_ROOT    = os.environ.get("VF_DATA_ROOT", "/path/to/vf_oct_pairs")
INPUT_FOLDER  = os.path.join(_DATA_ROOT, "original_below350")
# NOTE: the original scripts wrote to "denoised_n2n"; the directory was renamed
#       to "denoised_n2n_FINAL" before evaluation, which is the name the
#       structure-function scripts expect.
OUTPUT_FOLDER = os.path.join(_DATA_ROOT, "denoised_n2n_FINAL")
TRAIN_NPZ     = "vf_dataset/dataset/dataset_finetune_train_retest/fine_tune_dataset_Nge2.npz"

# Normalization is disabled here to match the training configuration
NORMALIZE = False 
# =================================================

# Force TF 1.x behavior
tf.disable_v2_behavior()
sys.modules["tensorflow"] = tf

import dnnlib
import dnnlib.tflib as tflib
import dnnlib.tflib.tfutil as tfutil

def main():
    print(f"Starting N2N Inference (CPU-Safe Mode)")
    
    # ==========================================================================
    # 2. INITIALIZE TENSORFLOW PROPERLY (Fixing the Error)
    # dnnlib expects a DICTIONARY, not a tf.ConfigProto object.
    # We set 'allow_soft_placement' to True to stop the GPU crash.
    # ==========================================================================
    print("   Initializing TensorFlow Session...")
    
    tf_config_dict = {
        "gpu_options.allow_growth": True,
        "allow_soft_placement": True,   # <--- Tells TF to use CPU if GPU is missing
        "log_device_placement": False
    }
    
    tfutil.init_tf(tf_config_dict)
    
    # 3. LOAD NETWORK
    print(f"   Loading Model: {SNAPSHOT_PATH}")
    if not os.path.exists(SNAPSHOT_PATH):
        print("Error: Snapshot file not found!")
        return

    with open(SNAPSHOT_PATH, "rb") as f:
        net = pickle.load(f)

    # 4. VERIFY MODEL DIMENSIONS MATCH TRAINING (52-dim VF TD)
    EXPECTED_DIM = 52
    net_input_shape = net.input_shapes[0]  # e.g. [None, 52]
    model_dim = net_input_shape[-1]
    print(f"\n   Model input shape: {net_input_shape}")
    if model_dim != EXPECTED_DIM:
        print(f"DIMENSION MISMATCH: model expects {model_dim}, training used {EXPECTED_DIM}")
        return
    print(f"   Model expects {model_dim}-dim input (matches training)")

    # 5. DISTRIBUTION COMPARISON: Training (Nge2) vs Inference input
    print("\nDISTRIBUTION COMPARISON (Train vs Inference):")
    files = list(Path(INPUT_FOLDER).glob("*.npz"))

    # Load training distribution
    train_data = np.load(TRAIN_NPZ)
    train_td = train_data["td"].astype(np.float32)  # (N, 52)

    # Sample inference inputs (cap at 5000 to keep it fast)
    sample_cap = min(5000, len(files))
    sample_idx = np.random.choice(len(files), sample_cap, replace=False)
    inf_vecs = []
    for i in sample_idx:
        d = np.load(files[i], allow_pickle=True)
        if "td" in d:
            inf_vecs.append(d["td"].astype(np.float32).flatten())
    inf_td = np.stack(inf_vecs)  # (sample_cap, 52)

    # Per-location (pointwise) stats
    train_mean = train_td.mean(axis=0)  # (52,)
    inf_mean   = inf_td.mean(axis=0)
    train_std  = train_td.std(axis=0)
    inf_std    = inf_td.std(axis=0)

    # Global stats
    print(f"   {'':20s} {'Train (Nge2)':>14s}  {'Inference':>14s}")
    print(f"   {'Samples':20s} {train_td.shape[0]:>14d}  {inf_td.shape[0]:>14d}")
    print(f"   {'Global Mean (dB)':20s} {train_td.mean():>14.2f}  {inf_td.mean():>14.2f}")
    print(f"   {'Global Std  (dB)':20s} {train_td.std():>14.2f}  {inf_td.std():>14.2f}")
    print(f"   {'Min         (dB)':20s} {train_td.min():>14.1f}  {inf_td.min():>14.1f}")
    print(f"   {'Max         (dB)':20s} {train_td.max():>14.1f}  {inf_td.max():>14.1f}")

    # Per-location mean absolute difference
    loc_mean_diff = np.abs(train_mean - inf_mean).mean()
    loc_std_diff  = np.abs(train_std - inf_std).mean()
    print(f"\n   Avg per-location mean diff:   {loc_mean_diff:.2f} dB")
    print(f"   Avg per-location std  diff:   {loc_std_diff:.2f} dB")

    if loc_mean_diff > 5.0:
        print("   WARNING: Large mean shift — inference data may be out-of-distribution!")
    elif loc_mean_diff > 2.0:
        print("   NOTICE: Moderate mean shift — check if this is expected.")
    else:
        print("   Distributions look compatible.")

    # 6. SANITY CHECK (using real samples from inference set)
    print("\nSANITY CHECK (real VF samples):")
    n_check = min(5, len(inf_vecs))
    check_in = np.stack(inf_vecs[:n_check])  # (n_check, 52) — real VF data

    try:
        check_out = net.run(check_in)

        print(f"   Samples used:  {n_check}")
        print(f"   Input Mean:    {check_in.mean():.2f} dB")
        print(f"   Output Mean:   {check_out.mean():.2f} dB")
        print(f"   Input StdDev:  {check_in.std():.4f}")
        print(f"   Output StdDev: {check_out.std():.4f}")

        abs_diff = np.abs(check_out - check_in).mean()
        print(f"   Mean |diff|:   {abs_diff:.2f} dB")

        if np.allclose(check_in, check_out, atol=1e-5):
            print("CRITICAL FAILURE: Model is Identity (Input == Output).")
            return
        elif abs_diff < 0.01:
            print("CRITICAL FAILURE: Model output is near-identical to input.")
            return
        else:
            print(f"Model is active (avg change = {abs_diff:.2f} dB per location).")

    except Exception as e:
        print(f"Sanity check failed: {e}")
        return

    # 7. RUN INFERENCE
    print(f"\nProcessing files from: {INPUT_FOLDER}")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    count = 0
    for f in tqdm(files):
        try:
            # Load Data
            data = np.load(f, allow_pickle=True)
            if 'td' in data: td_raw = data['td']
            elif 'arr_0' in data: td_raw = data['arr_0']
            else: continue
            
            td_raw = td_raw.astype(np.float32)

            # Dimension check: must be (52,) or (N,52)
            feat_dim = td_raw.shape[-1] if td_raw.ndim <= 2 else None
            if feat_dim != EXPECTED_DIM:
                print(f"   SKIP {f.name}: dim={feat_dim}, expected {EXPECTED_DIM}")
                continue

            # Reshape for batch (1, 52)
            if td_raw.ndim == 1:
                td_in = td_raw[np.newaxis, :]
            else:
                td_in = td_raw

            # Run (Pass RAW data, receive RAW data)
            denoised_out = net.run(td_in)
            
            # Save
            save_path = os.path.join(OUTPUT_FOLDER, f.name)
            np.savez_compressed(
                save_path,
                td = denoised_out.flatten(),       
                rnflt = data['rnflt'],  
                meta = data['meta']     
            )
            count += 1
            
        except Exception as e:
            pass

    print(f"Finished. Processed {count} files.")

if __name__ == "__main__":
    main()