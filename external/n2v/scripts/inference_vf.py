import os
import numpy as np
import tensorflow as tf
from n2v.models import N2V
from pathlib import Path
from tqdm import tqdm

# ================= CONFIGURATION =================
# 1. Model Name (Must match training)
MODEL_NAME = 'n2v_vf_8x9'
BASE_DIR   = 'models'

# 2. Paths
# Data locations come from the environment (see config/env.example.sh)
DATA_ROOT  = Path(os.environ.get("VF_DATA_ROOT", "/path/to/vf_oct_pairs"))
INPUT_DIR  = DATA_ROOT / "original_below350"
OUTPUT_DIR = DATA_ROOT / "denoised_n2v"

# ================= MAPPING LOGIC (8x9 Grid) =================
# Matches the 24-2 grid layout used throughout the study
def get_valid_indices_8x9():
    nulls = [0,1,2,7,8,9,10,17,18,34,43,45,54,55,62,63,64,65,70,71]
    all_indices = np.arange(8 * 9) # 0 to 71
    return [i for i in all_indices if i not in nulls]

def vector_to_img_8x9(vector):
    """Maps (52,) vector -> (8, 9, 1) image"""
    H, W = 8, 9
    img_flat = np.zeros(H * W, dtype=np.float32)
    valid_indices = get_valid_indices_8x9()
    
    limit = min(len(valid_indices), len(vector))
    img_flat[valid_indices[:limit]] = vector[:limit]
    
    return img_flat.reshape(H, W, 1), valid_indices

def img_to_vector(img, valid_indices, vec_len=52):
    """Maps (8, 9, 1) image -> (52,) vector"""
    flat = img.flatten()
    extracted = flat[valid_indices]
    
    # Ensure exact length 52
    out = np.zeros(vec_len, dtype=np.float32)
    limit = min(len(extracted), vec_len)
    out[:limit] = extracted[:limit]
    return out

# ================= EXECUTION =================
def main():
    # 1. Patch N2V (Just in case initialization triggers callbacks)
    import csbdeep.utils.tf
    class Dummy(tf.keras.callbacks.Callback): pass
    csbdeep.utils.tf.CARETensorBoardImage = Dummy

    print(f"Loading N2V Model: {MODEL_NAME}")
    # config=None puts it in inference mode (loads config from disk)
    model = N2V(config=None, name=MODEL_NAME, basedir=BASE_DIR)

    # 2. Setup Output
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Created output directory: {OUTPUT_DIR}")

    files = list(INPUT_DIR.glob("*.npz"))
    print(f"Found {len(files)} files in {INPUT_DIR}")
    print(f"Saving to {OUTPUT_DIR}")

    success_count = 0
    
    # 3. Batch Inference Loop
    for f in tqdm(files, desc="Denoising"):
        try:
            # Load
            data = np.load(f, allow_pickle=True)
            td_raw = data['td'].astype(np.float32)
            
            # --- PREPROCESSING ---
            # A. Vector -> 8x9 Image
            img_in_8x9, valid_idxs = vector_to_img_8x9(td_raw)
            
            # B. Pad -> 8x12 (Divisible by 4 for U-Net)
            # Pad 3 columns on the right: ((top, bot), (left, right), (channel_before, channel_after))
            img_padded = np.pad(img_in_8x9, ((0,0), (0,3), (0,0)), mode='constant')
            
            # --- INFERENCE ---
            # C. Predict
            # Input: (8, 12, 1). Axes: 'YXC'.
            # N2V handles the batch dimension internally if missing.
            pred_padded = model.predict(img_padded, axes='YXC')
            
            # --- POSTPROCESSING ---
            # D. Crop -> 8x9 (Remove the 3 padded columns)
            pred_8x9 = pred_padded[:, :9, :]
            
            # E. Image -> Vector
            td_denoised = img_to_vector(pred_8x9, valid_idxs, vec_len=len(td_raw))
            
            # --- SAVE ---
            np.savez_compressed(
                OUTPUT_DIR / f.name,
                td=td_denoised,         # Denoised TD
                rnflt=data['rnflt'], # Keep original OCT
                meta=data['meta']    # Keep original Meta
            )
            success_count += 1
            
        except Exception as e:
            tqdm.write(f"Error {f.name}: {e}")

    print(f"Generation Complete. Processed {success_count}/{len(files)} files.")

if __name__ == "__main__":
    main()

# nohup python3 inference_vf.py > vf_inference.log 2>&1 &