import os
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

# ==========================================
# 1. Configuration
# ==========================================
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")   # override with CUDA_VISIBLE_DEVICES=<id>
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths (see config/env.example.sh)
CODE_DIR     = Path(__file__).resolve().parent
DATA_ROOT    = Path(os.environ.get("VF_DATA_ROOT", "/path/to/vf_oct_pairs"))
DENOISE_ROOT = Path(os.environ.get("VF_DENOISE_ROOT", CODE_DIR))

INPUT_DIR   = DATA_ROOT / "original_below350"
CKPT_PATH   = (DENOISE_ROOT / "model_weight/trained_NAFNet_Advanced/"
                              "NAFNet_Adv_N3_lr0.0001.pth")
OUTPUT_BASE = DATA_ROOT / "denoised_NAFNet_models"

# Add paths to system so imports work
sys.path.append(str(CODE_DIR / "model"))
sys.path.append(str(CODE_DIR / "utils"))

# Normalization Params (Must match training!)
MIN_VAL, MAX_VAL = -38.0, 26.0

# ==========================================
# 2. Helpers
# ==========================================
def norm(x):
    return 2 * (x - MIN_VAL) / (MAX_VAL - MIN_VAL) - 1

def denorm(x):
    return (x + 1) / 2 * (MAX_VAL - MIN_VAL) + MIN_VAL

try:
    from DN_vf_tools import convertvf2image
except ImportError:
    print("Error: 'utils/DN_vf_tools.py' missing.")
    sys.exit(1)

def vector_to_naf_input(td_vector):
    """Converts 1D vector -> 4D Tensor (Batch, Channel, Height, Width)"""
    # 1. Map 52 points to 12x12 grid
    grid = convertvf2image(td_vector)
    
    # 2. Create Mask (where is the actual data?)
    # We will use this later to extract only the valid pixels
    mask = (~np.isnan(grid))
    
    # 3. Clean NaNs (replace with min value for normalization)
    grid_clean = np.nan_to_num(grid, nan=MIN_VAL)
    
    # 4. Normalize to [-1, 1]
    grid_norm = norm(grid_clean)
    
    # 5. Add Batch (1) and Channel (1) dimensions
    # Shape becomes (1, 1, 12, 12)
    tensor = torch.from_numpy(grid_norm).float().unsqueeze(0).unsqueeze(0)
    
    return tensor, mask

def naf_output_to_vector(naf_out_tensor, mask):
    """Converts Model Output Tensor -> 1D vector"""
    # 1. Remove Batch/Channel dims -> (12, 12)
    grid_norm = naf_out_tensor.squeeze().cpu().numpy()
    
    # 2. Denormalize back to dB values
    grid_db = denorm(grid_norm)
    
    # 3. Extract only the valid pixels using the mask we saved earlier
    valid_points = grid_db[mask]
    
    return valid_points.astype(np.float32)

# ==========================================
# 3. Inference Logic
# ==========================================
def main():
    if not INPUT_DIR.exists():
        print(f"Input directory not found: {INPUT_DIR}")
        return

    # --- 1. Load Model ---
    if not CKPT_PATH.exists():
        print(f"Checkpoint not found: {CKPT_PATH}")
        return

    print(f"\nProcessing with NAFNet: {CKPT_PATH.name}")
    
    try:
        from model.NAFNet import NAFNet
    except ImportError:
        print("Error: Could not import NAFNet. Make sure model/NAFNet.py exists.")
        return

    # Initialize structure (Must match training params!)
    model = NAFNet(img_channel=1, width=16, middle_blk_num=1, 
                   enc_blk_nums=[1, 1, 1], dec_blk_nums=[1, 1, 1]).to(DEVICE)
    
    # Load weights
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    model.eval()
    
    # --- 2. Setup Output ---
    save_dir = OUTPUT_BASE / "denoised_NAFNet_Advanced"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    files = list(INPUT_DIR.glob("*.npz"))
    
    # --- 3. Process Files ---
    for f in tqdm(files, desc="Denoising"):
        try:
            # Load Noisy Data
            data = np.load(f, allow_pickle=True)
            td_raw = data['td'].astype(np.float32)
            
            # Prepare Input
            x_tensor, mask = vector_to_naf_input(td_raw)
            x_tensor = x_tensor.to(DEVICE)
            
            # Run Model
            with torch.no_grad():
                # NAFNet output is same size as input (1, 1, 12, 12)
                pred = model(x_tensor)
                
            # Post-process (2D -> 1D)
            td_denoised = naf_output_to_vector(pred, mask)
            
            # Save
            np.savez_compressed(
                save_dir / f.name,
                td=td_denoised,       # <--- The Cleaned Data
                rnflt=data['rnflt'],  # Keep original OCT
                meta=data['meta']     # Keep original Metadata
            )
            
        except Exception as e:
            tqdm.write(f"Error on {f.name}: {e}")

    print(f"Saved results to: {save_dir}")

if __name__ == "__main__":
    main()