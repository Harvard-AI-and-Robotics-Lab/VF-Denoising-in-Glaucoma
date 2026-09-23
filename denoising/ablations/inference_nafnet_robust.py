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
CODE_DIR     = Path(__file__).resolve().parents[1]          # denoising/
DATA_ROOT    = Path(os.environ.get("VF_DATA_ROOT", "/path/to/vf_oct_pairs"))
DENOISE_ROOT = Path(os.environ.get("VF_DENOISE_ROOT", CODE_DIR))

INPUT_DIR   = DATA_ROOT / "original_below350"
WEIGHTS_DIR = DENOISE_ROOT / "model_weight/trained_NAFNet_Advanced"
OUTPUT_BASE = DATA_ROOT / "denoised_NAFNet_final"

sys.path.append(str(CODE_DIR / "model"))
sys.path.append(str(CODE_DIR / "utils"))

MIN_VAL, MAX_VAL = -38.0, 26.0

# ==========================================
# 2. Helpers
# ==========================================
def norm(x): return 2 * (x - MIN_VAL) / (MAX_VAL - MIN_VAL) - 1
def denorm(x): return (x + 1) / 2 * (MAX_VAL - MIN_VAL) + MIN_VAL

try:
    from DN_vf_tools import convertvf2image
except ImportError:
    print("Error: 'utils/DN_vf_tools.py' missing.")
    sys.exit(1)

def vector_to_naf_input(td_vector):
    grid = convertvf2image(td_vector)
    mask = (~np.isnan(grid))
    grid_clean = np.nan_to_num(grid, nan=MIN_VAL)
    grid_norm = norm(grid_clean)
    tensor = torch.from_numpy(grid_norm).float().unsqueeze(0).unsqueeze(0)
    return tensor, mask

def naf_output_to_vector(naf_out_tensor, mask):
    grid_norm = naf_out_tensor.squeeze().cpu().numpy()
    grid_db = denorm(grid_norm)
    valid_points = grid_db[mask]
    return valid_points.astype(np.float32)

def smart_load_weights(model, ckpt_path):
    """
    Robustly loads weights, handling 'state_dict' keys and 'module.' prefixes.
    """
    print(f"Loading weights from: {ckpt_path.name}")
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    
    # 1. Unwrap if it's a dict containing 'state_dict' (common in PyTorch Lightning/some repos)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        print("   -> Found 'state_dict' key in checkpoint.")
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        print("   -> Found 'model_state_dict' key in checkpoint.")
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # 2. Fix 'module.' prefix (if trained on Multi-GPU DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    # 3. Load
    try:
        model.load_state_dict(new_state_dict, strict=True)
        print("Weights loaded successfully (Strict Mode).")
    except Exception as e:
        print(f"Strict loading failed: {e}")
        print("   -> Retrying with strict=False (Partial Load)...")
        model.load_state_dict(new_state_dict, strict=False)
        
    return model

# ==========================================
# 3. Main
# ==========================================
def main():
    if not INPUT_DIR.exists(): return

    # Checkpoint Search
    files = list(WEIGHTS_DIR.glob("*.pth"))
    if not files: print("No .pth files found!"); return
    files.sort(key=os.path.getmtime, reverse=True)
    ckpt = files[0]

    # Load NAFNet
    from model.NAFNet import NAFNet
    model = NAFNet(img_channel=1, width=16, middle_blk_num=1, 
                   enc_blk_nums=[1, 1, 1], dec_blk_nums=[1, 1, 1]).to(DEVICE)
    
    # --- SANITY CHECK BEFORE LOAD ---
    w_before = model.intro.weight.mean().item()
    
    # Load
    model = smart_load_weights(model, ckpt)
    model.eval()
    
    # --- SANITY CHECK AFTER LOAD ---
    w_after = model.intro.weight.mean().item()
    if w_before == w_after:
        print("\nCRITICAL WARNING: Model weights did not change after loading!")
        print("   This means the model is UNTRAINED (Random Init).")
        print("   Output would be identical to the input; check the checkpoint path.\n")
    else:
        print(f"   (Weight check pass: {w_before:.5f} -> {w_after:.5f})\n")

    # Output
    save_dir = OUTPUT_BASE / "denoised_NAFNet_Advanced"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    files = list(INPUT_DIR.glob("*.npz"))
    
    for f in tqdm(files, desc="Denoising"):
        try:
            data = np.load(f, allow_pickle=True)
            td_raw = data['td'].astype(np.float32)
            
            x_tensor, mask = vector_to_naf_input(td_raw)
            x_tensor = x_tensor.to(DEVICE)
            
            with torch.no_grad():
                pred = model(x_tensor)
            
            # If pred == x_tensor, the model applied no correction.
                
            td_denoised = naf_output_to_vector(pred, mask)
            
            np.savez_compressed(
                save_dir / f.name,
                td=td_denoised,       
                rnflt=data['rnflt'],  
                meta=data['meta']     
            )
        except Exception as e:
            tqdm.write(f"Error: {e}")

    print(f"Saved to: {save_dir}")

if __name__ == "__main__":
    main()