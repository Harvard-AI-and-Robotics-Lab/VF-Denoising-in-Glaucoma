import os
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

# ==========================================
# 0. SETUP & IMPORTS
# ==========================================
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")   # override with CUDA_VISIBLE_DEVICES=<id>
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths (see config/env.example.sh)
CODE_DIR     = Path(__file__).resolve().parent          # denoising/N3
DATA_ROOT    = Path(os.environ.get("VF_DATA_ROOT", "/path/to/vf_oct_pairs"))
DENOISE_ROOT = Path(os.environ.get("VF_DENOISE_ROOT", CODE_DIR.parent))

INPUT_DIR   = DATA_ROOT / "original_below350"
WEIGHTS_DIR = DENOISE_ROOT / "model_weight/TwoStage_Final"  # Where we saved the models
OUTPUT_BASE = DATA_ROOT / "denoised_TwoStage_Final"

# Normalization Params
MIN_VAL, MAX_VAL = -38.0, 26.0

sys.path.append(str(CODE_DIR))
sys.path.append(str(CODE_DIR.parent))   # denoising/ -> provides utils/
try:
    from models import CNN_AE, CNN_VAE, MLP_AE, MLP_VAE, PosEncAE
    from utils.DN_vf_tools import convertvf2image # Needed for CNN mapping
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================
def norm(x):
    return 2 * (x - MIN_VAL) / (MAX_VAL - MIN_VAL) - 1

def denorm(x):
    return (x + 1) / 2 * (MAX_VAL - MIN_VAL) + MIN_VAL

def get_model_class(model_type):
    if model_type == 'mlp_ae': return MLP_AE
    if model_type == 'mlp_vae': return MLP_VAE
    if model_type == 'cnn_ae': return CNN_AE
    if model_type == 'cnn_vae': return CNN_VAE
    if model_type == 'posenc': return PosEncAE
    return None

def find_checkpoint(model_type):
    """Finds the final checkpoint for a specific model type."""
    # Pattern matches the naming convention in train_two_stage.py
    # e.g., "mlp_ae_z8_beta0.1_0209-1230_final.pth"
    pattern = f"{model_type}_*_final.pth"
    files = list(WEIGHTS_DIR.glob(pattern))
    
    if not files:
        return None
    # Sort by modification time to get the latest run
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

def prepare_input(td_vector, model_type):
    """
    Converts raw 52-vector into the correct tensor format.
    MLP/PosEnc -> (1, 52)
    CNN        -> (1, 2, 12, 12) [Value, Mask]
    """
    # 1. Normalize Vector
    vec_norm = norm(td_vector)
    
    if 'cnn' in model_type:
        # Convert to 2D Grid
        grid = convertvf2image(td_vector)
        
        # Create Mask (1=Valid, 0=NaN)
        mask = (~np.isnan(grid)).astype(np.float32)
        
        # Fill NaNs for Value Channel
        grid_clean = np.nan_to_num(grid, nan=MIN_VAL)
        val_norm = norm(grid_clean)
        
        # Stack: (2, 12, 12) -> (1, 2, 12, 12)
        input_np = np.stack([val_norm, mask], axis=0)
        tensor = torch.from_numpy(input_np).unsqueeze(0).float()
    else:
        # MLP/PosEnc: Just the vector
        tensor = torch.from_numpy(vec_norm).unsqueeze(0).float()
        
    return tensor.to(DEVICE)

def extract_output(output_tensor, model_type, mask=None):
    """
    Extracts the clean 52-vector from the model output.
    """
    # 1. Unpack Tuple (if VAE)
    if isinstance(output_tensor, tuple):
        recon = output_tensor[0]
    else:
        recon = output_tensor

    recon = recon.detach().cpu()

    # 2. Extract Data
    if 'cnn' in model_type:
        # CNN Output: (1, 2, 12, 12) -> We want Channel 0
        recon_grid = recon.squeeze(0)[0].numpy() # (12, 12)
        
        # Denormalize Grid
        grid_db = denorm(recon_grid)
        
        # Extract valid pixels using the original mask
        # Note: 'mask' here is the 12x12 numpy mask we created during input prep
        mask_bool = mask.astype(bool)
        valid_values = grid_db[mask_bool]
        return valid_values
    else:
        # MLP Output: (1, 52)
        recon_vec = recon.squeeze(0).numpy()
        return denorm(recon_vec)

# ==========================================
# 2. INFERENCE LOOP
# ==========================================
def run_inference(model_type, latent_dim=8):
    ckpt_path = find_checkpoint(model_type)
    if not ckpt_path:
        print(f"Skipping {model_type}: No checkpoint found in {WEIGHTS_DIR}")
        return

    print(f"\nRunning Inference: {model_type.upper()}")
    print(f"   Checkpoint: {ckpt_path.name}")
    
    # 1. Load Model
    ModelClass = get_model_class(model_type)
    # Note: PosEncAE might need extra args if you customized embedding dims, 
    # but defaults (emb=32) usually match training.
    model = ModelClass(latent_dim=latent_dim).to(DEVICE)
    
    # Load Weights
    state_dict = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    
    # 2. Setup Output
    save_dir = OUTPUT_BASE / f"denoised_{model_type}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    files = list(INPUT_DIR.glob("*.npz"))
    
    # 3. Processing Loop
    for f in tqdm(files, desc=model_type):
        try:
            # Load
            data = np.load(f, allow_pickle=True)
            td_raw = data['td'].astype(np.float32)
            
            # Prepare Input
            inp_tensor = prepare_input(td_raw, model_type)
            
            # Save mask for CNN extraction later
            mask_np = None
            if 'cnn' in model_type:
                # Re-generate mask to extract values later
                grid = convertvf2image(td_raw)
                mask_np = (~np.isnan(grid)).astype(np.float32)
            
            # Inference
            with torch.no_grad():
                out = model(inp_tensor)
            
            # Post-Process
            td_denoised = extract_output(out, model_type, mask=mask_np)
            
            # Save
            np.savez_compressed(
                save_dir / f.name,
                td=td_denoised,
                rnflt=data['rnflt'],
                meta=data['meta']
            )
            
        except Exception as e:
            tqdm.write(f"Error on {f.name}: {e}")
            
    print(f"Saved results to: {save_dir}")

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
def main():
    if not INPUT_DIR.exists():
        print(f"Input dir not found: {INPUT_DIR}")
        return

    # Define the models you want to run (must match names in train script)
    # Ensure latent_dim matches what you trained with!
    MODELS_TO_RUN = [
        ('mlp_ae', 8),
        ('mlp_vae', 8),
        ('posenc', 8),
        ('cnn_ae', 8),
        ('cnn_vae', 8)
    ]

    print(f"Reading from: {INPUT_DIR}")
    print(f"Saving to:   {OUTPUT_BASE}")
    
    for m_name, z_dim in MODELS_TO_RUN:
        run_inference(m_name, latent_dim=z_dim)

if __name__ == "__main__":
    main()