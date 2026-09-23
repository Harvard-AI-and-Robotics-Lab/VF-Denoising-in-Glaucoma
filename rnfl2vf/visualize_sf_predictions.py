import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from prediction_VF_struct_fun import EfficientNet_RNFLT, normalize_rnflt
import random
from tqdm.auto import tqdm
import argparse

# ==========================================
# Configuration
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, required=True, help="Path to denoised .npz files")
parser.add_argument('--checkpoint', type=str, required=True, help="Path to trained model checkpoint")
parser.add_argument('--output_name', type=str, required=True, help="Name for output (e.g., 'raw', 'n2n')")
parser.add_argument('--num_samples', type=int, default=3, help="Number of samples to visualize")
args = parser.parse_args()

OUTPUT_DIR = Path(f"sf_visualizations_{args.output_name}")
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIN_V, MAX_V = -38., 26.
VF_VMIN, VF_VMAX = -38, 26

# ==========================================
# Helpers
# ==========================================
def flip_rnfl_image(rnflt):
    """Flip RNFL image horizontally for left eye normalization"""
    return np.fliplr(rnflt).copy()

def denorm_vf(x):
    """Model outputs in [-1,1] -> VF dB in [MIN_V, MAX_V]."""
    return (x + 1) / 2 * (MAX_V - MIN_V) + MIN_V

def convertvf2image(vec):
    """Convert 52-point VF vector to 12x12 grid (24-2 pattern)"""
    img = np.full((12, 12), np.nan)
    if len(vec) == 52:
        # Insert blind spot positions
        vec = np.insert(vec, 25, np.nan)  
        vec = np.insert(vec, 34, np.nan)
        mask = [
            (0,3),(0,4),(0,5),(0,6),
            (1,2),(1,3),(1,4),(1,5),(1,6),(1,7),
            (2,1),(2,2),(2,3),(2,4),(2,5),(2,6),(2,7),(2,8),
            (3,0),(3,1),(3,2),(3,3),(3,4),(3,5),(3,6),(3,7),(3,8), 
            (4,0),(4,1),(4,2),(4,3),(4,4),(4,5),(4,6),(4,7),(4,8),
            (5,1),(5,2),(5,3),(5,4),(5,5),(5,6),(5,7),(5,8),
            (6,2),(6,3),(6,4),(6,5),(6,6),(6,7),
            (7,3),(7,4),(7,5),(7,6)
        ]
        for i, (r, c) in enumerate(mask):
            if i < len(vec): 
                img[r, c] = vec[i]
    return img

def plot_vf(ax, data_vec, title, vmin, vmax, show_values=True):
    """Plot VF map on axis"""
    if data_vec.ndim == 1:
        img = convertvf2image(data_vec)
    else:
        img = data_vec
    
    cmap = plt.cm.get_cmap('gray').copy()
    cmap.set_bad(color='white')
    
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.axis('off')
    
    if show_values:
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                val = img[i, j]
                if not np.isnan(val):
                    color = 'white' if val < -10 else 'black'
                    ax.text(j, i, f"{val:.0f}", ha='center', va='center', 
                            color=color, fontsize=7, fontweight='bold')
    return im

def plot_rnflt(ax, rnflt_data, title):
    """Plot RNFLT thickness map"""
    # RNFLT is typically (768, 512) - transpose for better visualization
    if rnflt_data.shape[0] > rnflt_data.shape[1]:
        rnflt_data = rnflt_data.T
    
    im = ax.imshow(rnflt_data, cmap='viridis', aspect='auto')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.axis('off')
    return im

# ==========================================
# Main
# ==========================================
def main():
    print(f"Loading model from {args.checkpoint}...")
    model = EfficientNet_RNFLT().to(DEVICE)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    model.eval()
    
    # Get all files
    all_files = list(Path(args.folder).glob("*.npz"))
    print(f"Found {len(all_files)} files")
    
    # Randomly sample
    selected_files = random.sample(all_files, min(args.num_samples, len(all_files)))
    
    print(f"Generating visualizations for {len(selected_files)} samples...")
    
    with torch.no_grad():
        for file_idx, file_path in enumerate(tqdm(selected_files, desc="Visualizing")):
            # Load data
            data = np.load(file_path, allow_pickle=True)
            
            # RNFLT - flip left eye for orientation alignment (matching training)
            rnflt_raw = data['rnflt']
            fname = file_path.name
            is_left_eye = "_0_" in fname
            if is_left_eye:
                rnflt_raw = flip_rnfl_image(rnflt_raw)
            rnflt_norm = normalize_rnflt(rnflt_raw)
            rnflt_tensor = torch.tensor(rnflt_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)
            
            # Predict VF
            pred = model(rnflt_tensor).cpu().numpy().squeeze()
            pred_denorm = denorm_vf(pred).astype(np.float32)
            
            # Ground truth VF
            vf_gt = data['td'].astype(np.float32)
            if vf_gt.ndim > 1:
                vf_gt = vf_gt.squeeze() if vf_gt.shape[0] == 1 else np.mean(vf_gt, axis=0)
            vf_gt = vf_gt.astype(np.float32)
            
            # Averaged VF baseline (constant = mean of GT)
            vf_avg = np.full_like(vf_gt, np.mean(vf_gt))
            
            # ==========================================
            # Create Figure: 1 row, 4 columns
            # ==========================================
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            patient_id = file_path.stem
            fig.suptitle(f"Patient: {patient_id} | Model: {args.output_name.upper()}", 
                        fontsize=14, fontweight='bold')
            
            # Column 1: RNFLT Input
            plot_rnflt(axes[0], rnflt_raw, "RNFLT Input")
            
            # Column 2: Predicted VF
            plot_vf(axes[1], pred_denorm, "Predicted VF", VF_VMIN, VF_VMAX, show_values=True)
            
            # Column 3: Ground Truth VF
            plot_vf(axes[2], vf_gt, "Ground Truth VF", VF_VMIN, VF_VMAX, show_values=True)
            
            # Column 4: Averaged VF Baseline
            im = plot_vf(axes[3], vf_avg, "Averaged VF Baseline", VF_VMIN, VF_VMAX, show_values=True)
            
            # Add colorbar for VF maps
            cbar_ax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
            fig.colorbar(im, cax=cbar_ax, label="VF Sensitivity (dB)")
            
            plt.tight_layout(rect=[0, 0, 0.9, 0.96])
            
            # Save
            save_path = OUTPUT_DIR / f"final_{file_idx+1}_{patient_id}.png"
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close()
    
    print(f"Visualizations saved to: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()