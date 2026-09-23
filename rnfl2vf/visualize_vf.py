import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from prediction_VF_struct_fun import EfficientNet_RNFLT, normalize_rnflt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, required=True, help="Path to data folder")
parser.add_argument('--checkpoint', type=str, required=True, help="Path to model checkpoint")
parser.add_argument('--num_samples', type=int, default=9, help="Number of samples to visualize")
args = parser.parse_args()

"""
python visualize_vf_check.py \
    --folder $VF_DATA_ROOT/original_below350 \
    --checkpoint checkpoints/sf_raw_avg.pth \
    --num_samples 9
"""

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIN_V, MAX_V = -38., 26.

def flip_rnfl_image(rnflt):
    """Flip RNFL image horizontally for left eye normalization"""
    return np.fliplr(rnflt).copy()

def denorm_vf(x):
    return (x + 1)/2*(MAX_V - MIN_V) + MIN_V

def norm_vf(x): 
    return 2*(x - MIN_V)/(MAX_V - MIN_V) - 1

def vf_to_grid(vf_52):
    """Convert 52-point VF to 8x9 grid"""
    grid = np.full((8, 9), np.nan)
    indices = [
        (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
        (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
        (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7),
        (3, 0), (3, 1), (3, 2), (3, 3), (3, 5), (3, 6), (3, 7), (3, 8),
        (4, 0), (4, 1), (4, 2), (4, 3), (4, 5), (4, 6), (4, 7), (4, 8),
        (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7),
        (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7),
        (7, 2), (7, 3), (7, 4), (7, 5), (7, 6),
    ]
    for i, (row, col) in enumerate(indices[:len(vf_52)]):
        grid[row, col] = vf_52[i]
    return grid

# Load model
print(f"Loading model from {args.checkpoint}...")
model = EfficientNet_RNFLT().to(DEVICE)
model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
model.eval()

# Load data
all_files = list(Path(args.folder).glob("*.npz"))
sample_files = all_files[:args.num_samples]

# Compute VF average from ALL files (just like the training does)
print("Computing VF average from all files...")
all_vf = []
for file in all_files:
    data = np.load(file, allow_pickle=True)
    all_vf.append(data['td'])
vf_avg = np.mean(np.stack(all_vf, axis=0), axis=0).astype(np.float32)
print(f"VF avg shape: {vf_avg.shape}, mean: {vf_avg.mean():.2f} dB")

# Generate predictions
predictions = []
print("Generating predictions...")
with torch.no_grad():
    for file in sample_files:
        data = np.load(file, allow_pickle=True)
        rnflt = normalize_rnflt(data['rnflt'])

        # Flip RNFL if left eye (matching training alignment)
        fname = file.name
        if "_0_" in fname:
            rnflt = flip_rnfl_image(rnflt)

        rnflt_tensor = torch.tensor(rnflt).unsqueeze(0).unsqueeze(0).to(DEVICE)
        
        pred = model(rnflt_tensor).cpu().numpy().squeeze()
        pred_denorm = denorm_vf(pred)
        predictions.append(pred_denorm)

# Visualize
fig, axes = plt.subplots(args.num_samples, 2, figsize=(8, 3*args.num_samples))
if args.num_samples == 1:
    axes = axes.reshape(1, -1)

for i in range(args.num_samples):
    avg_grid = vf_to_grid(vf_avg)
    pred_grid = vf_to_grid(predictions[i])
    
    # Column 1: VF Average (Target - what model should predict)
    im0 = axes[i, 0].imshow(avg_grid, cmap='RdYlGn', vmin=MIN_V, vmax=MAX_V)
    axes[i, 0].set_title(f"Sample {i+1}: VF Avg (Target)\nMean: {vf_avg.mean():.2f} dB")
    axes[i, 0].axis('off')
    
    # Column 2: Predicted VF (what model actually predicted)
    im1 = axes[i, 1].imshow(pred_grid, cmap='RdYlGn', vmin=MIN_V, vmax=MAX_V)
    mae = np.mean(np.abs(predictions[i] - vf_avg))
    axes[i, 1].set_title(f"Predicted VF\nMean: {predictions[i].mean():.2f} dB, MAE: {mae:.2f}")
    axes[i, 1].axis('off')

# Add colorbar
fig.colorbar(im0, ax=axes.ravel().tolist(), label='VF Value (dB)', shrink=0.6)
plt.suptitle("VF Average (Target) vs Model Prediction", fontsize=16, y=1.00)
plt.tight_layout()

output_name = f"verify_r2_{Path(args.checkpoint).stem}.png"
plt.savefig(output_name, dpi=150, bbox_inches='tight')
print(f"\nVisualization saved to {output_name}")

# Compute overall metrics
all_preds = np.vstack(predictions)
all_targets = np.tile(vf_avg, (args.num_samples, 1))

from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr

r2 = r2_score(all_targets.flatten(), all_preds.flatten())
pearson = pearsonr(all_targets.flatten(), all_preds.flatten())[0]
mae = mean_absolute_error(all_targets.flatten(), all_preds.flatten())

print(f"\nMetrics on {args.num_samples} samples:")
print(f"R²:      {r2:.4f}")
print(f"Pearson: {pearson:.4f}")
print(f"MAE:     {mae:.4f} dB")
print(f"\nSince target is constant (avg VF), R² should be close to 0 or negative")
print(f"   if model learns to predict the mean well, R² → 0")