import torch
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
from tqdm import tqdm
import argparse
import pandas as pd

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")   # override with CUDA_VISIBLE_DEVICES=<id>
from prediction_VF_struct_fun import EfficientNet_RNFLT, normalize_rnflt

# -------------------------
# ARGS
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, required=True, help="Path to denoised .npz files")
parser.add_argument('--checkpoint', type=str, required=True, help="Path to avg-VF trained model checkpoint")
parser.add_argument('--baseline_checkpoint', type=str, required=True, help="Path to original-VF trained model checkpoint")
parser.add_argument('--output_name', type=str, required=True, help="Name for output results")
args = parser.parse_args()

# -------------------------
# CONFIG
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIN_V, MAX_V = -38., 26.

def denorm_vf(x):
    """Model outputs in [-1,1] -> VF dB in [MIN_V, MAX_V]."""
    return (x + 1) / 2 * (MAX_V - MIN_V) + MIN_V

def flip_rnfl_image(rnflt):
    """Flip RNFL image horizontally for left eye normalization"""
    return np.fliplr(rnflt).copy()

def safe_pearson(x, y):
    """Pearson can be nan if x or y has ~0 variance."""
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return np.nan
    return pearsonr(x, y)[0]

# -------------------------
# LOAD FILES / SPLIT (same as training)
# -------------------------
all_files = list(Path(args.folder).glob("*.npz"))
train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)
print(f"Validation set size: {len(val_files)} files")

# -------------------------
# LOAD MODELS
# -------------------------
print(f"Loading avg-VF model from {args.checkpoint}...")
model_avg = EfficientNet_RNFLT().to(DEVICE)
model_avg.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
model_avg.eval()

print(f"Loading original-VF baseline model from {args.baseline_checkpoint}...")
model_orig = EfficientNet_RNFLT().to(DEVICE)
model_orig.load_state_dict(torch.load(args.baseline_checkpoint, map_location=DEVICE))
model_orig.eval()

# -------------------------
# EVAL
# -------------------------
# avg-VF model metrics (only MAE/RMSE make sense)
avg_mae_list, avg_rmse_list = [], []
avg_preds_global, avg_targets_global = [], []

# baseline (original-VF) model metrics (full metrics)
base_r2_list, base_pearson_list, base_mae_list, base_rmse_list = [], [], [], []
base_preds_global, base_targets_global = [], []

with torch.no_grad():
    for file in tqdm(val_files, desc="Inference"):
        fname = os.path.basename(file)
        is_left_eye = "_0_" in fname

        data = np.load(file, allow_pickle=True)

        # RNFLT -> normalize + flip left eyes
        rnflt = normalize_rnflt(data['rnflt'])
        if is_left_eye:
            rnflt = flip_rnfl_image(rnflt)

        rnflt_tensor = torch.tensor(rnflt).unsqueeze(0).unsqueeze(0).to(DEVICE)

        # Load original VF
        vf_orig = data['td'].astype(np.float32)
        if vf_orig.ndim > 1:
            vf_orig = vf_orig.squeeze() if vf_orig.shape[0] == 1 else np.mean(vf_orig, axis=0)
        vf_orig = vf_orig.astype(np.float32)

        # Create averaged VF ground truth (constant = mean of original VF)
        vf_avg = np.full_like(vf_orig, np.mean(vf_orig))

        # --- avg-VF model predictions (compare to AVERAGED ground truth) ---
        pred_avg = model_avg(rnflt_tensor).cpu().numpy().squeeze()
        pred_avg_denorm = denorm_vf(pred_avg).astype(np.float32)

        avg_mae_list.append(mean_absolute_error(vf_avg, pred_avg_denorm))
        avg_rmse_list.append(mean_squared_error(vf_avg, pred_avg_denorm, squared=False))
        avg_preds_global.append(pred_avg_denorm)
        avg_targets_global.append(vf_avg)

        # --- baseline (original-VF) model predictions (compare to ORIGINAL ground truth) ---
        pred_base = model_orig(rnflt_tensor).cpu().numpy().squeeze()
        pred_base_denorm = denorm_vf(pred_base).astype(np.float32)

        base_r2_list.append(r2_score(vf_orig, pred_base_denorm))
        base_pearson_list.append(safe_pearson(vf_orig, pred_base_denorm))
        base_mae_list.append(mean_absolute_error(vf_orig, pred_base_denorm))
        base_rmse_list.append(mean_squared_error(vf_orig, pred_base_denorm, squared=False))
        base_preds_global.append(pred_base_denorm)
        base_targets_global.append(vf_orig)

# -------------------------
# AGGREGATE (mean +/- std)
# -------------------------
def mean_std(x):
    x = np.array(x, dtype=np.float64)
    return float(np.nanmean(x)), float(np.nanstd(x))

def fmt(vals):
    m, s = mean_std(vals)
    return f"{m:.4f} +/- {s:.4f}"

# Global R2 for baseline only
base_r2_global = r2_score(np.concatenate(base_targets_global), np.concatenate(base_preds_global))

# -------------------------
# SAVE RESULTS
# -------------------------
results = {
    'Metric': ['R2 (per-sample avg)', 'R2 (global)', 'Pearson', 'MAE (dB)', 'RMSE (dB)'],
    'Baseline_OrigVF_Model': [
        fmt(base_r2_list),
        f"{base_r2_global:.4f}",
        fmt(base_pearson_list),
        fmt(base_mae_list),
        fmt(base_rmse_list)
    ],
    'AvgVF_Model': [
        'N/A (constant target)',
        'N/A (constant target)',
        'N/A (constant target)',
        fmt(avg_mae_list),
        fmt(avg_rmse_list)
    ]
}

df = pd.DataFrame(results)
output_dir = Path("eval_results")
output_dir.mkdir(exist_ok=True)
output_file = output_dir / f"eval_avg_{args.output_name}.csv"
df.to_csv(output_file, index=False)

print("\n" + "=" * 70)
print("EVALUATION RESULTS")
print("=" * 70)
print(df.to_string(index=False))
print("=" * 70)
print(f"\nResults saved to {output_file}")

print("\nNotes:")
print(" - Baseline_OrigVF_Model: trained on full 52-point VF → evaluated vs ORIGINAL VF")
print(" - AvgVF_Model: trained on averaged VF (constant per sample) → evaluated vs AVERAGED VF")
print(" - R²/Pearson N/A for AvgVF_Model because both prediction and target are constants (zero variance)")
print(" - Only MAE/RMSE are meaningful for constant targets")
print(" - Left-eye RNFL images were flipped to match training preprocessing.")