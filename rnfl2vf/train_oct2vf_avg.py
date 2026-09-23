import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np, os, json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
from tqdm import tqdm

# --- GPU SELECTION ---
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")   # override with CUDA_VISIBLE_DEVICES=<id>
from prediction_VF_struct_fun import EfficientNet_RNFLT, normalize_rnflt

import argparse

def flip_rnfl_image(rnflt):
    """Flip RNFL image horizontally for left eye normalization"""
    return np.fliplr(rnflt).copy()

# --- 1. DYNAMIC CONFIGURATION ---
parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, required=True, help="Path to denoised .npz files")
parser.add_argument('--run_name', type=str, required=True, help="Name for checkpoints and logs")
args = parser.parse_args()

DATA_DIR = Path(args.folder)
LOG_FILE = Path(f"logs/{args.run_name}.txt")
SAVE_PATH = Path(f"checkpoints/{args.run_name}.pth")
Path("logs").mkdir(exist_ok=True)
Path("checkpoints").mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS, LR, BATCH_SIZE = 35, 1e-4, 32
MIN_V, MAX_V = -38., 26.
NUM_WORKERS = 8

def denorm_vf(x): return (x + 1)/2*(MAX_V - MIN_V) + MIN_V
def norm_vf(x): return 2*(x - MIN_V)/(MAX_V - MIN_V) - 1

# --- 2. DATASET & LOADERS ---
class SF_Training_Dataset(Dataset):
    def __init__(self, paths): self.paths = paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        fname = os.path.basename(self.paths[idx])
        is_left_eye = "_0_" in fname

        data = np.load(self.paths[idx], allow_pickle=True)
        rnflt_norm = normalize_rnflt(data['rnflt'])

        if is_left_eye:
            rnflt_norm = flip_rnfl_image(rnflt_norm)

        # Target: averaged VF (constant vector of per-sample mean)
        vf = data['td'].astype(np.float32)
        vf_avg = np.full_like(vf, np.mean(vf))

        return torch.tensor(rnflt_norm).unsqueeze(0), torch.tensor(norm_vf(vf_avg))

all_files = list(DATA_DIR.glob("*.npz"))
train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)

loader_args = {'batch_size': BATCH_SIZE, 'num_workers': NUM_WORKERS, 'pin_memory': True, 'persistent_workers': True}
train_loader = DataLoader(SF_Training_Dataset(train_files), shuffle=True, **loader_args)
val_loader = DataLoader(SF_Training_Dataset(val_files), shuffle=False, **loader_args)

# --- 3. TRAINING WITH LOGGING ---
model = EfficientNet_RNFLT().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.L1Loss()

best_mae = np.inf

with open(LOG_FILE, "w") as f:
    f.write("Epoch,Train_Loss,Val_MAE,Val_RMSE\n")

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0
    for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward(); optimizer.step()
        train_loss += loss.item()

    # VALIDATION
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in val_loader:
            pred = model(x.to(DEVICE, non_blocking=True))
            all_preds.append(denorm_vf(pred.cpu().numpy()))
            all_targets.append(denorm_vf(y.numpy()))

    y_pred, y_true = np.vstack(all_preds).flatten(), np.vstack(all_targets).flatten()
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    avg_loss = train_loss/len(train_loader)

    with open(LOG_FILE, "a") as f:
        f.write(f"{epoch},{avg_loss:.4f},{mae:.4f},{rmse:.4f}\n")

    print(f"Ep {epoch} | Loss: {avg_loss:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f}")

    if mae < best_mae:
        best_mae = mae
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"New Best MAE! Saved to {SAVE_PATH}")

print(f"\nTraining Complete. Best MAE: {best_mae:.4f}. Logs saved to {LOG_FILE}")