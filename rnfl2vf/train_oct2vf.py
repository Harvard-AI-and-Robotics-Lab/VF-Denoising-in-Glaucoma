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
        # 1. Check Filename for Left Eye (_0_)
        fname = os.path.basename(self.paths[idx])
        is_left_eye = "_0_" in fname

        data = np.load(self.paths[idx], allow_pickle=True)
        rnflt_norm = normalize_rnflt(data['rnflt'])

        # 2. Flip RNFL if Left Eye (CRITICAL for consistency)
        if is_left_eye:
            rnflt_norm = flip_rnfl_image(rnflt_norm)

        return torch.tensor(rnflt_norm).unsqueeze(0), torch.tensor(norm_vf(data['td'].astype(np.float32)))

all_files = list(DATA_DIR.glob("*.npz"))
train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)

loader_args = {'batch_size': BATCH_SIZE, 'num_workers': NUM_WORKERS, 'pin_memory': True, 'persistent_workers': True}
train_loader = DataLoader(SF_Training_Dataset(train_files), shuffle=True, **loader_args)
val_loader = DataLoader(SF_Training_Dataset(val_files), shuffle=False, **loader_args)

# --- 3. TRAINING WITH LOGGING ---
model = EfficientNet_RNFLT().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.L1Loss()

best_r2 = -np.inf

# Open log file once and append results
with open(LOG_FILE, "w") as f:
    f.write("Epoch,Train_Loss,Val_R2,Val_Pearson,Val_MAE\n") # CSV-style header

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
    r2, p_r, mae = r2_score(y_true, y_pred), pearsonr(y_true, y_pred)[0], mean_absolute_error(y_true, y_pred)
    avg_loss = train_loss/len(train_loader)

    # Save to TXT
    with open(LOG_FILE, "a") as f:
        f.write(f"{epoch},{avg_loss:.4f},{r2:.4f},{p_r:.4f},{mae:.4f}\n")

    print(f"Ep {epoch} | Loss: {avg_loss:.4f} | R2: {r2:.4f} | Pearson: {p_r:.4f}")

    # Checkpoint Best Model
    if r2 > best_r2:
        best_r2 = r2
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"New Best R²! Saved to {SAVE_PATH}")

print(f"\nTraining Complete. Best R²: {best_r2:.4f}. Logs saved to {LOG_FILE}")