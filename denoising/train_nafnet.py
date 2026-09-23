import os
import sys
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from pathlib import Path
from tqdm.auto import tqdm
import wandb

# ==========================================
# 1. Configuration
# ==========================================
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")   # override with CUDA_VISIBLE_DEVICES=<id>
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths (see config/env.example.sh)
CODE_DIR = Path(__file__).resolve().parent                 # this repository
BASE_DIR = Path(os.environ.get("VF_DENOISE_ROOT", CODE_DIR))  # datasets + weights
sys.path.append(str(CODE_DIR))
sys.path.append(str(CODE_DIR / "utils"))
sys.path.append(str(CODE_DIR / "model"))

# Dataset Paths
TRAIN_PATH = BASE_DIR / "dataset_train_val_test/MLP/fine_tune_dataset_N3_train.npz"
VAL_PATH   = BASE_DIR / "dataset_train_val_test/MLP/fine_tune_dataset_N3_val.npz"
SAVE_DIR   = BASE_DIR / "model_weight/trained_NAFNet_Advanced"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 64
LR         = 1e-4
EPOCHS     = 100
MIN_VAL    = -38.0
MAX_VAL    = 26.0
SEED       = 42

# Reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ==========================================
# 2. Imports
# ==========================================
try:
    from model.NAFNet import NAFNet 
    from DN_vf_tools import convertvf2image
except ImportError as e:
    print(f"Error: {e}")
    sys.exit(1)
    
# ==========================================
# 3. Advanced Loss Logic
# ==========================================
def get_spatial_weight_matrix():
    """Static spatial weights (Center > Periphery)"""
    w = np.zeros([12, 12])
    row_weights = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0],
        [0, 0, 0, 0.5, 0, 0, 0, 0, 0.5, 0, 0, 0],
        [0, 0, 0.5, 0, 0, 2, 2, 0, 0, 0.5, 0, 0], # High priority center
        [0, 0.5, 0, 0, 2, 2, 2, 2, 0, 0.5, 0, 0],
        [0, 0.5, 0, 0, 2, 2, 2, 2, 0, 0.5, 0, 0],
        [0, 0, 0.5, 0, 0, 2, 2, 0, 0, 0.5, 0, 0],
        [0, 0, 0, 0.5, 0, 0, 0, 0, 0.5, 0, 0, 0],
        [0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    w = np.array(row_weights)
    w[w == 0] = 1.0 
    return torch.FloatTensor(w).unsqueeze(0).unsqueeze(0).to(DEVICE)

STATIC_WEIGHT = get_spatial_weight_matrix()

def advanced_loss(pred, target):
    """
    Weighted MSE = Static_Spatial * Dynamic_Severity * MSE
    """
    # Dynamic Severity: Sigmoid(-Target). 
    # If Target is -30 (Blind), -Target is +30, Sigmoid is 1.0 (High weight).
    # If Target is +25 (Healthy), -Target is -25, Sigmoid is 0.0 (Low weight).
    
    # We use the target (clean data) to determine the weight importance
    severity_weight = torch.sigmoid(-target) 
    
    # Combine weights (Clip min weight to 0.1 so healthy spots aren't ignored)
    combined_weight = STATIC_WEIGHT * torch.clamp(severity_weight, min=0.1)
    
    # MSE
    loss = torch.mean(combined_weight * (pred - target) ** 2)
    return loss

# ==========================================
# 4. Dataset with Balancing
# ==========================================
class AdvancedNAFDataset(Dataset):
    def __init__(self, npz_file, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val
        
        d = np.load(npz_file, allow_pickle=True)
        self.x = d["td"].astype(np.float32)         
        self.y = d["td_target"].astype(np.float32)
        
        # --- Pre-calculate Labels for Balancing ---
        self.labels = []
        print("Calculating dataset balance...")
        for target in self.y:
            mean_val = np.mean(target)
            # Threshold: If mean MD < -5 dB, treat as 'Diseased' (Label 1)
            # This ensures we sample severe cases more often
            if mean_val < -5.0: 
                self.labels.append(1) 
            else:
                self.labels.append(0)
        self.labels = np.array(self.labels)
        
        pos = np.sum(self.labels == 1)
        neg = np.sum(self.labels == 0)
        print(f"Balance: {neg} Healthy (MD > -5) | {pos} Diseased (MD < -5)")
        
        # Calculate Sample Weights for WeightedRandomSampler
        # weight = 1 / frequency
        if pos > 0 and neg > 0:
            weight_neg = 1.0 / neg
            weight_pos = 1.0 / pos
            self.sample_weights = torch.from_numpy(
                np.array([weight_pos if t == 1 else weight_neg for t in self.labels])
            ).float()
        else:
            # Fallback if dataset is too small/skewed
            self.sample_weights = torch.ones(len(self.labels)).float()

    def __len__(self):
        return len(self.x)
    
    def to_img(self, vec):
        grid = convertvf2image(vec)
        grid = np.nan_to_num(grid, nan=self.min_val)
        norm = 2 * (grid - self.min_val) / (self.max_val - self.min_val) - 1
        return norm.astype(np.float32)

    def __getitem__(self, idx):
        img_in = self.to_img(self.x[idx])
        img_gt = self.to_img(self.y[idx])
        return torch.from_numpy(img_in).unsqueeze(0), torch.from_numpy(img_gt).unsqueeze(0)

# ==========================================
# 5. Loaders
# ==========================================
print(f"Loading Train: {TRAIN_PATH}")
train_ds = AdvancedNAFDataset(TRAIN_PATH, MIN_VAL, MAX_VAL)
print(f"Loading Val:   {VAL_PATH}")
val_ds   = AdvancedNAFDataset(VAL_PATH, MIN_VAL, MAX_VAL)

# SAMPLER: Forces 50/50 mix of Healthy/Diseased in every batch
sampler = WeightedRandomSampler(train_ds.sample_weights, len(train_ds.sample_weights))

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4)
val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ==========================================
# 6. Training Loop
# ==========================================
def main():
    run_name = f"NAFNet_Adv_N3_lr{LR}"
    wandb.init(project="Denoising_NAFNet", name=run_name, config={"lr": LR, "model": "NAFNet_Adv"})
    
    # Init Model (Same architecture, smarter training)
    model = NAFNet(img_channel=1, width=16, middle_blk_num=1, 
                   enc_blk_nums=[1, 1, 1], dec_blk_nums=[1, 1, 1]).to(DEVICE)
    
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    
    print(f"Training {run_name}...")
    
    for ep in range(1, EPOCHS+1):
        model.train()
        tr_loss = 0
        
        for x, y in tqdm(train_dl, desc=f"Ep {ep}", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            opt.zero_grad()
            pred = model(x)
            
            # Use Advanced Loss (Spatial + Severity)
            loss = advanced_loss(pred, y)
            
            loss.backward()
            opt.step()
            tr_loss += loss.item()
            
        tr_loss /= len(train_dl)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x)
                # Validation uses same advanced loss to track severity preservation
                val_loss += advanced_loss(pred, y).item()
        val_loss /= len(val_dl)
        
        print(f"Ep {ep}: Train {tr_loss:.4f} | Val {val_loss:.4f}")
        wandb.log({"train_loss": tr_loss, "val_loss": val_loss})
        
    # Save
    torch.save(model.state_dict(), SAVE_DIR / f"{run_name}.pth")
    print(f"Saved model to {SAVE_DIR}")
    wandb.finish()

if __name__ == "__main__":
    main()