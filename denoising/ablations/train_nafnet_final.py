import os
import sys
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from tqdm.auto import tqdm
import wandb

# ==========================================
# 1. Configuration
# ==========================================
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")   # override with CUDA_VISIBLE_DEVICES=<id>
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths (see config/env.example.sh)
CODE_DIR = Path(__file__).resolve().parents[1]              # denoising/
BASE_DIR = Path(os.environ.get("VF_DENOISE_ROOT", CODE_DIR))
sys.path.append(str(CODE_DIR))
sys.path.append(str(CODE_DIR / "utils"))
sys.path.append(str(CODE_DIR / "model"))

# Dataset Paths
TRAIN_PATH = BASE_DIR / "dataset_train_val_test/VAE/fine_tune_dataset_Nge1_train.npz"
VAL_PATH   = BASE_DIR / "dataset_train_val_test/VAE/fine_tune_dataset_Nge1_val.npz"
SAVE_DIR   = BASE_DIR / "model_weight/trained_NAFNet_Fixed"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 32
LR         = 2e-4
EPOCHS     = 10  # short diagnostic run; the full schedule is 150 epochs
MIN_VAL    = -38.0
MAX_VAL    = 26.0
SEED       = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

try:
    from model.NAFNet import NAFNet 
    from DN_vf_tools import convertvf2image
except ImportError as e:
    print(f"Error: {e}")
    sys.exit(1)

# ==========================================
# LOSS: static spatial weighting only, without the severity term
# ==========================================
def get_spatial_weight_matrix():
    """Static spatial weights (Center > Periphery)"""
    w = np.zeros([12, 12])
    row_weights = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0],
        [0, 0, 0, 0.5, 0, 0, 0, 0, 0.5, 0, 0, 0],
        [0, 0, 0.5, 0, 0, 2, 2, 0, 0, 0.5, 0, 0],
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

def simple_weighted_mse(pred, target):
    """Weighted MSE using the static spatial weights only (no severity term)."""
    loss = torch.mean(STATIC_WEIGHT * (pred - target) ** 2)
    return loss

def mse_loss(pred, target):
    """Plain MSE - simplest baseline"""
    return torch.mean((pred - target) ** 2)

# ==========================================
# Dataset
# ==========================================
class SimpleNAFDataset(Dataset):
    def __init__(self, npz_file, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val
        
        d = np.load(npz_file, allow_pickle=True)
        self.x = d["td"].astype(np.float32)         
        self.y = d["td_target"].astype(np.float32)
        
        print(f"Loaded {len(self.x)} samples")
        
        # DIAGNOSTIC: Check noise level
        noise_levels = []
        for i in range(min(1000, len(self.x))):
            diff = np.abs(self.x[i] - self.y[i])
            noise_levels.append(np.mean(diff))
        
        avg_noise = np.mean(noise_levels)
        print(f"Average noise in data: {avg_noise:.3f} dB")
        
        if avg_noise < 0.5:
            print("WARNING: very little noise detected in the training pairs.")
        else:
            print("Noise level is adequate for training.")

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
# Loaders
# ==========================================
print(f"Loading Train: {TRAIN_PATH}")
train_ds = SimpleNAFDataset(TRAIN_PATH, MIN_VAL, MAX_VAL)
print(f"Loading Val:   {VAL_PATH}")
val_ds   = SimpleNAFDataset(VAL_PATH, MIN_VAL, MAX_VAL)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ==========================================
# Training Loop with DIAGNOSTICS
# ==========================================
def test_on_samples(model, dataset, num_samples=10):
    """Test model on specific samples to see actual denoising"""
    model.eval()
    changes = []
    
    print("\nTesting on sample data:")
    for i in range(num_samples):
        x, y = dataset[i]
        x = x.unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            pred = model(x)
        
        # Denormalize
        pred_np = pred.squeeze().cpu().numpy()
        y_np = y.squeeze().numpy()
        
        pred_db = (pred_np + 1) / 2 * (MAX_VAL - MIN_VAL) + MIN_VAL
        y_db = (y_np + 1) / 2 * (MAX_VAL - MIN_VAL) + MIN_VAL
        
        # Calculate change
        diff = np.abs(pred_db - y_db)
        mean_change = np.mean(diff)
        changes.append(mean_change)
    
    avg_change = np.mean(changes)
    print(f"   Average error vs ground truth: {avg_change:.4f} dB")
    return avg_change

def main():
    # Test both loss functions
    for loss_type in ['plain_mse', 'weighted_mse']:
        
        run_name = f"NAFNet_QuickTest_{loss_type}_width32"
        print(f"\n{'='*70}")
        print(f"QUICK TEST: {run_name}")
        print(f"{'='*70}")
        
        wandb.init(project="Denoising_NAFNet", name=run_name, 
                   config={"lr": LR, "model": "NAFNet_Fixed", "loss": loss_type, "epochs": EPOCHS})
        
        # Model
        model = NAFNet(img_channel=1, width=32, middle_blk_num=2,
                       enc_blk_nums=[2, 2, 2], dec_blk_nums=[2, 2, 2]).to(DEVICE)
        
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        
        print(f"   Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
        
        # Test BEFORE training
        print("\nBEFORE TRAINING:")
        initial_error = test_on_samples(model, val_ds, num_samples=20)
        
        best_val_loss = float('inf')
        
        # Training
        for ep in range(1, EPOCHS+1):
            model.train()
            tr_loss = 0
            
            for x, y in tqdm(train_dl, desc=f"Ep {ep}/{EPOCHS}", leave=False):
                x, y = x.to(DEVICE), y.to(DEVICE)
                
                opt.zero_grad()
                pred = model(x)
                
                if loss_type == 'plain_mse':
                    loss = mse_loss(pred, y)
                else:
                    loss = simple_weighted_mse(pred, y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                    if loss_type == 'plain_mse':
                        val_loss += mse_loss(pred, y).item()
                    else:
                        val_loss += simple_weighted_mse(pred, y).item()
            val_loss /= len(val_dl)
            
            scheduler.step()
            
            print(f"Ep {ep:2d}: Train {tr_loss:.6f} | Val {val_loss:.6f} | LR {scheduler.get_last_lr()[0]:.6f}")
            wandb.log({"train_loss": tr_loss, "val_loss": val_loss, "lr": scheduler.get_last_lr()[0], "epoch": ep})
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), SAVE_DIR / f"{run_name}_best.pth")
        
        # Test AFTER training
        print("\nAFTER TRAINING:")
        final_error = test_on_samples(model, val_ds, num_samples=20)
        
        improvement = initial_error - final_error
        improvement_pct = (improvement / initial_error) * 100
        
        print(f"\n{'='*70}")
        print(f"LEARNING SUMMARY for {loss_type}:")
        print(f"   Initial error:  {initial_error:.4f} dB")
        print(f"   Final error:    {final_error:.4f} dB")
        print(f"   Improvement:    {improvement:.4f} dB ({improvement_pct:.1f}%)")
        
        if improvement < 0:
            print("   Error increased during training.")
        elif improvement < 0.5:
            print("   Marginal improvement; check the loss function and training pairs.")
        elif improvement < 2.0:
            print("   Moderate improvement; consider the full 150-epoch schedule.")
        else:
            print("   Substantial improvement.")
        
        print(f"{'='*70}\n")
        
        wandb.log({
            "initial_error": initial_error,
            "final_error": final_error,
            "improvement": improvement,
            "improvement_pct": improvement_pct
        })
        
        torch.save(model.state_dict(), SAVE_DIR / f"{run_name}_final.pth")
        wandb.finish()
    
    print("\nQuick test complete!")
    print("   Check results above to decide which loss function to use for full training.")

if __name__ == "__main__":
    main()