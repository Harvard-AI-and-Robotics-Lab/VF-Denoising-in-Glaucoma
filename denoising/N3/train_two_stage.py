import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb
from datetime import datetime

# ==============================================================================
# 0. IMPORTS & UTILS
# ==============================================================================
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))   # denoising/N3
sys.path.append(_CODE_DIR)
sys.path.append(os.path.dirname(_CODE_DIR))              # denoising/ -> provides utils/

try:
    from models import CNN_AE, CNN_VAE, MLP_AE, MLP_VAE, PosEncAE
    from utils.seed_everything import seed_everything
    from utils.DN_vf_tools import convertvf2image 
except ImportError as e:
    print(f"Import Error: {e}")
    print("Ensure 'models.py', 'utils/seed_everything.py', and 'utils/DN_vf_tools.py' exist.")
    sys.exit(1)

# ==============================================================================
# 1. DATASET
# ==============================================================================
class TwoStageDataset(Dataset):
    def __init__(self, npz_file, mode="unsupervised", data_type="mlp", min_val=-38.0, max_val=26.0):
        self.data_type = data_type
        self.mode = mode
        self.min_val = min_val
        self.max_val = max_val

        data = np.load(npz_file, allow_pickle=True)
        
        # Load Input (CNN=2D, MLP=1D)
        if self.data_type == 'cnn':
            self.inputs = data["cnn"].astype(np.float32) # (N, 2, 12, 12)
        else:
            self.inputs = data["td"].astype(np.float32)  # (N, 52)

        # Load Target (Clean) if finetuning
        if self.mode == 'finetune':
            if "td_target" in data:
                self.targets_raw = data["td_target"].astype(np.float32)
            else:
                self.targets_raw = self.inputs 

    def __len__(self):
        return len(self.inputs)

    def norm(self, x):
        return 2 * (x - self.min_val) / (self.max_val - self.min_val) - 1

    def __getitem__(self, idx):
        # 1. Prepare INPUT
        if self.data_type == 'cnn':
            raw_in = self.inputs[idx]
            val = self.norm(raw_in[0])
            mask = raw_in[1]
            x_in = np.stack([val, mask], axis=0)
        else:
            x_in = self.norm(self.inputs[idx])

        # 2. Prepare TARGET
        if self.mode == 'unsupervised':
            x_target = x_in.copy() # Autoencoder: Target = Input
        else:
            # Finetune: Target = Clean N3 Mean
            t_raw = self.targets_raw[idx]
            
            if self.data_type == 'cnn':
                mask = self.inputs[idx][1]
                grid = convertvf2image(t_raw) 
                grid = np.nan_to_num(grid, nan=self.min_val)
                target_norm = self.norm(grid)
                x_target = np.stack([target_norm, mask], axis=0)
            else:
                x_target = self.norm(t_raw)

        return torch.from_numpy(x_in), torch.from_numpy(x_target)

# ==============================================================================
# 2. LOSS FUNCTION
# ==============================================================================
def loss_fn(recon, target, mu=None, logvar=None, beta=1.0, data_type='mlp'):
    # 1. Reconstruction Loss
    if data_type == 'cnn':
        # Masked MSE for 2D (Channel 0=Value, Channel 1=Mask)
        target_val = target[:, 0:1, :, :]
        mask       = target[:, 1:2, :, :]
        recon_val  = recon[:, 0:1, :, :] # Extract value channel
        mse = ((recon_val - target_val)**2 * mask).sum() / mask.sum()
    else:
        # Standard MSE for 1D
        mse = F.mse_loss(recon, target)

    # 2. KL Divergence (Only for VAEs)
    kld = torch.tensor(0.0, device=recon.device)
    if mu is not None and logvar is not None:
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kld = kld / recon.size(0)

    return mse + beta * kld, mse, kld

# ==============================================================================
# 3. TRAINING ROUTINES
# ==============================================================================
def get_model(name, latent_dim):
    if name == 'cnn_ae':   return CNN_AE(latent_dim=latent_dim), 'cnn'
    if name == 'cnn_vae':  return CNN_VAE(latent_dim=latent_dim), 'cnn'
    if name == 'mlp_ae':   return MLP_AE(latent_dim=latent_dim), 'mlp'
    if name == 'mlp_vae':  return MLP_VAE(latent_dim=latent_dim), 'mlp'
    if name == 'posenc':   return PosEncAE(latent_dim=latent_dim), 'mlp'
    raise ValueError(f"Unknown model: {name}")

def run_epoch(model, loader, opt, beta, data_type, device, is_train=True):
    if is_train: model.train()
    else: model.eval()
    
    total_loss, total_mse = 0, 0
    
    with torch.set_grad_enabled(is_train):
        for x, y in tqdm(loader, leave=False, desc="Train" if is_train else "Val"):
            x, y = x.to(device), y.to(device)
            
            if is_train: opt.zero_grad()
            
            # Forward
            out = model(x)
            if isinstance(out, tuple): recon, mu, logvar = out
            else: recon, mu, logvar = out, None, None
            
            # Loss
            loss, mse, _ = loss_fn(recon, y, mu, logvar, beta, data_type)
            
            if is_train:
                loss.backward()
                opt.step()
            
            total_loss += loss.item()
            total_mse += mse.item()
            
    return total_loss / len(loader), total_mse / len(loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--val_path", required=True)
    parser.add_argument("--save_dir", default="./weights")
    parser.add_argument("--model", choices=['cnn_ae', 'cnn_vae', 'mlp_ae', 'mlp_vae', 'posenc'], required=True)
    parser.add_argument("--latent_dim", type=int, default=8)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--stage1_epochs", type=int, default=50)
    parser.add_argument("--stage1_lr", type=float, default=1e-3)
    parser.add_argument("--stage2_epochs", type=int, default=50)
    parser.add_argument("--stage2_lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project_name", default="TwoStage_VF")
    parser.add_argument("--run_name", default=None)
    
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 1. Setup Model
    model, data_type = get_model(args.model, args.latent_dim)
    model = model.to(device)
    
    if args.run_name is None:
        ts = datetime.now().strftime("%m%d-%H%M")
        run_name = f"{args.model}_z{args.latent_dim}_beta{args.beta}_{ts}"
    else:
        run_name = args.run_name

    wandb.init(project=args.project_name, name=run_name, config=vars(args))
    
    # ==========================================================================
    # STAGE 1: UNSUPERVISED (Input -> Input)
    # ==========================================================================
    print(f"\nSTAGE 1: Unsupervised Pre-training ({args.stage1_epochs} epochs)")
    ds1_train = TwoStageDataset(args.train_path, mode="unsupervised", data_type=data_type)
    dl1_train = DataLoader(ds1_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    ds1_val = TwoStageDataset(args.val_path, mode="unsupervised", data_type=data_type)
    dl1_val = DataLoader(ds1_val, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    opt1 = optim.Adam(model.parameters(), lr=args.stage1_lr)
    
    for ep in range(1, args.stage1_epochs + 1):
        tr_loss, tr_mse = run_epoch(model, dl1_train, opt1, args.beta, data_type, device, is_train=True)
        val_loss, val_mse = run_epoch(model, dl1_val, opt1, args.beta, data_type, device, is_train=False)
        
        wandb.log({
            "stage": 1, 
            "epoch": ep,
            "s1_train_loss": tr_loss, 
            "s1_val_loss": val_loss,
            "s1_val_mse": val_mse
        })
        print(f"S1 Ep {ep}: Train Loss {tr_loss:.4f} | Val MSE {val_mse:.4f}")
    
    s1_path = os.path.join(args.save_dir, f"{run_name}_stage1.pth")
    torch.save(model.state_dict(), s1_path)
    print(f"Stage 1 saved: {s1_path}")

    # ==========================================================================
    # STAGE 2: FINE-TUNING (Input -> Clean Target)
    # ==========================================================================
    print(f"\nSTAGE 2: Supervised Fine-tuning ({args.stage2_epochs} epochs)")
    
    ds2_train = TwoStageDataset(args.train_path, mode="finetune", data_type=data_type)
    dl2_train = DataLoader(ds2_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    ds2_val = TwoStageDataset(args.val_path, mode="finetune", data_type=data_type)
    dl2_val = DataLoader(ds2_val, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Reset Optimizer with lower LR
    opt2 = optim.Adam(model.parameters(), lr=args.stage2_lr)
    
    for ep in range(1, args.stage2_epochs + 1):
        tr_loss, tr_mse = run_epoch(model, dl2_train, opt2, args.beta, data_type, device, is_train=True)
        val_loss, val_mse = run_epoch(model, dl2_val, opt2, args.beta, data_type, device, is_train=False)
        
        wandb.log({
            "stage": 2, 
            "epoch": args.stage1_epochs + ep,
            "s2_train_loss": tr_loss, 
            "s2_val_loss": val_loss,
            "s2_val_mse": val_mse
        })
        print(f"S2 Ep {ep}: Train Loss {tr_loss:.4f} | Val MSE {val_mse:.4f}")
        
    s2_path = os.path.join(args.save_dir, f"{run_name}_final.pth")
    torch.save(model.state_dict(), s2_path)
    print(f"Stage 2 saved: {s2_path}")

if __name__ == "__main__":
    main()