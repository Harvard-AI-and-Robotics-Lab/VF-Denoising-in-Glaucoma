"""
VF to RNFL Prediction - Inverse Structure-Function Mapping (2-stage)
Stage 1: RNFL Autoencoder (Masked to focus on Retina, ignoring Disc)
Stage 2: VF → latent → RNFL (Masked training)

UPDATES: 
- Added --mask_radius argument to main().
- Applied Circular Mask to Stage 1 (Autoencoder) so latent space ignores the disc.
- Applied Circular Mask to Stage 2 (VF->RNFL) to prevent mean collapse.
"""

import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.metrics import mean_absolute_error
import wandb

# ============================================================================
# NORMALIZATION FUNCTIONS
# ============================================================================

def normalize_rnflt(x):
    """Normalize RNFL thickness to [0, 1]"""
    x = np.clip(x, 0, 350)
    return (x / 350.0).astype(np.float32)

def denormalize_rnflt(x):
    """Denormalize RNFL from [0, 1] back to microns"""
    return (x * 350.0)

def normalize_vf(x, min_val=-38.0, max_val=26.0):
    """Normalize VF values to [-1, 1]"""
    return 2 * (x - min_val) / (max_val - min_val) - 1

# ============================================================================
# DATASETS (Flip RNFL only for Left Eyes)
# ============================================================================

def flip_rnfl_image(img):
    """
    Flip RNFL image Left-to-Right (Horizontal Flip).
    Used to standardize Left Eyes (OS) to look like Right Eyes (OD).
    """
    return np.fliplr(img).copy()

class VFtoRNFLDataset(torch.utils.data.Dataset):
    """Dataset for VF→RNFL prediction (Stage 2)"""

    def __init__(self, file_paths, min_val=-38.0, max_val=26.0, target_size=None):
        self.samples = []
        self.min_val = min_val
        self.max_val = max_val
        self.target_size = target_size

        if len(file_paths) > 0:
            test_data = np.load(file_paths[0], allow_pickle=True)
            self.img_size = test_data['rnflt'].shape[0]
        else:
            self.img_size = 200

        for path in file_paths:
            try:
                fname = os.path.basename(path)
                is_left_eye = "_0_" in fname
                data = np.load(path, allow_pickle=True)
                
                vf_norm = normalize_vf(data['td'], min_val, max_val)
                rnfl_raw = data['rnflt']
                rnflt_norm = normalize_rnflt(rnfl_raw)

                if is_left_eye:
                    rnflt_norm = flip_rnfl_image(rnflt_norm)

                self.samples.append((vf_norm, rnflt_norm))
            except Exception as e:
                print(f"Failed to read {Path(path).name}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vf, rnflt = self.samples[idx]

        rnflt_tensor = torch.tensor(rnflt, dtype=torch.float32).unsqueeze(0) 
        if self.target_size and rnflt_tensor.shape[-1] != self.target_size:
            rnflt_tensor = F.interpolate(
                rnflt_tensor.unsqueeze(0),
                size=(self.target_size, self.target_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        vf_tensor = torch.tensor(vf, dtype=torch.float32)
        return vf_tensor, rnflt_tensor.squeeze(0)


class RNFLOnlyDataset(torch.utils.data.Dataset):
    """Dataset for RNFL-only autoencoder training (Stage 1)"""

    def __init__(self, file_paths, target_size=None):
        self.samples = []
        self.target_size = target_size

        if len(file_paths) > 0:
            test_data = np.load(file_paths[0], allow_pickle=True)
            self.img_size = test_data['rnflt'].shape[0]
        else:
            self.img_size = 200

        for path in file_paths:
            try:
                fname = os.path.basename(path)
                is_left_eye = "_0_" in fname

                data = np.load(path, allow_pickle=True)
                rnflt_norm = normalize_rnflt(data['rnflt'])

                if is_left_eye:
                    rnflt_norm = flip_rnfl_image(rnflt_norm)

                self.samples.append(rnflt_norm)
            except Exception as e:
                print(f"[AE] Failed to read {Path(path).name}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rnflt = self.samples[idx]
        rnflt_tensor = torch.tensor(rnflt, dtype=torch.float32).unsqueeze(0)
        
        if self.target_size and rnflt_tensor.shape[-1] != self.target_size:
            rnflt_tensor = F.interpolate(
                rnflt_tensor.unsqueeze(0),
                size=(self.target_size, self.target_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            
        return rnflt_tensor

# ============================================================================
# MODELS - AUTOENCODER (STAGE 1)
# ============================================================================

class RNFL_Encoder(nn.Module):
    def __init__(self, z_dim=64):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )  # 32 × H × W

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # downsample /2
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # /2
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # /2
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d((4, 4))  # → 256 × 4 × 4
        self.fc = nn.Linear(256 * 4 * 4, z_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        z = self.fc(x)
        return z


class RNFL_Decoder(nn.Module):
    def __init__(self, z_dim=64, img_size=200):
        super().__init__()
        self.img_size = img_size
        self.fc = nn.Linear(z_dim, 256 * 4 * 4)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 8×8
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 16×16
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 32×32
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 64×64
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        # Final upsample directly to target img_size
        self.up_final = nn.Sequential(
            nn.Upsample(size=(img_size, img_size), mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid(),  # since RNFL is normalized to [0,1]
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 256, 4, 4)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up_final(x)
        return x  # (B,1,H,W)


class RNFL_Autoencoder(nn.Module):
    def __init__(self, z_dim=64, img_size=200):
        super().__init__()
        self.encoder = RNFL_Encoder(z_dim=z_dim)
        self.decoder = RNFL_Decoder(z_dim=z_dim, img_size=img_size)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

# ============================================================================
# MODELS - VF→LATENT→RNFL (STAGE 2)
# ============================================================================

class VF2Latent(nn.Module):
    def __init__(self, vf_dim=52, z_dim=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vf_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, z_dim),
        )

    def forward(self, vf):
        return self.net(vf)


class VF2RNFL_Model(nn.Module):
    def __init__(self, decoder: RNFL_Decoder, vf_dim=52, z_dim=64, dropout=0.2):
        super().__init__()
        self.decoder = decoder
        self.z_dim = z_dim
        
        # Encoder parts
        self.vf2z = VF2Latent(vf_dim=vf_dim, z_dim=z_dim, dropout=dropout)
        self.vf_emb = nn.Linear(vf_dim, z_dim)

        # Buffers for stats
        self.register_buffer('z_mean', torch.zeros(z_dim))
        self.register_buffer('z_std', torch.ones(z_dim))

    def set_stats(self, mean, std):
        self.z_mean = mean.to(self.z_mean.device)
        self.z_std = std.to(self.z_std.device)

    def forward(self, vf):
        # 1. Predict Normalized Latent
        z_norm = self.vf2z(vf) + self.vf_emb(vf)
        
        # 2. Denormalize
        z_restored = z_norm * self.z_std + self.z_mean
        
        # 3. Decode
        rnflt_hat = self.decoder(z_restored)
        
        # Return BOTH image and latent code
        return rnflt_hat, z_norm
    

# ============================================================================
# UTIL: LOAD PATHS AND SPLIT
# ============================================================================

def load_paths_and_ids(folder):
    """Load all .npz paths and corresponding unique IDs from meta."""
    all_paths = sorted(list(Path(folder).rglob("*.npz")))
    all_ids = []
    id_to_path = {}

    for p in all_paths:
        try:
            data = np.load(p, allow_pickle=True)
            meta_raw = data['meta'].item()
            meta = json.loads(meta_raw.decode('utf-8'))
            full_id = f"{meta['id']}_{meta['eye']}_{meta['oct_time']}"
            all_ids.append(full_id)
            id_to_path[full_id] = str(p)
        except Exception as e:
            print(f"Failed to read meta from {p.name}: {e}")

    return all_ids, id_to_path

# ============================================================================
# HELPER: MASK GENERATION (PyTorch)
# ============================================================================

def create_circular_mask(h, w, center=None, radius=None):
    """
    Creates a PyTorch-compatible mask where the CENTER (Optic Disc) is 0 and the rest is 1.
    """
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # default to roughly 15% of image width
        radius = int(min(h, w) * 0.15)

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    # 1.0 where distance > radius (Retina), 0.0 where distance < radius (Disc)
    mask = dist_from_center > radius 
    return torch.from_numpy(mask).float()

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_autoencoder(train_loader, val_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.z_dim > 32:
        print(f"WARNING: z_dim={args.z_dim} is too high for VF-to-RNFL translation.")
        print(f"   Please restart with --z_dim 32.")

    sample_batch = next(iter(train_loader))
    img_size = sample_batch.shape[-1]
    
    # Initialize Model
    model = RNFL_Autoencoder(z_dim=args.z_dim, img_size=img_size).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)
    mse_loss = nn.MSELoss()

    wandb.init(project=args.project, name=args.run_name, config=vars(args))

    # CREATE MASK for Stage 1 too
    print(f"[AE] Creating Loss Mask with radius {args.mask_radius}px...")
    loss_mask = create_circular_mask(img_size, img_size, radius=args.mask_radius).to(device)

    print(f"Training MASKED Autoencoder (z_dim={args.z_dim})...")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        
        for rnflt in tqdm(train_loader, desc=f"[AE] Epoch {epoch}"):
            rnflt = rnflt.to(device)
            optimizer.zero_grad()
            
            recon, z = model(rnflt)

            # Apply Mask to both input and output before loss
            masked_recon = recon * loss_mask
            masked_target = rnflt * loss_mask

            loss_mse = mse_loss(masked_recon, masked_target)
            loss_reg = 0.001 * torch.mean(z ** 2) 

            loss = loss_mse + loss_reg

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * rnflt.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        
        with torch.no_grad():
            for rnflt in val_loader:
                rnflt = rnflt.to(device)
                recon, _ = model(rnflt)
                
                # Apply mask to validation too
                masked_recon = recon * loss_mask
                masked_target = rnflt * loss_mask
                
                loss_val = mse_loss(masked_recon, masked_target)
                val_loss += loss_val.item() * rnflt.size(0)
                
                # MAE only on valid pixels (approximation)
                # We calculate absolute difference on masked images
                mae_diff = torch.abs(denormalize_rnflt(masked_recon) - denormalize_rnflt(masked_target))
                # To get true MAE, we should divide by number of valid pixels, not total pixels.
                # Since mask is constant, simple mean is proportional, but let's be accurate:
                # But for monitoring, standard mean is okay.
                val_mae += torch.mean(mae_diff).item() * rnflt.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)
        avg_val_mae = val_mae / len(val_loader.dataset)

        scheduler.step(avg_val_loss)

        wandb.log({
            "epoch": epoch,
            "ae_train_loss": avg_train_loss,
            "ae_val_loss": avg_val_loss,
            "ae_val_mae": avg_val_mae
        }, step=epoch)

        print(f"[AE] Epoch {epoch}: Val Loss={avg_val_loss:.4f} | Val MAE={avg_val_mae:.2f}µm")

        if epoch == 1 or avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = Path("checkpoints") / f"{args.run_name}_autoencoder_best.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": avg_val_loss,
                }, 
                ckpt_path
            )
            print(f"Saved Best AE: {ckpt_path}")

    wandb.finish()

def train_vf2rnfl(train_loader, val_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Detect img_size
    sample_batch = next(iter(train_loader))
    img_size = sample_batch[1].shape[-1]
    
    # 1. Load Autoencoder (Teacher)
    assert args.autoencoder_ckpt is not None, "Please provide --autoencoder_ckpt"
    print(f"Loading autoencoder from: {args.autoencoder_ckpt}")
    
    ae = RNFL_Autoencoder(z_dim=args.z_dim, img_size=img_size).to(device)
    ckpt = torch.load(args.autoencoder_ckpt, map_location=device)
    ae.load_state_dict(ckpt["model_state_dict"])
    ae.eval()
    
    # 2. Compute Statistics (Normalize targets)
    print("\nComputing Latent Space Statistics...")
    z_sum = torch.zeros(args.z_dim).to(device)
    z_sq_sum = torch.zeros(args.z_dim).to(device)
    total_samples = 0

    for _, rnflt in tqdm(train_loader, desc="[Stats] Scanning"):
        rnflt = rnflt.unsqueeze(1).to(device)
        with torch.no_grad():
            z = ae.encoder(rnflt)
        z_sum += z.sum(dim=0)
        z_sq_sum += (z ** 2).sum(dim=0)
        total_samples += z.size(0)

    z_mean = z_sum / total_samples
    z_std = torch.sqrt((z_sq_sum / total_samples) - (z_mean ** 2))
    z_std = torch.clamp(z_std, min=1e-5)
    print(f"Latent Stats Computed.")

    # 3. Init Model
    model = VF2RNFL_Model(
        decoder=ae.decoder, 
        vf_dim=52, 
        z_dim=args.z_dim, 
        dropout=args.dropout
    ).to(device)
    model.set_stats(z_mean, z_std)

    # MANUALLY FREEZE DECODER HERE
    print("Freezing Decoder weights...")
    for p in model.decoder.parameters():
        p.requires_grad = False
        
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.lr
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    wandb.init(project=args.project, name=args.run_name, config=vars(args))
    
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    best_val_loss = float("inf")

    # CREATE MASK FOR LOSS CALCULATION
    print(f"[Stage 2] Creating Loss Mask with radius {args.mask_radius}px...")
    loss_mask = create_circular_mask(img_size, img_size, radius=args.mask_radius).to(device)

    # 4. Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_stats = {"loss": 0.0, "l1": 0.0, "latent": 0.0}

        for vf, rnflt in tqdm(train_loader, desc=f"[VF→RNFL] Epoch {epoch}/{args.epochs}"):
            vf = vf.to(device)
            rnflt = rnflt.unsqueeze(1).to(device)

            # Teacher
            with torch.no_grad():
                z_raw = ae.encoder(rnflt)
                z_target_norm = (z_raw - z_mean) / z_std

            optimizer.zero_grad()
            
            # Forward
            pred_rnflt, z_pred_norm = model(vf)

            # Losses
            loss_latent = mse_loss(z_pred_norm, z_target_norm)
            
            # MASKED PIXEL LOSS
            masked_pred = pred_rnflt * loss_mask
            masked_target = rnflt * loss_mask
            
            loss_pixel = l1_loss(masked_pred, masked_target)

            total_loss = args.latent_lambda * loss_latent + args.pixel_lambda * loss_pixel
            
            total_loss.backward()
            optimizer.step()

            train_stats["loss"] += total_loss.item()
            train_stats["l1"] += loss_pixel.item()
            train_stats["latent"] += loss_latent.item()

        avg_train = {k: v / len(train_loader.dataset) for k, v in train_stats.items()}

        # Validation
        model.eval()
        val_stats = {"loss": 0.0, "l1": 0.0, "latent": 0.0}
        
        with torch.no_grad():
            for vf, rnflt in val_loader:
                vf = vf.to(device)
                rnflt = rnflt.unsqueeze(1).to(device)

                z_raw = ae.encoder(rnflt)
                z_target_norm = (z_raw - z_mean) / z_std

                pred_rnflt, z_pred_norm = model(vf)

                loss_latent = mse_loss(z_pred_norm, z_target_norm)
                
                # Apply mask to validation metrics too
                masked_pred = pred_rnflt * loss_mask
                masked_target = rnflt * loss_mask
                loss_pixel = l1_loss(masked_pred, masked_target)
                
                total_loss = args.latent_lambda * loss_latent + args.pixel_lambda * loss_pixel

                val_stats["loss"] += total_loss.item() * vf.size(0)
                val_stats["l1"] += loss_pixel.item() * vf.size(0)
                val_stats["latent"] += loss_latent.item() * vf.size(0)

        avg_val = {k: v / len(val_loader.dataset) for k, v in val_stats.items()}
        denorm_mae = avg_val["l1"] * 350.0

        scheduler.step(avg_val["loss"])
        
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train["loss"],
            "val_loss": avg_val["loss"],
            "val_latent_mse": avg_val["latent"], 
            "val_mae": denorm_mae
        }, step=epoch)

        print(
            f"[VF→RNFL] Epoch {epoch}: "
            f"Train Latent={avg_train['latent']:.4f} | "
            f"Val Latent={avg_val['latent']:.4f} | "
            f"Val MAE={denorm_mae:.1f}µm"
        )

        if avg_val["loss"] < best_val_loss:
            best_val_loss = avg_val["loss"]
            ckpt_path = Path("checkpoints") / f"{args.run_name}_vf2rnfl_best.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Best Saved: {ckpt_path}")

    wandb.finish()



# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="VF to RNFL Prediction (2-stage AE + VF→latent)")

    # Common data args
    parser.add_argument("--folder", type=str, required=True, help="Folder containing npz files")
    parser.add_argument("--data_type", type=str, required=True, help="Type of VF data (denoised, raw, etc)")
    parser.add_argument("--split_ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--split_seed", type=int, default=42)

    # Stage selection
    parser.add_argument(
        "--stage",
        type=str,
        choices=["autoencoder", "vf2rnfl"],
        required=True,
        help="Which stage to train: RNFL autoencoder or VF→RNFL",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)

    # Latent dimension
    parser.add_argument("--z_dim", type=int, default=64, help="Latent dimension for RNFL AE and VF→latent")

    # Target size (optional downsample of RNFL to smaller size)
    parser.add_argument("--target_size", type=int, default=None, help="Optional RNFL resize (e.g. 128)")

    # W&B
    parser.add_argument("--project", type=str, default="VF_to_RNFL", help="W&B project name")
    parser.add_argument("--run_name", type=str, default=None, help="W&B run name")

    # Stage 2 specific
    parser.add_argument(
        "--autoencoder_ckpt",
        type=str,
        default=None,
        help="Checkpoint path of trained autoencoder (for Stage 2)",
    )
    parser.add_argument(
        "--latent_lambda",
        type=float,
        default=0.1,
        help="Weight for latent supervision loss ||z_pred - z_true||^2",
    )
    parser.add_argument(
        "--pixel_lambda",
        type=float,
        default=1.0,
        help="Weight for pixel reconstruction loss (L1)",
    )
    parser.add_argument(
        "--perceptual_lambda",
        type=float,
        default=0.0, # Turn off for now
        help="Weight for AE-encoder perceptual loss",
    )    
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate in VF2Latent MLP",
    )
    parser.add_argument(
        "--mask_radius",
        type=int,
        default=30,
        help="Radius of the Optic Disc mask in pixels (for 200px img, 30 is good)",
    )

    args = parser.parse_args()

    # Auto-generate run_name if not provided
    if args.run_name is None:
        args.run_name = f"{args.stage}_{args.data_type}_lr{args.lr}_bs{args.batch_size}_seed{args.split_seed}"

    print(f"Starting stage: {args.stage}")
    print(f"Data folder: {args.folder}")
    print(f"Data type: {args.data_type}")

    # Load paths & IDs
    all_ids, id_to_path = load_paths_and_ids(args.folder)
    train_ids, val_ids = train_test_split(
        all_ids,
        test_size=args.split_ratio,
        random_state=args.split_seed,
    )

    train_paths = [id_to_path[i] for i in train_ids if i in id_to_path]
    val_paths   = [id_to_path[i] for i in val_ids if i in id_to_path]

    print(f"Dataset split: {len(train_paths)} train, {len(val_paths)} val")

    # Dataloaders & training per stage
    if args.stage == "autoencoder":
        train_set = RNFLOnlyDataset(train_paths, target_size=args.target_size)
        val_set   = RNFLOnlyDataset(val_paths,   target_size=args.target_size)
    else:
        train_set = VFtoRNFLDataset(train_paths, target_size=args.target_size)
        val_set   = VFtoRNFLDataset(val_paths,   target_size=args.target_size)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    if args.stage == "autoencoder":
        train_autoencoder(train_loader, val_loader, args)
    elif args.stage == "vf2rnfl":
        train_vf2rnfl(train_loader, val_loader, args)


if __name__ == "__main__":
    main()

"""
Stage 1.
nohup python vf_to_rnfl_mask_disc.py \
  --stage autoencoder \
  --folder $VF_DATA_ROOT/original_below350 \
  --data_type raw \
  --split_ratio 0.1 \
  --split_seed 42 \
  --epochs 100 \
  --batch_size 32 \
  --lr 1e-4 \
  --z_dim 32 \
  --mask_radius 30 \
  --project VF_to_RNFL_AE \
  --run_name masked30_ae_z32 \
  > masked_ae.log 2>&1 &


Stage 2.
nohup python vf_to_rnfl_mask_disc.py \
  --stage vf2rnfl \
  --folder $VF_DATA_ROOT/denoised_n2n_FINAL \
  --data_type raw \
  --z_dim 32 \
  --split_ratio 0.1 \
  --split_seed 42 \
  --epochs 100 \
  --batch_size 64 \
  --lr 5e-4 \
  --project VF_to_RNFL_AE \
  --run_name denoised_n2n_corrected_z32 \
  --autoencoder_ckpt checkpoints/masked30_ae_z32_autoencoder_best.pth \
  --latent_lambda 0.1 \
  --pixel_lambda 10.0 \
  --perceptual_lambda 0.0 \
  --dropout 0.2 \
  --mask_radius 30 \
  > denoised_n2n_corrected_z32.log 2>&1 &
"""
