import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import argparse
import wandb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ============================================================================
# 1. VISUAL FIELD MAPPING (8x9 Grid Logic)
# ============================================================================

def get_valid_indices_8x9():
    """Returns the indices in a flattened 8x9 (72) grid that correspond to the 52 VF points."""
    nulls = [0,1,2,7,8,9,10,17,18,34,43,45,54,55,62,63,64,65,70,71]
    all_indices = np.arange(8 * 9)
    return [i for i in all_indices if i not in nulls]

VALID_INDICES_8x9 = get_valid_indices_8x9()

def map_vf_to_8x9(vector):
    """Maps (52,) vector -> (8, 9) spatial grid"""
    img_flat = np.zeros(72, dtype=np.float32)
    limit = min(len(VALID_INDICES_8x9), len(vector))
    img_flat[VALID_INDICES_8x9[:limit]] = vector[:limit]
    return img_flat.reshape(8, 9)

# ============================================================================
# 2. NORMALIZATION & UTILS
# ============================================================================

def normalize_rnflt(x):
    x = np.clip(x, 0, 350)
    return (x / 350.0).astype(np.float32)

def denormalize_rnflt(x):
    return (x * 350.0)

def normalize_vf(x, min_val=-38.0, max_val=26.0):
    """Normalize VF values to [-1, 1] range"""
    return 2 * (x - min_val) / (max_val - min_val) - 1

def denormalize_vf(x, min_val=-38.0, max_val=26.0):
    """Denormalize VF values back to original range"""
    return (x + 1) * (max_val - min_val) / 2 + min_val

def flip_rnfl_image(img):
    return np.fliplr(img).copy()

def create_circular_mask(h, w, center=None, radius=None):
    if center is None: 
        center = (int(w/2), int(h/2))
    if radius is None: 
        radius = int(min(h, w) * 0.15)
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = dist_from_center > radius 
    return torch.from_numpy(mask).float()

def load_paths_and_ids(folder):
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
        except: 
            continue
    return all_ids, id_to_path

# ============================================================================
# 3. DATASET
# ============================================================================

class VFtoRNFLDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, min_val=-38.0, max_val=26.0, target_size=None):
        self.samples = []
        self.target_size = target_size
        
        if len(file_paths) > 0:
            try:
                test_data = np.load(file_paths[0], allow_pickle=True)
                self.img_size = test_data['rnflt'].shape[0]
            except: self.img_size = 200
        else: self.img_size = 200

        for path in file_paths:
            try:
                fname = os.path.basename(path)
                is_left_eye = "_0_" in fname
                data = np.load(path, allow_pickle=True)
                
                vf_norm = normalize_vf(data['td'], min_val, max_val)
                rnflt_norm = normalize_rnflt(data['rnflt'])

                if is_left_eye: 
                    rnflt_norm = flip_rnfl_image(rnflt_norm)
                
                self.samples.append((vf_norm, rnflt_norm))
            except: pass

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        vf_vector, rnflt = self.samples[idx]
        
        vf_grid = map_vf_to_8x9(vf_vector)
        vf_tensor = torch.tensor(vf_grid, dtype=torch.float32).unsqueeze(0)

        # avg
        vf_avg = torch.full_like(vf_tensor, vf_tensor.mean())

        rnflt_tensor = torch.tensor(rnflt, dtype=torch.float32).unsqueeze(0) 
        if self.target_size and rnflt_tensor.shape[-1] != self.target_size:
            rnflt_tensor = F.interpolate(rnflt_tensor.unsqueeze(0), size=(self.target_size, self.target_size), mode='bilinear').squeeze(0)
            
        return vf_avg, rnflt_tensor

# ============================================================================
# 3. MODELS (Same structure as VF-to-RNFL, but reversed direction)
# ============================================================================

# --- Standard AE Components (from pretrained autoencoder) ---
class RNFL_Encoder(nn.Module):
    def __init__(self, z_dim=64):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(), nn.Conv2d(256, 256, 3, padding=1), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(256 * 4 * 4, z_dim)
    def forward(self, x):
        x = self.conv4(self.conv3(self.conv2(self.conv1(x))))
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)

class RNFL_Decoder(nn.Module):
    def __init__(self, z_dim=64, img_size=200):
        super().__init__()
        self.img_size = img_size
        self.fc = nn.Linear(z_dim, 256 * 4 * 4)
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(256, 256, 3, padding=1), nn.ReLU())
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(256, 128, 3, padding=1), nn.ReLU())
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(128, 64, 3, padding=1), nn.ReLU())
        self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(64, 32, 3, padding=1), nn.ReLU())
        self.up_final = nn.Sequential(
            nn.Upsample(size=(img_size, img_size), mode='bilinear'),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1), nn.Sigmoid()
        )
    def forward(self, z):
        x = self.fc(z).view(z.size(0), 256, 4, 4)
        x = self.up4(self.up3(self.up2(self.up1(x))))
        return self.up_final(x)

class RNFL_Autoencoder(nn.Module):
    def __init__(self, z_dim=64, img_size=200):
        super().__init__()
        self.encoder = RNFL_Encoder(z_dim=z_dim)
        self.decoder = RNFL_Decoder(z_dim=z_dim, img_size=img_size)
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

# --- VF SPATIAL ENCODER ---
class VF_Spatial_Encoder(nn.Module):
    def __init__(self, z_dim=32, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        
        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 9, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, z_dim)
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1) 
        return self.fc(x)

# --- VF SKIP CONNECTION MODULE ---
class VF_Skip_Module(nn.Module):
    """
    Extracts spatial features directly from VF grid and upsamples to RNFL size.
    This provides a direct pathway for VF information to influence final reconstruction.
    """
    def __init__(self, img_size=200, skip_channels=16):
        super().__init__()
        self.img_size = img_size
        
        # Extract spatial features from 8x9 VF grid
        self.vf_feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, skip_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(skip_channels),
            nn.ReLU(),
        )
        
        # Upsample to match RNFL spatial dimensions
        self.upsampler = nn.Upsample(size=(img_size, img_size), mode='bilinear', align_corners=False)
    
    def forward(self, vf):
        """
        Args:
            vf: (B, 1, 8, 9) - VF spatial grid
        Returns:
            (B, skip_channels, img_size, img_size) - Upsampled VF features
        """
        features = self.vf_feature_extractor(vf)  # (B, skip_channels, 8, 9)
        features_upsampled = self.upsampler(features)  # (B, skip_channels, 200, 200)
        return features_upsampled

# --- MODEL WRAPPER WITH SKIP CONNECTIONS ---
class VF2RNFL_Model(nn.Module):
    def __init__(self, decoder, z_dim=32, dropout=0.2, img_size=200, skip_channels=16):
        super().__init__()
        self.decoder = decoder
        self.img_size = img_size
        
        # Main VF encoder (to latent space)
        self.vf_enc = VF_Spatial_Encoder(z_dim=z_dim, dropout=dropout)
        
        # Skip connection module
        self.vf_skip = VF_Skip_Module(img_size=img_size, skip_channels=skip_channels)
        
        # Fusion layer to combine decoder output with VF skip features
        # Input: 1 (decoder) + skip_channels (VF features) = 1 + skip_channels
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(1 + skip_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),  # Final 1x1 conv to single channel
            nn.Sigmoid()
        )
        
        self.register_buffer('z_mean', torch.zeros(z_dim))
        self.register_buffer('z_std', torch.ones(z_dim))

    def forward(self, vf):
        """
        Args:
            vf: (B, 1, 8, 9) - VF spatial grid (averaged or spatial)
        Returns:
            rnflt_hat: (B, 1, img_size, img_size) - Reconstructed RNFL
            z_norm: (B, z_dim) - Normalized latent code
        """
        # Path 1: Main latent pathway (VF -> z -> decoder -> base RNFL)
        z_norm = self.vf_enc(vf)
        z_restored = z_norm * self.z_std + self.z_mean
        rnflt_base = self.decoder(z_restored)  # (B, 1, img_size, img_size)
        
        # Path 2: Skip connection pathway (VF -> spatial features -> upsampled)
        vf_features = self.vf_skip(vf)  # (B, skip_channels, img_size, img_size)
        
        # Fusion: Combine decoder output with VF skip features
        combined = torch.cat([rnflt_base, vf_features], dim=1)  # (B, 1+skip_channels, H, W)
        rnflt_final = self.fusion_layer(combined)  # (B, 1, img_size, img_size)
        
        return rnflt_final, z_norm

# ============================================================================
# 4. TRAINING LOOP
# ============================================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    print(f"Loading data from: {args.folder}")
    all_ids, id_to_path = load_paths_and_ids(args.folder)
    print(f"Found {len(all_ids)} samples")
    
    train_ids, val_ids = train_test_split(
        all_ids, 
        test_size=args.split_ratio, 
        random_state=args.split_seed
    )
    
    train_paths = [id_to_path[i] for i in train_ids if i in id_to_path]
    val_paths = [id_to_path[i] for i in val_ids if i in id_to_path]
    
    print(f"Train: {len(train_paths)} | Val: {len(val_paths)}")
    
    train_set = VFtoRNFLDataset(train_paths)
    val_set = VFtoRNFLDataset(val_paths)
    
    train_loader = torch.utils.data.DataLoader(
        train_set, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    img_size = train_set.img_size
    print(f"Image Size: {img_size}x{img_size}")

    # 2. Load AE and Compute Stats
    print("Loading Autoencoder Checkpoint...")
    ae = RNFL_Autoencoder(z_dim=args.z_dim, img_size=img_size).to(device)
    ckpt = torch.load(args.autoencoder_ckpt, map_location=device)
    ae.load_state_dict(ckpt["model_state_dict"])
    ae.eval()

    print("Computing Latent Space Stats (Mean/Std)...")
    z_list = []
    with torch.no_grad():
        for _, rnflt in tqdm(train_loader):
            z = ae.encoder(rnflt.to(device))
            z_list.append(z)
    z_all = torch.cat(z_list, dim=0)
    z_mean = z_all.mean(dim=0)
    z_std = z_all.std(dim=0) + 1e-6
    print("Stats Computed.")

    # 3. Init Model WITH SKIP CONNECTIONS
    print(f"Initializing model with {args.skip_channels}-channel VF skip connections...")
    model = VF2RNFL_Model(
        ae.decoder, 
        z_dim=args.z_dim, 
        dropout=args.dropout,
        img_size=img_size,
        skip_channels=args.skip_channels
    ).to(device)
    model.z_mean.data = z_mean
    model.z_std.data = z_std

    # FREEZING STRATEGY
    print(f"\nApplying Freeze Strategy: {args.freeze_strategy.upper()}")
    
    # Freeze everything first
    for p in model.parameters(): 
        p.requires_grad = False
    
    # Always unfreeze VF encoder
    for p in model.vf_enc.parameters(): 
        p.requires_grad = True
    
    # Always unfreeze skip connection and fusion modules (they are new!)
    for p in model.vf_skip.parameters(): 
        p.requires_grad = True
    for p in model.fusion_layer.parameters(): 
        p.requires_grad = True
    print("   -> VF Encoder: UNFROZEN (trainable)")
    print("   -> VF Skip Module: UNFROZEN (trainable)")
    print("   -> Fusion Layer: UNFROZEN (trainable)")
    
    # Conditional decoder unfreezing
    if args.freeze_strategy == "all":
        print("   -> Decoder: 100% FROZEN")
    elif args.freeze_strategy == "tail":
        print("   -> Decoder: Unfreezing 'up_final'")
        for p in model.decoder.up_final.parameters(): 
            p.requires_grad = True
    elif args.freeze_strategy == "half":
        print("   -> Decoder: Unfreezing 'up4' and 'up_final'")
        for p in model.decoder.up4.parameters(): 
            p.requires_grad = True
        for p in model.decoder.up_final.parameters(): 
            p.requires_grad = True

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Loss Setup
    print(f"Using Circular Mask (Radius {args.mask_radius}px)")
    loss_mask = create_circular_mask(img_size, img_size, radius=args.mask_radius).to(device)
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    
    wandb.init(project=args.project, name=args.run_name, config=vars(args))
    best_loss = float('inf')

    # 4. Training Loop
    print(f"\nStarting training for {args.epochs} epochs...\n")
    
    for epoch in range(1, args.epochs + 1):
        # === TRAINING ===
        model.train()
        train_stats = {"loss": 0, "l1": 0, "latent": 0}
        
        for vf, rnflt in tqdm(train_loader, desc=f"Ep {epoch}"):
            vf, rnflt = vf.to(device), rnflt.to(device)
            
            with torch.no_grad():
                z_teacher = (ae.encoder(rnflt) - z_mean) / z_std

            optimizer.zero_grad()
            pred_rnflt, z_pred = model(vf)
            
            # Masked Loss
            masked_pred = pred_rnflt * loss_mask
            masked_target = rnflt * loss_mask
            loss_pix = l1_loss(masked_pred, masked_target)
            loss_lat = mse_loss(z_pred, z_teacher)
            
            total_loss = args.pixel_lambda * loss_pix + args.latent_lambda * loss_lat
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_stats["loss"] += total_loss.item()
            train_stats["l1"] += loss_pix.item()
            train_stats["latent"] += loss_lat.item()

        # === VALIDATION ===
        model.eval()
        val_loss, val_mae = 0, 0
        r2_preds = []
        r2_targets = []
        
        with torch.no_grad():
            for vf, rnflt in val_loader:
                vf, rnflt = vf.to(device), rnflt.to(device)
                pred, z = model(vf)
                
                masked_pred = pred * loss_mask
                masked_target = rnflt * loss_mask
                
                loss_val = args.pixel_lambda * l1_loss(masked_pred, masked_target) + \
                           args.latent_lambda * mse_loss(z, (ae.encoder(rnflt) - z_mean)/z_std)
                
                val_loss += loss_val.item()
                val_mae += torch.mean(torch.abs(denormalize_rnflt(masked_pred) - denormalize_rnflt(masked_target))).item()
                
                # R2
                mask_bool = loss_mask.bool().unsqueeze(0).unsqueeze(0)   # (1,1,H,W)
                mask_bool = mask_bool.expand_as(pred)                    # (B,1,H,W)

                p_dn = denormalize_rnflt(pred)
                t_dn = denormalize_rnflt(rnflt)

                valid_p = p_dn.masked_select(mask_bool).cpu().numpy()
                valid_t = t_dn.masked_select(mask_bool).cpu().numpy()

                r2_preds.append(valid_p)
                r2_targets.append(valid_t)

        avg_val_loss = val_loss / len(val_loader)
        avg_mae = val_mae / len(val_loader)
        all_preds = np.concatenate(r2_preds)
        all_targets = np.concatenate(r2_targets)
        val_r2 = r2_score(all_targets, all_preds)
        
        scheduler.step(avg_val_loss)
        
        wandb.log({
            "epoch": epoch,
            "val_mae": avg_mae, 
            "val_loss": avg_val_loss,
            "val_r2": val_r2,
            "train_loss": train_stats["loss"] / len(train_loader),
            "train_l1": train_stats["l1"] / len(train_loader),
            "train_latent": train_stats["latent"] / len(train_loader),
            "lr": optimizer.param_groups[0]['lr']
        })
        
        print(f"Ep {epoch} | Val MAE: {avg_mae:.2f} µm | R²: {val_r2:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_mae': avg_mae,
                'val_r2': val_r2,
                'args': vars(args)
            }, f"checkpoints/{args.run_name}_best.pth")
            print("Saved Best Model")

    # Save final checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'val_mae': avg_mae,
        'val_r2': val_r2,
        'args': vars(args)
    }, f"checkpoints/{args.run_name}_final.pth")
    print(f"Saved Final Checkpoint (Epoch {epoch})")
    
    wandb.finish()

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AvgVF to RNFL model")
    
    # Data
    parser.add_argument("--folder", type=str, required=True, 
                        help="Path to folder containing .npz files")
    parser.add_argument("--autoencoder_ckpt", type=str, required=True,
                        help="Path to pretrained autoencoder checkpoint")
    parser.add_argument("--split_ratio", type=float, default=0.2,
                        help="Validation split ratio")
    parser.add_argument("--split_seed", type=int, default=42,
                        help="Random seed for train/val split")
    
    # Model
    parser.add_argument("--z_dim", type=int, default=32,
                        help="Latent dimension size")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    parser.add_argument("--freeze_strategy", type=str, default="all", 
                        choices=["all", "tail", "half"],
                        help="Decoder freeze strategy")
    
    # Training
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    
    # Loss weights
    parser.add_argument("--pixel_lambda", type=float, default=10.0,
                        help="Weight for pixel loss")
    parser.add_argument("--latent_lambda", type=float, default=0.5,
                        help="Weight for latent loss")
    parser.add_argument("--mask_radius", type=int, default=30,
                        help="Circular mask radius in pixels")
    
    # Skip connections
    parser.add_argument("--skip_channels", type=int, default=16, 
                        help="Number of channels in VF skip connection")
    
    # Logging
    parser.add_argument("--project", type=str, default="AvgVF_to_RNFL",
                        help="W&B project name")
    parser.add_argument("--run_name", type=str, default="avgvf_baseline",
                        help="W&B run name")
    
    args = parser.parse_args()
    
    # Create checkpoint directory
    os.makedirs("checkpoints", exist_ok=True)
    
    # Start training
    train(args)

"""
# ============================================================================
# USAGE EXAMPLES - AvgVF to RNFL (Same structure as VF-to-RNFL)
# ============================================================================

# Version 1: Fully frozen decoder (most conservative)
python rnfl_to_avg_vf.py \
  --folder $VF_DATA_ROOT/original_below350 \
  --autoencoder_ckpt checkpoints/masked30_ae_z32_autoencoder_best.pth \
  --run_name AvgVF_Raw_Frozen \
  --z_dim 32 \
  --lr 3e-4 \
  --pixel_lambda 10.0 \
  --latent_lambda 0.5 \
  --dropout 0.1 \
  --mask_radius 30 \
  --freeze_strategy all

# Version 2: Tail unfrozen (recommended baseline)
python rnfl_to_avg_vf.py \
  --folder $VF_DATA_ROOT/original_below350 \
  --autoencoder_ckpt checkpoints/masked30_ae_z32_autoencoder_best.pth \
  --run_name AvgVF_Raw_Tail \
  --z_dim 32 \
  --lr 3e-4 \
  --pixel_lambda 8.0 \
  --latent_lambda 1.0 \
  --dropout 0.05 \
  --mask_radius 30 \
  --freeze_strategy tail

# Version 3: With denoised NAFNet data
nohup python rnfl_to_avg_vf.py \
  --folder $VF_DATA_ROOT/denoised_NAFNet_models/denoised_NAFNet_Advanced \
  --autoencoder_ckpt checkpoints/masked30_ae_z32_autoencoder_best.pth \
  --run_name AvgVF_NAFNet_Tail \
  --z_dim 32 \
  --lr 3e-4 \
  --pixel_lambda 10.0 \
  --latent_lambda 0.5 \
  --dropout 0.05 \
  --mask_radius 30 \
  --freeze_strategy tail \
  > logs/avgvf_nafnet.log 2>&1 &

# Version 4: Half decoder unfrozen (more flexibility)
nohup python rnfl_to_avg_vf.py \
  --folder $VF_DATA_ROOT/denoised_NAFNet_models/denoised_NAFNet_Advanced \
  --autoencoder_ckpt checkpoints/masked30_ae_z32_autoencoder_best.pth \
  --run_name AvgVF_NAFNet_Half \
  --z_dim 32 \
  --lr 1e-4 \
  --pixel_lambda 10.0 \
  --latent_lambda 1.0 \
  --dropout 0.0 \
  --mask_radius 30 \
  --freeze_strategy half \
  --epochs 150 \
  > logs/avgvf_nafnet_half.log 2>&1 &
"""