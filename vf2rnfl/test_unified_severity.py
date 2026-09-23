"""
Test Script with Unified Severity Mapping

Uses pre-computed severity mapping from raw data,
ensuring consistent severity grouping across all denoising methods.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import argparse
from pathlib import Path
from sklearn.metrics import r2_score

# ============================================================================
# 1. MODEL DEFINITIONS (WITH SKIP CONNECTIONS)
# ============================================================================

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

class VF_Skip_Module(nn.Module):
    def __init__(self, img_size=200, skip_channels=16):
        super().__init__()
        self.img_size = img_size
        self.vf_feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, skip_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(skip_channels),
            nn.ReLU(),
        )
        self.upsampler = nn.Upsample(size=(img_size, img_size), mode='bilinear', align_corners=False)
    
    def forward(self, vf):
        features = self.vf_feature_extractor(vf)
        features_upsampled = self.upsampler(features)
        return features_upsampled

class VF2RNFL_Model(nn.Module):
    def __init__(self, decoder, z_dim=32, dropout=0.2, img_size=200, skip_channels=16):
        super().__init__()
        self.decoder = decoder
        self.img_size = img_size
        self.vf_enc = VF_Spatial_Encoder(z_dim=z_dim, dropout=dropout)
        self.vf_skip = VF_Skip_Module(img_size=img_size, skip_channels=skip_channels)
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(1 + skip_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.register_buffer('z_mean', torch.zeros(z_dim))
        self.register_buffer('z_std', torch.ones(z_dim))

    def forward(self, vf):
        # Path 1: Latent pathway
        z_norm = self.vf_enc(vf)
        z_restored = z_norm * self.z_std + self.z_mean
        rnflt_base = self.decoder(z_restored)
        
        # Path 2: Skip connection pathway
        vf_features = self.vf_skip(vf)
        
        # Fusion
        combined = torch.cat([rnflt_base, vf_features], dim=1)
        rnflt_final = self.fusion_layer(combined)
        
        # Return only prediction for inference (no latent needed)
        return rnflt_final

# ============================================================================
# 2. UTILS
# ============================================================================

def get_valid_indices_8x9():
    nulls = [0,1,2,7,8,9,10,17,18,34,43,45,54,55,62,63,64,65,70,71]
    return [i for i in np.arange(72) if i not in nulls]

VALID_INDICES_8x9 = get_valid_indices_8x9()

def map_vf_to_8x9(vector):
    img_flat = np.zeros(72, dtype=np.float32)
    limit = min(len(VALID_INDICES_8x9), len(vector))
    img_flat[VALID_INDICES_8x9[:limit]] = vector[:limit]
    return img_flat.reshape(8, 9)

def normalize_vf(x, min_val=-38.0, max_val=26.0):
    return 2 * (x - min_val) / (max_val - min_val) - 1

def denormalize_rnflt(x):
    return x * 350.0

def create_circular_mask(h, w, center=None, radius=30):
    if center is None: center = (int(w/2), int(h/2))
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    return dist > radius 

# ============================================================================
# 3. TEST DATASET WITH SEVERITY MAPPING
# ============================================================================

class TestDatasetWithSeverity(torch.utils.data.Dataset):
    def __init__(self, test_folder, severity_mapping, use_split_filter=False):
        """
        Args:
            test_folder: Path to test data folder (any denoising method)
            severity_mapping: Dict mapping filename -> {md, severity, split}
            use_split_filter: If True, only use files marked as 'test' in mapping.
                             If False, use ALL files in test_folder (for already-split datasets)
        """
        print(f"Loading Test Data from: {test_folder}")
        
        self.test_folder = Path(test_folder)
        self.severity_mapping = severity_mapping
        
        # Load all test files
        all_files = sorted(list(self.test_folder.rglob("*.npz")))
        
        # Filter files based on mode
        self.samples = []
        self.filenames = []
        
        for fpath in all_files:
            filename = os.path.basename(str(fpath))
            
            # Check if filename exists in severity mapping
            if filename not in severity_mapping:
                print(f"Warning: {filename} not in severity mapping, skipping...")
                continue
            
            # If use_split_filter=True, only include files marked as 'test' split
            # If False, include ALL files in the folder (assumes folder already contains test split)
            if use_split_filter:
                if severity_mapping[filename]['split'] == 'test':
                    self.samples.append(str(fpath))
                    self.filenames.append(filename)
            else:
                self.samples.append(str(fpath))
                self.filenames.append(filename)
        
        print(f"Loaded {len(self.samples)} samples with severity mapping")
        print(f"   Mode: {'Using split filter (test only)' if use_split_filter else 'Using all files in folder'}")
        
        # Print severity distribution
        severity_counts = {}
        for fname in self.filenames:
            sev = severity_mapping[fname]['severity']
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        
        print(f"Test Set Severity Distribution:")
        for sev in ["Borderline", "Mild", "Moderate", "Severe"]:
            count = severity_counts.get(sev, 0)
            pct = 100 * count / len(self.samples) if self.samples else 0
            print(f"   {sev:<12}: {count:>4} samples ({pct:>5.1f}%)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        filename = self.filenames[idx]

        # Load data
        data = np.load(path, allow_pickle=True)
        td = data['td']
        rnfl = data['rnflt'].astype(np.float32)

        # Normalize RNFL if needed
        if rnfl.max() > 1.5:
            rnfl = np.clip(rnfl, 0, 350) / 350.0

        # Flip left eye RNFL (same as training: '_0_' in filename)
        is_left_eye = "_0_" in filename
        if is_left_eye:
            rnfl = np.fliplr(rnfl).copy()

        # Process VF
        vf_norm = normalize_vf(td)
        vf_grid = map_vf_to_8x9(vf_norm)
        vf_tensor = torch.tensor(vf_grid, dtype=torch.float32).unsqueeze(0)

        # Process RNFL
        rnfl_tensor = torch.tensor(rnfl, dtype=torch.float32).unsqueeze(0)

        # Get severity info
        severity_info = self.severity_mapping[filename]

        return vf_tensor, rnfl_tensor, filename, severity_info

# ============================================================================
# 4. EVALUATION
# ============================================================================

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load severity mapping
    print(f"Loading severity mapping from: {args.severity_mapping}")
    with open(args.severity_mapping, 'r') as f:
        severity_map = json.load(f)
    print(f"Loaded severity mapping for {len(severity_map)} samples\n")
    
    # Load dataset (use_split_filter=False means use ALL files in folder)
    dataset = TestDatasetWithSeverity(
        args.test_folder, 
        severity_map,
        use_split_filter=args.use_split_filter
    )
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=4
    )
    
    # Build model
    print(f"\nBuilding model with {args.skip_channels}-channel skip connections...")
    decoder = RNFL_Decoder(z_dim=args.z_dim, img_size=200).to(device)
    model = VF2RNFL_Model(
        decoder, 
        z_dim=args.z_dim, 
        dropout=0.0,
        img_size=200,
        skip_channels=args.skip_channels
    ).to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in ckpt: 
        model.load_state_dict(ckpt['model_state_dict'])
        if 'epoch' in ckpt:
            print(f"   Epoch: {ckpt['epoch']}")
        if 'val_mae' in ckpt:
            print(f"   Val MAE: {ckpt['val_mae']:.2f} µm")
        if 'val_r2' in ckpt:
            print(f"   Val R²: {ckpt['val_r2']:.4f}")
    else: 
        model.load_state_dict(ckpt)
    
    model.eval()
    
    # Mask
    mask = create_circular_mask(200, 200, radius=args.mask_radius)
    mask_bool = mask.astype(bool)
    
    # Storage for group analysis
    groups = {
        "All":        {"mae": [], "r2": [], "preds": [], "targs": []},
        "Borderline": {"mae": [], "r2": [], "preds": [], "targs": []},
        "Mild":       {"mae": [], "r2": [], "preds": [], "targs": []},
        "Moderate":   {"mae": [], "r2": [], "preds": [], "targs": []},
        "Severe":     {"mae": [], "r2": [], "preds": [], "targs": []},
    }
    
    print("\nRunning Inference...")
    print(f"   Dataset: {Path(args.test_folder).name}")
    print(f"   Model: {os.path.basename(args.checkpoint)}")
    print(f"   Mask Radius: {args.mask_radius}px\n")
    
    with torch.no_grad():
        for vf, gt, filenames, severity_info_batch in tqdm(loader, desc="Evaluating"):
            vf = vf.to(device)
            
            # Predict
            pred_norm = model(vf).cpu().squeeze(1).numpy()
            gt_norm = gt.squeeze(1).numpy()
            
            pred_um = denormalize_rnflt(pred_norm)
            gt_um = denormalize_rnflt(gt_norm)
            
            # Process batch
            batch_size = pred_um.shape[0]
            for i in range(batch_size):
                p = pred_um[i]
                t = gt_um[i]
                
                # Get pre-computed severity from mapping
                # severity_info_batch is a list of dicts
                severity = severity_info_batch['severity'][i]
                
                # Compute metrics (masked)
                p_valid = p[mask_bool]
                t_valid = t[mask_bool]
                
                mae = np.mean(np.abs(p_valid - t_valid))
                
                if np.var(t_valid) > 1e-5:
                    r2 = r2_score(t_valid, p_valid)
                else:
                    r2 = np.nan
                
                # Store in groups
                groups["All"]["mae"].append(mae)
                groups["All"]["r2"].append(r2)
                groups["All"]["preds"].append(p_valid)
                groups["All"]["targs"].append(t_valid)
                
                groups[severity]["mae"].append(mae)
                groups[severity]["r2"].append(r2)
                groups[severity]["preds"].append(p_valid)
                groups[severity]["targs"].append(t_valid)

    # ================= REPORTING =================
    print("\n" + "="*90)
    print(f"FINAL RESULTS: {os.path.basename(args.checkpoint)}")
    print(f"Dataset: {Path(args.test_folder).name}")
    print(f"Skip Channels: {args.skip_channels}")
    print(f"Severity based on: {os.path.basename(args.severity_mapping)}")
    print("="*90)
    print(f"{'GROUP':<15} | {'N_SAMPLES':<10} | {'MAE (µm)':<20} | {'AVG R²':<10} | {'GLOBAL R²':<10}")
    print("-" * 90)
    
    for g_name in ["All", "Borderline", "Mild", "Moderate", "Severe"]:
        metrics = groups[g_name]
        n = len(metrics["mae"])
        
        if n == 0:
            print(f"{g_name:<15} | {0:<10} | {'N/A':<20} | {'N/A':<10} | {'N/A':<10}")
            continue
            
        avg_mae = np.mean(metrics["mae"])
        std_mae = np.std(metrics["mae"])
        avg_r2 = np.nanmean(metrics["r2"])
        
        flat_p = np.concatenate(metrics["preds"])
        flat_t = np.concatenate(metrics["targs"])
        global_r2 = r2_score(flat_t, flat_p)
        
        print(f"{g_name:<15} | {n:<10} | {avg_mae:>7.2f} ± {std_mae:<7.2f} | {avg_r2:>8.4f}  | {global_r2:>8.4f}")
        
    print("="*90)
    
    # Additional stats
    all_metrics = groups["All"]
    if len(all_metrics["mae"]) > 0:
        print(f"\nAdditional Statistics:")
        print(f"   Min MAE:    {np.min(all_metrics['mae']):.2f} µm")
        print(f"   Max MAE:    {np.max(all_metrics['mae']):.2f} µm")
        print(f"   Median MAE: {np.median(all_metrics['mae']):.2f} µm")
        print(f"   25th %ile:  {np.percentile(all_metrics['mae'], 25):.2f} µm")
        print(f"   75th %ile:  {np.percentile(all_metrics['mae'], 75):.2f} µm")
    
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test VF→RNFL with unified severity mapping"
    )
    parser.add_argument(
        "--test_folder", 
        type=str, 
        required=True,
        help="Path to test data folder (any denoising method)"
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--severity_mapping",
        type=str,
        default="splits/severity_mapping.json",
        help="Path to severity mapping JSON file"
    )
    parser.add_argument(
        "--use_split_filter",
        action="store_true",
        help="If set, only test files marked as 'test' in severity mapping. "
             "If not set, test ALL files in test_folder (for pre-split test sets)"
    )
    parser.add_argument("--z_dim", type=int, default=32)
    parser.add_argument("--skip_channels", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--mask_radius", type=int, default=30)
    
    args = parser.parse_args()
    main(args)

"""
# ============================================================================
# COMPLETE WORKFLOW (For Already-Trained Models)
# ============================================================================

# STEP 1: Generate the severity mapping from the raw data
python unified_split_severity.py \
  --raw_folder $VF_DATA_ROOT/original_below350 \
  --output_dir ./splits \
  --seed 42

# STEP 2: Evaluate the trained models against the severity mapping
# The key: use --use_split_filter flag appropriately

# Option A: the test folder contains ONLY test-split samples (recommended)
# Just point to the test folder, don't filter by split
python test_unified_severity.py \
  --test_folder /path/to/test/folder \
  --checkpoint checkpoints/Skip16_NAFNet_best.pth \
  --severity_mapping splits/severity_mapping.json \
  --skip_channels 16
  # No --use_split_filter flag = uses ALL files in folder

# Option B: the test folder contains ALL samples (train+val+test mixed)
# Use the split filter to extract only test samples
python test_unified_severity.py \
  --test_folder $VF_DATA_ROOT/denoised_NAFNet_models/denoised_NAFNet_Advanced \
  --checkpoint checkpoints/Skip16_NAFNet_best.pth \
  --severity_mapping splits/severity_mapping.json \
  --skip_channels 16 \
  --use_split_filter  # This filters to 'test' split only

# STEP 3: Evaluate every model with the same severity grouping
DATASETS=(
  "original_below350"
  "denoised_n2n_FINAL"
  "denoised_NAFNet_models/denoised_NAFNet_Advanced"
  "denoised_n2v"
  "denoised_TwoStage_Final/denoised_cnn_ae"
)

for dataset in "${DATASETS[@]}"; do
  dataset_name=$(basename $dataset)
  echo "Testing: $dataset_name"
  
  python test_unified_severity.py \
    --test_folder $VF_DATA_ROOT/$dataset \
    --checkpoint checkpoints/Skip16_${dataset_name}_best.pth \
    --severity_mapping splits/severity_mapping.json \
    --skip_channels 16 \
    --use_split_filter
done

# KEY INSIGHT:
# Even though you trained with different random splits, the severity mapping
# ensures that a "Severe" case in NAFNet dataset is also "Severe" in Raw dataset,
# because severity is based on the FILENAME (which is the same across all datasets).
# 
# This allows fair comparison of how each denoising method performs on:
# - Borderline cases (MD >= -3 dB)
# - Mild cases (-6 <= MD < -3 dB)
# - Moderate cases (-12 <= MD < -6 dB)
# - Severe cases (MD < -12 dB)
"""