"""
Unified Split & Severity Mapping Generator

Creates train/val/test splits and severity mappings ONCE from raw data,
then all denoising methods use the same splits.

Output:
  - data_splits.json: Contains train/val/test filename splits
  - severity_mapping.json: Contains MD and severity for each sample
"""
"""Usage:

    python unified_split_severity.py \
      --raw_folder $VF_DATA_ROOT/original_below350 \
      --output_dir ./splits \
      --seed 42 \
      --val_ratio 0.1 \
      --test_ratio 0.1

Writes splits/data_splits.json (train/val/test filename lists) and
splits/severity_mapping.json (MD and severity group per sample), which every
denoising method then reuses so that all methods are evaluated on identical
eyes.
"""
import os
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import argparse
from tqdm import tqdm

# ============================================================================
# UTILS
# ============================================================================

def load_paths_and_ids(folder):
    """Load all .npz paths and extract unique IDs"""
    all_paths = sorted(list(Path(folder).rglob("*.npz")))
    all_ids = []
    id_to_path = {}
    
    for p in all_paths:
        try:
            data = np.load(p, allow_pickle=True)
            meta_raw = data['meta'].item()
            meta = json.loads(meta_raw.decode('utf-8'))
            full_id = f"{meta['id']}_{meta['eye']}_{meta['oct_time']}"
            
            # Also store the filename (without path) for universal matching
            filename = os.path.basename(str(p))
            
            all_ids.append(full_id)
            id_to_path[full_id] = {
                'full_path': str(p),
                'filename': filename
            }
        except Exception as e:
            print(f"Failed to read {p.name}: {e}")
            continue
    
    return all_ids, id_to_path

def calculate_md(td_values):
    """
    Calculate Mean Deviation (MD) from Total Deviation values
    MD ≈ mean of the 52 TD points
    """
    return np.mean(td_values)

def assign_severity_group(md):
    """
    Assign severity group based on MD value
    
    Groups:
    - Borderline: MD >= -3 dB
    - Mild:       -6 <= MD < -3 dB  
    - Moderate:   -12 <= MD < -6 dB
    - Severe:     MD < -12 dB
    """
    if md >= -3.0:
        return "Borderline"
    elif md >= -6.0:
        return "Mild"
    elif md >= -12.0:
        return "Moderate"
    else:
        return "Severe"

# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def generate_splits_and_severity(args):
    """
    Generate train/val/test splits and severity mapping from raw data
    """
    print("="*80)
    print("UNIFIED SPLIT & SEVERITY MAPPING GENERATOR")
    print("="*80)
    print(f"Raw data folder: {args.raw_folder}")
    print(f"Random seed: {args.seed}")
    print(f"Split ratio - Train: {1-args.val_ratio-args.test_ratio:.0%} | "
          f"Val: {args.val_ratio:.0%} | Test: {args.test_ratio:.0%}")
    print()
    
    # 1. Load all files from raw data
    print("Loading raw data files...")
    all_ids, id_to_path = load_paths_and_ids(args.raw_folder)
    print(f"Found {len(all_ids)} samples")
    
    if len(all_ids) == 0:
        print("No valid samples found!")
        return
    
    # 2. Create train/val/test splits
    print(f"\nCreating splits (seed={args.seed})...")
    
    # First split: train vs (val+test)
    train_ids, temp_ids = train_test_split(
        all_ids, 
        test_size=(args.val_ratio + args.test_ratio),
        random_state=args.seed
    )
    
    # Second split: val vs test
    val_ratio_adjusted = args.val_ratio / (args.val_ratio + args.test_ratio)
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=(1 - val_ratio_adjusted),
        random_state=args.seed
    )
    
    print(f"   Train: {len(train_ids)} samples")
    print(f"   Val:   {len(val_ids)} samples")
    print(f"   Test:  {len(test_ids)} samples")
    
    # 3. Calculate severity for each sample
    print(f"\nCalculating MD and severity groups...")
    severity_mapping = {}
    severity_counts = {"Borderline": 0, "Mild": 0, "Moderate": 0, "Severe": 0}
    
    for sample_id in tqdm(all_ids, desc="Processing"):
        try:
            path = id_to_path[sample_id]['full_path']
            filename = id_to_path[sample_id]['filename']
            
            data = np.load(path, allow_pickle=True)
            td = data['td']  # Total Deviation values
            
            # Calculate MD
            md = calculate_md(td)
            
            # Assign severity
            severity = assign_severity_group(md)
            severity_counts[severity] += 1
            
            # Store mapping (use filename as key for universal matching)
            severity_mapping[filename] = {
                'full_id': sample_id,
                'md': float(md),
                'severity': severity,
                'split': 'train' if sample_id in train_ids else ('val' if sample_id in val_ids else 'test')
            }
            
        except Exception as e:
            print(f"Failed to process {sample_id}: {e}")
            continue
    
    print(f"\nSeverity distribution:")
    for severity, count in severity_counts.items():
        print(f"   {severity:<12}: {count:>4} samples ({100*count/len(all_ids):>5.1f}%)")
    
    # 4. Save splits (as filenames for universal matching)
    splits = {
        'train': [id_to_path[i]['filename'] for i in train_ids if i in id_to_path],
        'val': [id_to_path[i]['filename'] for i in val_ids if i in id_to_path],
        'test': [id_to_path[i]['filename'] for i in test_ids if i in id_to_path],
        'metadata': {
            'raw_folder': args.raw_folder,
            'seed': args.seed,
            'total_samples': len(all_ids),
            'train_samples': len(train_ids),
            'val_samples': len(val_ids),
            'test_samples': len(test_ids),
        }
    }
    
    splits_file = args.output_dir / 'data_splits.json'
    with open(splits_file, 'w') as f:
        json.dump(splits, f, indent=2)
    print(f"\nSaved splits to: {splits_file}")
    
    # 5. Save severity mapping
    severity_file = args.output_dir / 'severity_mapping.json'
    with open(severity_file, 'w') as f:
        json.dump(severity_mapping, f, indent=2)
    print(f"Saved severity mapping to: {severity_file}")
    
    # 6. Print per-split severity distribution
    print(f"\nSeverity distribution per split:")
    print("-" * 80)
    print(f"{'Split':<10} | {'Borderline':<12} | {'Mild':<12} | {'Moderate':<12} | {'Severe':<12}")
    print("-" * 80)
    
    for split_name in ['train', 'val', 'test']:
        split_files = splits[split_name]
        split_severities = [severity_mapping[f]['severity'] for f in split_files if f in severity_mapping]
        
        counts = {s: split_severities.count(s) for s in ["Borderline", "Mild", "Moderate", "Severe"]}
        total = len(split_files)
        
        print(f"{split_name:<10} | "
              f"{counts['Borderline']:>4} ({100*counts['Borderline']/total:>5.1f}%) | "
              f"{counts['Mild']:>4} ({100*counts['Mild']/total:>5.1f}%) | "
              f"{counts['Moderate']:>4} ({100*counts['Moderate']/total:>5.1f}%) | "
              f"{counts['Severe']:>4} ({100*counts['Severe']/total:>5.1f}%)")
    
    print("-" * 80)
    print()
    print("Done! Use these files for ALL denoising methods.")
    print()

# ============================================================================
# HELPER: Get paths for a specific denoising method
# ============================================================================

def get_paths_for_dataset(splits_file, dataset_folder):
    """
    Helper function to get train/val/test paths for any denoising dataset
    using the universal splits.
    
    Usage in training scripts:
        train_paths = get_paths_for_dataset('data_splits.json', '/path/to/denoised_NAFNet')
    """
    with open(splits_file, 'r') as f:
        splits = json.load(f)
    
    dataset_path = Path(dataset_folder)
    
    result = {}
    for split_name in ['train', 'val', 'test']:
        filenames = splits[split_name]
        
        # Find matching files in the dataset folder
        paths = []
        for fname in filenames:
            matching = list(dataset_path.rglob(fname))
            if matching:
                paths.append(str(matching[0]))
            else:
                print(f"File not found in {dataset_folder}: {fname}")
        
        result[split_name] = paths
        print(f"{split_name:<5}: {len(paths)} files found in {dataset_path.name}")
    
    return result

# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate unified train/val/test splits and severity mapping from raw data"
    )
    
    parser.add_argument(
        "--raw_folder", 
        type=str, 
        default=str(Path(os.environ.get("VF_DATA_ROOT", "/path/to/vf_oct_pairs"))
                    / "original_below350"),
        help="Path to raw (original) data folder"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./splits",
        help="Output directory for splits and severity mapping"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits"
    )
    
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation set ratio (default: 0.1 = 10%)"
    )
    
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Test set ratio (default: 0.1 = 10%)"
    )
    
    args = parser.parse_args()
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(exist_ok=True, parents=True)
    
    generate_splits_and_severity(args)

"""
# ============================================================================
# USAGE
# ============================================================================

# 1. Generate splits and severity mapping ONCE from raw data
python unified_split_severity.py \
  --raw_folder $VF_DATA_ROOT/original_below350 \
  --output_dir ./splits \
  --seed 42 \
  --val_ratio 0.1 \
  --test_ratio 0.1

# This creates:
#   splits/data_splits.json      - train/val/test filename lists
#   splits/severity_mapping.json - MD and severity for each file

# 2. In a training script, load the splits for any denoised dataset:

from pathlib import Path
import json

def load_dataset_splits(splits_file, dataset_folder):
    with open(splits_file, 'r') as f:
        splits = json.load(f)
    
    dataset_path = Path(dataset_folder)
    result = {}
    
    for split_name in ['train', 'val', 'test']:
        filenames = splits[split_name]
        paths = []
        for fname in filenames:
            matching = list(dataset_path.rglob(fname))
            if matching:
                paths.append(str(matching[0]))
        result[split_name] = paths
    
    return result

# Example usage in training:
splits = load_dataset_splits('splits/data_splits.json', '$VF_DATA_ROOT/denoised_NAFNet_models/denoised_NAFNet_Advanced')
train_paths = splits['train']
val_paths = splits['val']
test_paths = splits['test']

# 3. In a test script, load the severity mapping:

with open('splits/severity_mapping.json', 'r') as f:
    severity_map = json.load(f)

# Get severity for a file:
filename = '<patientid>_<eye>_<timestamp>.npz'
severity_info = severity_map[filename]
print(f"MD: {severity_info['md']:.2f} dB")
print(f"Severity: {severity_info['severity']}")
print(f"Split: {severity_info['split']}")
"""