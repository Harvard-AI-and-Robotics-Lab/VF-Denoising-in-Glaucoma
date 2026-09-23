"""Build the Noise2Noise / test-retest training set from the VF-only cohort.

Selects eyes with at least N reliable Humphrey 24-2 examinations inside a
100-day window and writes them as a single .npz of total-deviation vectors.

Usage:
    python vf_dataset_generate.py

Input   : $VF_SOURCE_ROOT (see source_data_config.py)
Output  : $VF_DENOISE_ROOT/dataset_train_val_test/N2N/fine_tune_dataset_Nge2.npz
"""


import sys
import os
import numpy as np
import pandas as pd
import pyreadr
import itertools

# ================= CONFIGURATION =================
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from source_data_config import train_boland

# OUTPUT PATH (see config/env.example.sh)
DENOISE_ROOT = os.environ.get("VF_DENOISE_ROOT", "/path/to/denoising_workdir")
PATH_DIR = os.path.join(DENOISE_ROOT, "dataset_train_val_test", "N2N")
os.makedirs(PATH_DIR, exist_ok=True)
SAVE_PATH = os.path.join(PATH_DIR, "fine_tune_dataset_Nge2.npz")

N = 3  # Strictly 3 images per cluster
# =================================================

print("Loading data...")
vf24 = pyreadr.read_r(train_boland)
vf24_pd = vf24['bolandSS24.20.20.33']

# 1. QUALITY FILTERING
criteria = (vf24_pd['malfixrate'] <= 0.33) & \
           (vf24_pd['falsenegrate'] <= 0.2) & \
           (vf24_pd['falseposrate'] <= 0.2) & \
           (vf24_pd['site'] != 5)
filtered_df = vf24_pd[criteria].copy()

# Create Identifiers
filtered_df['test_id'] = filtered_df['id'].astype(str) + '_' + \
                         filtered_df['righteye'].astype(str) + '_' + \
                         filtered_df['testdate'].astype(str)
filtered_df['eyeid'] = filtered_df['id'].astype(str) + "_" + filtered_df['righteye'].astype(str)

# 2. REMOVE DUPLICATES
td_cols = [f'td{i}' for i in range(1, 55) if i not in (26, 35)]
key_cols = ['id', 'righteye', 'age', 'duration'] + td_cols  
vffile = filtered_df.drop_duplicates(subset=key_cols)
print(f"Data cleaned. {vffile.shape[0]} tests remaining.")

# 3. IDENTIFY VALID SERIES (Window <= 100 days, N=3)
print("Identifying valid series in 100-day window...")
counts = vffile.groupby('eyeid').size()
eyes_possible = counts[counts >= N].index 
df_multi = vffile[vffile['eyeid'].isin(eyes_possible)].copy()

valid_eyes_map = {} 

for eyeid, group in df_multi.groupby('eyeid'):
    group_sorted = group.sort_values('testdate')
    all_test_ids = group_sorted['test_id'].tolist()
    all_dates = group_sorted['testdate'].tolist()
    n_total = len(all_test_ids)

    found = False
    # Search for largest cluster fitting in 100 days
    for k in range(n_total, 1, -1): 
        for i in range(n_total - k + 1):
            sub_dates = all_dates[i:i+k]
            duration = sub_dates[-1] - sub_dates[0]
            
            if duration <= 100:
                valid_eyes_map[eyeid] = all_test_ids[i:i+k]
                found = True
                break
        if found: break

# 4. GENERATE N2N PAIRS (Strictly N=3)
print(f"Generating Training Pairs (Strictly N={N})...")

X_raw, Y_raw = [], []
id_arr, cid_arr = [], []
malfix_arr, fn_arr, fp_arr = [], [], []

for eyeid, test_ids in valid_eyes_map.items():
    # Strict N=3 Filter (Same as notebook)
    if len(test_ids) != N:
        continue
        
    subset = vffile[vffile['test_id'].isin(test_ids)]
    td_array = subset[td_cols].astype(np.float32).values # Shape: (3, 52)
    
    # Generate Permutations for Noise2Noise Training
    # (A->B, A->C, B->A, B->C, C->A, C->B)
    indices = range(len(td_array))
    pairs = list(itertools.permutations(indices, 2))

    for src_i, tgt_i in pairs:
        # Input (Noisy)
        X_raw.append(td_array[src_i])
        
        # Target (Noisy Pair - N2N Requirement)
        Y_raw.append(td_array[tgt_i]) 

        # Metadata
        row = subset.iloc[src_i]
        id_arr.append(row['id'])
        cid_arr.append(f"{row['id']}_{row['righteye']}_{row['testdate']}")
        malfix_arr.append(row['malfixrate'])
        fn_arr.append(row['falsenegrate'])
        fp_arr.append(row['falseposrate'])

# 5. SAVE RESULT
if len(X_raw) > 0:
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    
    np.savez_compressed(
        SAVE_PATH,
        td=np.stack(X_raw), 
        td_target=np.stack(Y_raw), 
        ids=np.array(id_arr), 
        cids=np.array(cid_arr),
        malfix=np.array(malfix_arr, dtype=np.float32), 
        fnrate=np.array(fn_arr, dtype=np.float32), 
        fprate=np.array(fp_arr, dtype=np.float32)
    )

    print(f"Saved TRAIN Dataset: {SAVE_PATH}")
    print(f"Total Samples: {len(X_raw)}")
else:
    print("No data found matching criteria!")