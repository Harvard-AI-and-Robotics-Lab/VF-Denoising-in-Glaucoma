import os
import sys
import pickle
import numpy as np

# --- numpy compat patches (old TF / dnnlib code sometimes expects these) ---
if not hasattr(np, "object"):  np.object  = object
if not hasattr(np, "bool"):    np.bool    = bool
if not hasattr(np, "int"):     np.int     = int
if not hasattr(np, "float"):   np.float   = float
if not hasattr(np, "complex"): np.complex = complex

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
sys.modules["tensorflow"] = tf  # ensure internal libs import TF1 compat

import dnnlib.tflib.tfutil as tfutil

# ----------------- paths -----------------
SNAPSHOT_PATH = "results/vf-mlp-n2n-dynamic/network_final.pickle"
# Data locations come from the environment (see config/env.example.sh)
_DATA_ROOT    = os.environ.get("VF_DATA_ROOT", "/path/to/vf_oct_pairs")
INPUT_ROOT    = os.path.join(_DATA_ROOT, "original_below350")
# NOTE: the original scripts wrote to "denoised_n2n"; the directory was renamed
#       to "denoised_n2n_FINAL" before evaluation, which is the name the
#       structure-function scripts expect.
OUTPUT_ROOT   = os.path.join(_DATA_ROOT, "denoised_n2n_FINAL")
BATCH_SIZE    = 256

# If you still have config.tf_config as a tf.ConfigProto, DO NOT pass it to tfutil.init_tf.
# Pass a dict instead (this avoids the `.items()` crash).
TFUTIL_CONFIG = {
    "allow_soft_placement": True,
    "gpu_options.allow_growth": True,
}

# ----------------- helpers -----------------
def load_net(snapshot_path: str):
    tfutil.init_tf(TFUTIL_CONFIG)

    with open(snapshot_path, "rb") as f:
        snapshot = pickle.load(f)

    # Common snapshot formats
    if isinstance(snapshot, dict):
        for k in ["net", "G", "network", "model"]:
            if k in snapshot:
                return snapshot[k]
        for v in snapshot.values():
            if hasattr(v, "run") and hasattr(v, "input_shapes"):
                return v

    if hasattr(snapshot, "run"):
        return snapshot

    raise ValueError("Could not locate network inside snapshot.")

def find_npz_files(root: str):
    out = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".npz"):
                out.append(os.path.join(dirpath, fn))
    return sorted(out)

EXPECTED_DIM = 52          # model was trained on 52-dim VF TD vectors
TD_RANGE     = (-40, 20)   # plausible TD value range (dB)

def pick_input_array(npz):
    """
    The model was trained on VF vectors (typically shape (N,52) or (52,)).
    Samples store the TD vector under the key 'td'; fall back to alternatives.
    """
    for key in ["td", "vf", "x", "noisy", "data", "inputs"]:
        if key in npz.files:
            arr = npz[key]
            return key, arr
    raise KeyError(f"No suitable VF key found. Available keys: {npz.files}")

def validate_input(arr: np.ndarray, path: str, key: str):
    """
    Verify that the array is compatible with the trained N2N model:
      1. Feature dimension must be EXPECTED_DIM (52)
      2. Values should fall within a plausible TD range
    Raises ValueError with a descriptive message on failure.
    """
    if arr.ndim == 1:
        feat_dim = arr.shape[0]
    elif arr.ndim == 2:
        feat_dim = arr.shape[1]
    else:
        raise ValueError(
            f"{path}: expected 1-D or 2-D array for key '{key}', "
            f"got shape {arr.shape}"
        )

    if feat_dim != EXPECTED_DIM:
        raise ValueError(
            f"{path}: feature dim = {feat_dim}, expected {EXPECTED_DIM}. "
            f"Key '{key}' may not be VF TD data."
        )

    lo, hi = TD_RANGE
    vmin, vmax = float(np.nanmin(arr)), float(np.nanmax(arr))
    if vmin < lo or vmax > hi:
        print(
            f"  {path}: values [{vmin:.1f}, {vmax:.1f}] outside "
            f"expected TD range {TD_RANGE} — results may be unreliable"
        )

def ensure_2d_float32(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[None, :]
    return x.astype(np.float32)

def denoise_array(net, noisy_2d: np.ndarray, batch_size: int) -> np.ndarray:
    den = np.empty_like(noisy_2d, dtype=np.float32)
    n = noisy_2d.shape[0]
    for i in range(0, n, batch_size):
        batch = noisy_2d[i:i+batch_size]
        # net.run expects (N,52) float32 in this pipeline
        den[i:i+len(batch)] = net.run(batch)
    return den

def save_npz_with_denoised(in_path: str, out_root: str, in_root: str, payload: dict):
    rel = os.path.relpath(in_path, in_root)
    out_path = os.path.join(out_root, rel)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, **payload)
    return out_path

# ----------------- main -----------------
def main():
    print("Starting N2N Inference (folder mode)")
    print("   Model :", SNAPSHOT_PATH)
    print("   Input :", INPUT_ROOT)
    print("   Output:", OUTPUT_ROOT)

    net = load_net(SNAPSHOT_PATH)
    npz_paths = find_npz_files(INPUT_ROOT)
    if len(npz_paths) == 0:
        raise RuntimeError(f"No .npz files found under {INPUT_ROOT}")

    print(f"   Found {len(npz_paths)} npz files")

    # --- quick sanity check on the first file ---
    first = np.load(npz_paths[0], allow_pickle=True)
    print(f"   First file keys: {first.files}")
    fk, farr = pick_input_array(first)
    validate_input(farr, npz_paths[0], fk)
    print(f"   First file: key='{fk}', shape={farr.shape}, "
          f"range=[{np.nanmin(farr):.1f}, {np.nanmax(farr):.1f}]")
    print(f"   Data looks compatible with {EXPECTED_DIM}-dim VF TD model")

    for idx, path in enumerate(npz_paths, 1):
        try:
            data = np.load(path, allow_pickle=True)
            key, noisy = pick_input_array(data)
            validate_input(noisy, path, key)
            noisy2d = ensure_2d_float32(noisy)

            denoised = denoise_array(net, noisy2d, BATCH_SIZE)

            # Build payload: keep everything, add 'denoised', and also store the original noisy under 'noisy'
            payload = {k: data[k] for k in data.files}
            payload["noisy"] = noisy2d
            payload["denoised"] = denoised
            payload["_n2n_input_key"] = np.array(key)

            out_path = save_npz_with_denoised(path, OUTPUT_ROOT, INPUT_ROOT, payload)
            print(f"[{idx:>4}/{len(npz_paths)}] {path} -> {out_path} (key={key}, shape={noisy2d.shape})")

        except Exception as e:
            print(f"[{idx:>4}/{len(npz_paths)}] Failed on {path}: {repr(e)}")

    print("Done.")

if __name__ == "__main__":
    main()

"""
run:
nohup python3 vf_inference.py > vf_inference_Nge1.log 2>&1 &

"""