import numpy as np
if not hasattr(np, "object"):  np.object  = object
if not hasattr(np, "bool"):    np.bool    = bool
if not hasattr(np, "int"):     np.int     = int
if not hasattr(np, "float"):   np.float   = float
if not hasattr(np, "complex"): np.complex = complex

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()          # disables eager, restores TF1 symbols
import os, sys
sys.modules["tensorflow"] = tf    # IMPORTANT: makes later `import tensorflow as tf` inside libs use compat.v1

from tqdm import trange
import dnnlib
import dnnlib.tflib as tflib
import dnnlib.tflib.tfutil as tfutil
import config
from util import save_snapshot

TRAIN_NPZ_PATH = os.path.expandvars(
    "$VF_DENOISE_ROOT/dataset_train_val_test/N2N/fine_tune_dataset_Nge2.npz")

def generate_n2n_pairs(flat_data, flat_cids):
    """
    Groups data by 'cids' (eye ID) and generates all valid permutations
    for Noise2Noise training.
    
    For a group of size k:
      - Creates k * (k-1) pairs.
      - Input: Test_i, Target: Test_j (where i != j).
    """
    from collections import defaultdict
    print("Grouping data by Patient/Eye ID...")
    
    # 1. Group indices by CID
    groups = defaultdict(list)
    for idx, cid in enumerate(flat_cids):
        groups[cid].append(idx)
        
    inputs = []
    targets = []
    
    # 2. Generate permutations
    # If an eye has tests [A, B, C], we generate:
    # (A,B), (A,C), (B,A), (B,C), (C,A), (C,B)
    for cid, indices in groups.items():
        if len(indices) < 2:
            continue # Skip eyes with only 1 test (cannot do N2N)
            
        # Create all permutations for this group
        # This is efficient for small k (VF series are usually k=2 to ~10)
        for i in indices:
            for j in indices:
                if i == j: 
                    continue
                inputs.append(flat_data[i])
                targets.append(flat_data[j])
                
    inputs = np.array(inputs, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)
    
    print(f"Generated {len(inputs)} training pairs from {len(flat_data)} raw samples.")
    return inputs, targets

def load_custom_data(batch_size):
    # 1. Load the raw data
    # We assume 'td' is all VFs and 'cids' are the identifiers
    data = np.load(TRAIN_NPZ_PATH)
    
    # Check keys to ensure compatibility
    if 'cids' not in data:
        raise ValueError("The .npz file must contain a 'cids' array to group tests by eye.")
        
    raw_td = data["td"].astype(np.float32)
    raw_cids = data["cids"] # ID string or int

    # 2. Generate Pairs on the fly (RAM is usually fine for VF data)
    train_input, train_target = generate_n2n_pairs(raw_td, raw_cids)

    # 3. Create TF Dataset
    dataset = tf.data.Dataset.from_tensor_slices((train_input, train_target))
    dataset = dataset.shuffle(buffer_size=min(50000, train_input.shape[0]))
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size, drop_remainder=True)

    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    input_batch, target_batch = iterator.get_next()
    
    # Return 3 items to match the expected signature: (input, target, clean_ref)
    # In N2N, we don't use clean_ref for loss, but we pass target again to satisfy the signature.
    return input_batch, target_batch, target_batch


def compute_ramped_down_lrate(i, iteration_count, ramp_down_perc, learning_rate):
    ramp_down_start_iter = iteration_count * (1 - ramp_down_perc)
    if i >= ramp_down_start_iter:
        t = ((i - ramp_down_start_iter) / ramp_down_perc) / iteration_count
        smooth = (0.5 + np.cos(t * np.pi) / 2) ** 2
        return learning_rate * smooth
    return learning_rate

def train_custom():
    iteration_count = 500000
    eval_interval   = 1000
    minibatch_size  = 256
    learning_rate   = 1e-3
    ramp_down_perc  = 0.3
    noise2noise     = True

    submit_config = dnnlib.SubmitConfig()
    submit_config.run_dir  = "results/vf-mlp-n2n-dynamic"
    submit_config.run_id   = 0
    submit_config.num_gpus = 1
    os.makedirs(submit_config.run_dir, exist_ok=True)

    ctx = dnnlib.RunContext(submit_config, config)
    tfutil.init_tf(config.tf_config)

    lrate_in = tf.placeholder(tf.float32, name="lrate_in", shape=[])
    
    # This now calls the updated loader
    noisy_input, noisy_target, _ = load_custom_data(minibatch_size)

    print("Building TensorFlow graph...")
    net = tflib.Network(**config.net_config)
    net.print_layers()

    opt = tflib.Optimizer(learning_rate=lrate_in, **config.optimizer_config)

    denoised = net.get_output_for(noisy_input)
    
    # N2N Loss: Minimize distance between Denoised(Input) and Noisy(Target)
    if noise2noise:
        meansq_error = tf.reduce_mean(tf.square(noisy_target - denoised))
    else:
        # Fallback if you ever have clean data (unlikely here)
        clean_target = noisy_target 
        meansq_error = tf.reduce_mean(tf.square(clean_target - denoised))

    opt.register_gradients(meansq_error, net.trainables)
    train_step = opt.apply_updates()

    print("Training...")
    pbar = trange(iteration_count, desc="Training", unit="iter")
    for i in pbar:
        if ctx.should_stop():
            break

        lrate = compute_ramped_down_lrate(i, iteration_count, ramp_down_perc, learning_rate)
        tfutil.run([train_step], {lrate_in: lrate})

        if i > 0 and i % eval_interval == 0:
            loss_val = tfutil.run(meansq_error)
            pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{lrate:.2e}")
            save_snapshot(submit_config, net, f"{i:06d}")

    save_snapshot(submit_config, net, "final")
    ctx.close()

if __name__ == "__main__":
    train_custom()
