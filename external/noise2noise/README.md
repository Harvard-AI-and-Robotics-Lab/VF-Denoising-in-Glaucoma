# Noise2Noise for visual fields

Noise2Noise is used with one substantive modification: the convolutional
`autoencoder` in `network.py` is replaced by a fully connected network that
maps a 52-point total-deviation vector to a denoised 52-point vector. The
original RGB/NHWC version is kept in the patch, commented out, for reference.

Upstream: the fork https://github.com/juliaseungjoobaek/noise2noise (of
NVlabs/noise2noise), commit `7355519e7bfc49e0606cca8867748e736431244b`.
Licensed by NVIDIA under CC BY-NC 4.0; see the upstream `LICENSE.txt`.

## Setup

```bash
git clone https://github.com/juliaseungjoobaek/noise2noise.git
cd noise2noise
git checkout 7355519e7bfc49e0606cca8867748e736431244b
git apply /path/to/this/repo/external/noise2noise/upstream.patch
cp /path/to/this/repo/external/noise2noise/{train_vf.py,vf_inference.py,inference_n2n_corrected.py} .
```

The code runs on TensorFlow 2.6 through `tf.compat.v1` (`tf.disable_v2_behavior()`
plus a numpy alias shim at the top of each script).

## VF-specific scripts

| File | Purpose |
| --- | --- |
| `train_vf.py` | builds all ordered noisy/noisy permutations within each eye's test-retest series and trains the 52-dimensional denoiser |
| `vf_inference.py` | batch inference over the paired dataset |
| `inference_n2n_corrected.py` | inference with the corrected normalisation handling |

Training pairs come from eyes with >= 2 reliable tests inside a 90-day window
(28,863 samples), produced by `data_preparation/vf_dataset_generate.py`.

Both inference scripts write `$VF_DATA_ROOT/denoised_n2n_FINAL/`, the directory
the evaluation scripts read.
