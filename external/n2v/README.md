# Noise2Void for visual fields

Noise2Void is used unmodified except for two compatibility fixes. Rather than
redistributing the upstream source, this directory holds the patch and the
VF-specific scripts.

Upstream: https://github.com/juglab/n2v, commit `c61b3b47c5cacc533f7e0e3db7cefab139d7b62a`
(BSD 3-Clause; see the upstream `LICENSE.txt`).

## Setup

```bash
git clone https://github.com/juglab/n2v.git
cd n2v
git checkout c61b3b47c5cacc533f7e0e3db7cefab139d7b62a
git apply /path/to/this/repo/external/n2v/upstream.patch
pip install -e .
cp -r /path/to/this/repo/external/n2v/scripts ./scripts
```

`upstream.patch` contains only compatibility fixes for newer numpy/Keras:
`np.product` becomes `np.prod`, and the `weights_*.h5` becomes `weights_*.weights.h5`
checkpoint naming required by Keras 3.

## VF-specific scripts

| File | Purpose |
| --- | --- |
| `scripts/train_vf.ipynb` | trains N2V on the VF-only cohort; each 24-2 field is mapped onto an 8x9 grid (20 invalid corner cells left empty), model name `n2v_vf_8x9` |
| `scripts/inference_vf.py` | applies the trained model to every paired sample and writes `$VF_DATA_ROOT/denoised_n2v/` |

N2V needs no test-retest pairing, so it is trained on all 394,083 reliable
single tests.

Run from inside the cloned repo, with `VF_DATA_ROOT` and `VF_DENOISE_ROOT` set
(see `config/env.example.sh` in the repository root).
