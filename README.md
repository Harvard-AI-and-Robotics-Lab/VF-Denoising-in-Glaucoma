# Developing and Evaluating Deep Learning Approaches for Visual Field Denoising in Glaucoma

This repository contains code for comparing nine deep learning methods for denoising Humphrey 24-2 visual fields (VF).

Each denoised VF is passed through a fixed VF to RNFL prediction model, and the predicted RNFL map is compared with the OCT RNFL map.

The methods include Noise2Noise, Noise2Void, NAFNet, MLP/CNN autoencoders and VAEs, and a positional encoding autoencoder. Raw visual fields are used as the baseline.

## Requirements

Python 3.8, PyTorch 2.4.1 and TensorFlow 2.6. The full specification is in
`environment.yml`.

```bash
conda env create -f environment.yml
conda activate py38
```

Weights and Biases is used only to record training curves. Set
`WANDB_MODE=disabled` to run without it.

## Configuration

Data locations are set using three environment variables.

```bash
cp config/env.example.sh config/env.sh
source config/env.sh
```

| Variable | Contents |
| --- | --- |
| `VF_DATA_ROOT` | paired VF and OCT samples, together with the denoised copies produced by each method |
| `VF_DENOISE_ROOT` | VF only training sets and denoiser checkpoints |
| `VF_SOURCE_ROOT` | raw clinical exports, required only by `data_preparation/` |

## Data format

Patient data is not distributed with this repository. Each sample is stored as
a single `.npz` file:

| Key | Shape | Description |
| --- | --- | --- |
| `td` | `(52,)` | total deviation, dB |
| `rnflt` | `(200, 200)` | OCT RNFL thickness, um |
| `meta` | JSON | id, eye, vf_time, oct_time, delta_days, md, vfi |

Denoised files keep the original filenames so that all methods are evaluated on the same samples.

## Repository structure

```
data_preparation/   data preprocessing and train/validation split
denoising/          NAFNet, AE, VAE, CNN, and PosEnc models
vf2rnfl/            VF to RNFL prediction, the primary evaluation
rnfl2vf/            RNFL to VF evaluation
external/           Noise2Noise and Noise2Void
config/             environment configuration
```

## Usage

### 1. Prepare the training data

```bash
python data_preparation/vf_dataset_generate.py
```

The notebooks in `data_preparation/` apply the reliability filters and create the training and validation sets.

### 2. Train and run the denoising models

| Method | Command |
| --- | --- |
| Raw | no denoising step |
| Noise2Noise | see `external/noise2noise/README.md` |
| Noise2Void | see `external/n2v/README.md` |
| NAFNet | `python denoising/train_nafnet.py`, then `python denoising/inference_nafnet.py` |
| AE, VAE, CNN-AE, CNN-VAE, PosEnc | `bash denoising/N3/run_train.sh`, then `python denoising/N3/inference.py` |

Denoised outputs are saved under `$VF_DATA_ROOT/denoised_*`.

### 3. VF to RNFL evaluation

```bash
cd vf2rnfl

python unified_split_severity.py \
  --raw_folder $VF_DATA_ROOT/original_below350 --output_dir ./splits --seed 42

python vf_to_rnfl_mask_disc.py --stage autoencoder \
  --folder $VF_DATA_ROOT/original_below350 \
  --z_dim 32 --mask_radius 30 --epochs 100 --batch_size 32 --lr 1e-4 \
  --run_name masked30_ae_z32

bash run_train_final.sh
bash run_test_severity.sh
```

This reports R2 and MAE overall and by disease severity. 

Training settings are defined in `run_train_final.sh`. The averaged VF control is run with `run_train_final_avg.sh` and `run_test_avg.sh`.

### 4. RNFL to VF evaluation

```bash
cd rnfl2vf
bash train_all.sh
bash run_sf_avg.sh
bash eval_oct2vf_avg_all.sh
python summarize_results.py
```

This evaluates the reverse prediction direction and reports MAE and RMSE.

### 5. Figures

`vf2rnfl/figures.ipynb` generates the VF and RNFL comparison panels.
`rnfl2vf/visualize_sf_predictions.py` and `rnfl2vf/visualize_vf.py` generate the RNFL to VF panels.

## Not included

Patient data, model checkpoints, training logs, and result files are not included. VF–OCT pairing step are also left out.

## Notes

Paths were updated to use environment variables, and notebook outputs and console logs were removed Noise2Noise outputs are saved to `denoised_n2n_FINAL` to match the evaluation scripts.

## Contact

Mengyu Wang, Ph.D., mengyu_wang@meei.harvard.edu
Schepens Eye Research Institute, Massachusetts Eye and Ear, Harvard Medical School.
Funded by NIH R01 EY036222 and P30 EY003790.