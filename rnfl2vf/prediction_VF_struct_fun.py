# %%could be good to add a tanh activation function to the output layer for the vf prediction as they are normallyzed between -1 and 1


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
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.models import efficientnet_b0
try:
    from torchvision.models import EfficientNet_B0_Weights
    Weights = EfficientNet_B0_Weights
except ImportError:
    from torchvision.models import efficientnet_b0 as _unused
    Weights = None  # fallback
from sklearn.metrics import r2_score, mean_absolute_error
import wandb
from torchvision.models import efficientnet_b0
# === Normalisation ===
def normalize_rnflt(x):
    x = np.clip(x, 0, 350)
    return (x / 350.0).astype(np.float32)

def normalize_vf(x, min_val=-38.0, max_val=26.0):
    return 2 * (x - min_val) / (max_val - min_val) - 1

# === Dataset ===
class RNFLTVFDataset(Dataset):
    def __init__(self, file_paths, min_val=-38.0, max_val=26.0):
        self.samples = []
        for path in file_paths:
            try:
                data = np.load(path, allow_pickle=True)
                self.samples.append((normalize_rnflt(data['rnflt']),
                                     normalize_vf(data['td'], min_val, max_val)))
            except Exception as e:
                print(f"Failed to read {Path(path).name}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rnflt, td = self.samples[idx]
        return torch.tensor(rnflt).unsqueeze(0), torch.tensor(td)

# === Modèles ===
class VGG_RNFLT(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.vgg11(pretrained=True).features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 52)
        )

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.backbone(x)
        x = self.pool(x)
        return self.head(x)


class EfficientNet_RNFLT(nn.Module):
    def __init__(self):
        super().__init__()

        # Charge le backbone pré-entraîné
        if Weights is None:
            # ancienne API torchvision (<0.13)
            self.backbone = efficientnet_b0(pretrained=True)
        else:
            self.backbone = efficientnet_b0(weights=Weights.IMAGENET1K_V1)

        # --- Patch 1-canal ---
        first_conv = self.backbone.features[0][0]            # Conv2d d'origine (3→32)
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False
        )
        # Poids = moyenne des 3 canaux RGB
        with torch.no_grad():
            new_conv.weight.copy_(first_conv.weight.mean(dim=1, keepdim=True))
        self.backbone.features[0][0] = new_conv
        # ---------------------

        # Remplace la tête
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 52)
        )

    def forward(self, x):
        # plus besoin de répéter x ; c’est déjà 1-canal
        return self.backbone(x)


# === Load paths ===
def load_filtered_paths(folder, id_set):
    paths = []
    for p in Path(folder).rglob("*.npz"):
        try:
            data = np.load(p, allow_pickle=True)
            meta_raw = data['meta'].item()
            meta = json.loads(meta_raw.decode('utf-8'))
            full_id = f"{meta['id']}_{meta['eye']}_{meta['oct_time']}"
            if full_id in id_set:
                paths.append(str(p))
        except Exception as e:
            print(f"Failed to read {p.name}: {e}")
    return paths

# === Entraînement ===
def train_model(train_loader, val_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "vgg":
        model = VGG_RNFLT().to(device)
    elif args.model == "efficientnet":
        model = EfficientNet_RNFLT().to(device)
    else:
        raise ValueError("Modèle non reconnu. Utilise 'vgg' ou 'efficientnet'.")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()

    wandb.init(project=args.project, name=args.run_name, config=vars(args))

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)

        avg_loss = running_loss / len(train_loader.dataset)

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                pred = model(x).cpu().numpy()
                preds.append(pred)
                targets.append(y.numpy())

        preds = np.vstack(preds)
        targets = np.vstack(targets)
        r2 = r2_score(targets, preds)
        mae = mean_absolute_error(targets, preds)

        wandb.log({"epoch": epoch, "train_loss": avg_loss, "val_r2": r2, "val_mae": mae})
        print(f"Epoch {epoch}: Train Loss={avg_loss:.4f} | R2={r2:.3f} | MAE={mae:.3f}")

    # Sauvegarde propre
    ckpt_folder = Path("checkpoints")
    ckpt_folder.mkdir(exist_ok=True)
    ckpt_path = ckpt_folder / f"{args.run_name}.pth"               
    torch.save(model.state_dict(), ckpt_path)
    print(f"Modèle sauvegardé dans {ckpt_path}")

    wandb.finish()

# === Script principal ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, required=True, help="Folder with npz files")
    parser.add_argument('--split_seed', type=int, default=42)
    parser.add_argument('--split_ratio', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--project', type=str, default="RNFLT_to_VF_rnflt_filtered_below_350")
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--model', type=str, choices=["vgg", "efficientnet"], default="vgg")
    parser.add_argument('--data_type', type=str, required=True)

    args = parser.parse_args()

    # Auto-naming du run
    if args.run_name is None:
        args.run_name = f"{args.model}_{args.data_type}_lr{args.lr}_bs{args.batch_size}"

    # Chargement des fichiers
    all_paths = sorted(list(Path(args.folder).rglob("*.npz")))
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
            print(f"Failed to read {p.name}: {e}")

    train_ids, val_ids = train_test_split(all_ids, test_size=args.split_ratio, random_state=args.split_seed)
    train_paths = [id_to_path[i] for i in train_ids if i in id_to_path]
    val_paths = [id_to_path[i] for i in val_ids if i in id_to_path]

    print(f"{len(train_paths)} fichiers d'entraînement, {len(val_paths)} pour validation.")

    # Datasets & loaders
    train_set = RNFLTVFDataset(train_paths)
    val_set = RNFLTVFDataset(val_paths)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    # Launch training
    train_model(train_loader, val_loader, args)
# CUDA_VISIBLE_DEVICES=1 python prediction_VF_struct_fun.py \
#   --folder $VF_DATA_ROOT/denoised_TwoStage_Final/denoised_mlp_ae \
#   --data_type denoised \
#   --model efficientnet \
#   --project RNFLT_to_VF \
#   --split_ratio 0.1 \
#   --epochs 30

# CUDA_VISIBLE_DEVICES=1 python prediction_VF_struct_fun.py \
#   --folder $VF_DATA_ROOT/original_below350 \
#   --data_type "random"\
#   --model "efficientnet" \
# --split_ratio 0.1 --split_seed 1337
