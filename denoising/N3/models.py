import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# ==============================================================================
# 1. MLP MODELS (Corrected to 52 -> 38 -> 26 -> z)
# ==============================================================================
class MLP_AE(nn.Module):
    def __init__(self, input_dim=52, latent_dim=8):
        super().__init__()
        # Encoder: 52 -> 38 -> 26 -> z
        self.enc = nn.Sequential(
            nn.Linear(input_dim, 38), nn.ReLU(),
            nn.Linear(38, 26),        nn.ReLU(),
            nn.Linear(26, latent_dim)
        )
        # Decoder: z -> 26 -> 38 -> 52
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 26), nn.ReLU(),
            nn.Linear(26, 38),         nn.ReLU(),
            nn.Linear(38, input_dim),  nn.Tanh()
        )

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z)

class MLP_VAE(nn.Module):
    def __init__(self, input_dim=52, latent_dim=8):
        super().__init__()
        # Encoder Base: 52 -> 38 -> 26
        self.enc_base = nn.Sequential(
            nn.Linear(input_dim, 38), nn.ReLU(),
            nn.Linear(38, 26),        nn.ReLU()
        )
        # Latent Heads (from 26 features)
        self.fc_mu = nn.Linear(26, latent_dim)
        self.fc_var = nn.Linear(26, latent_dim)
        
        # Decoder: z -> 26 -> 38 -> 52
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 26), nn.ReLU(),
            nn.Linear(26, 38),         nn.ReLU(),
            nn.Linear(38, input_dim),  nn.Tanh()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.enc_base(x)
        mu, logvar = self.fc_mu(h), self.fc_var(h)
        z = self.reparameterize(mu, logvar)
        return self.dec(z), mu, logvar

# ==============================================================================
# 2. CNN MODELS (Matches: 12x12 -> 6x6 -> 3x3)
# ==============================================================================
class CNN_AE(nn.Module):
    def __init__(self, in_channels=2, latent_dim=8):
        super().__init__()
        self.enc1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1)
        self.enc2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.flat_dim = 32 * 3 * 3      # 32 * 3 * 3 = 288 (32: output channels, 3, 3: width and height) 
        self.fc_enc = nn.Linear(self.flat_dim, latent_dim)  # Encoder to latent space
        self.fc_dec = nn.Linear(latent_dim, self.flat_dim)  # Latent space to decoder
        self.dec2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec1 = nn.ConvTranspose2d(16, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

    def encode(self, x):
        h = F.relu(self.enc1(x))
        h = F.relu(self.enc2(h))
        h = h.view(h.size(0), -1)
        return self.fc_enc(h)  # Direct latent representation

    def decode(self, z):
        h = F.relu(self.fc_dec(z))
        h = h.view(-1, 32, 3, 3)
        h = F.relu(self.dec2(h))
        return torch.tanh(self.dec1(h))  # output in [-1,1]

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon

class CNN_VAE(nn.Module):
    def __init__(self, in_channels=2, latent_dim=8):
        super().__init__()
        self.enc1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1)
        self.enc2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.flat_dim = 32 * 3 * 3
        self.fc_mu     = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.flat_dim)
        self.dec2   = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec1   = nn.ConvTranspose2d(16, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

    def encode(self, x):
        h = F.relu(self.enc1(x))
        h = F.relu(self.enc2(h))
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc_dec(z))
        h = h.view(-1, 32, 3, 3)
        h = F.relu(self.dec2(h))
        return torch.tanh(self.dec1(h))  # output in [-1,1]

    def forward(self, x):
        mu, logvar = self.encode(x)
        z          = self.reparameterize(mu, logvar)
        recon      = self.decode(z)
        return recon, mu, logvar

# ==============================================================================
# 3. PosEnc-AE (Matches: Embedding -> Concat PE -> Attention -> Flatten)
# ==============================================================================
def make_2d_sinusoidal_pos_enc(num_points=52, scale=1.0):
    try:
        from utils.DN_vf_tools import convertvf2image
    except ImportError:
        print("Warning: 'utils.DN_vf_tools' not found. PE will be zeros.")
        return torch.zeros(num_points, 4)

    img = convertvf2image(np.arange(1, num_points + 1, dtype=float))
    rows_np, cols_np = np.where(~np.isnan(img))
    indices = img[rows_np, cols_np].astype(int)

    coords = np.zeros((num_points, 2), dtype=np.float32)
    for r, c, idx in zip(rows_np, cols_np, indices):
        coords[idx - 1] = (r, c)

    rows = torch.from_numpy(coords[:, 0])
    cols = torch.from_numpy(coords[:, 1])
    R, C = rows.max(), cols.max()
    
    pe = torch.zeros(num_points, 4)
    pe[:, 0] = torch.sin(math.pi * rows / R)
    pe[:, 1] = torch.cos(math.pi * rows / R)
    pe[:, 2] = torch.sin(math.pi * cols / C)
    pe[:, 3] = torch.cos(math.pi * cols / C)
    return pe * scale

class SelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim, bias=False)
        self.key   = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.scale = dim ** 0.5
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        q = self.query(x); k = self.key(x).transpose(1, 2); v = self.value(x)
        attn = self.softmax(q @ k / self.scale)
        return attn @ v

class PosEncAE(nn.Module):
    def __init__(self, input_dim=1, emb_dim=32, latent_dim=8):
        super().__init__()
        self.feat_dim = emb_dim + 4 # 32 + 4 = 36
        
        self.value_emb = nn.Linear(input_dim, emb_dim)
        self.attn = SelfAttention(self.feat_dim)
        
        # Flatten: 52 * 36 -> latent
        self.fc_mu = nn.Linear(self.feat_dim * 52, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.feat_dim * 52)
        
        # Projection: 36 -> 1
        self.out = nn.Linear(self.feat_dim, 1)
        
        pe_tensor = make_2d_sinusoidal_pos_enc(num_points=52)
        self.register_buffer('pe', pe_tensor)

    def encode(self, x):
        h = self.value_emb(x.unsqueeze(-1)) # (B, 52, 32)
        pe_batch = self.pe.unsqueeze(0).expand(h.size(0), -1, -1)
        h = torch.cat([h, pe_batch], dim=-1) # (B, 52, 36)
        h = self.attn(h) + h
        h = h.flatten(1)
        return self.fc_mu(h)

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(-1, 52, self.feat_dim)
        h = self.out(h).squeeze(-1)
        return torch.tanh(h)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)