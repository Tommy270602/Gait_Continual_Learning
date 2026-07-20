#!/usr/bin/env python
# coding: utf-8

# # Phase 21c — FB-Mamba: Six-Model Security Evaluation
#
# **Thesis:** Code Division Modulation Layers Against Forgetting and Inference in
# Continual Gait Identification
# **Architecture:** FB-Mamba (Wu et al., 2026) — same backbone as Phase 21 / 21b
#
# ---
#
# ## Models evaluated (all six use the FB-Mamba backbone)
#
# | # | Strategy | Anti-forgetting | Privacy |
# |---|---|---|---|
# | 1 | **Mamba-Std-0%**         | None (naive sequential fine-tuning) | None |
# | 2 | **Mamba-Std-15%**        | 15% raw-data replay                 | None |
# | 3 | **Mamba-CDML**           | Sequence modulation per task        | Per-task CDML seed |
# | 4 | **Mamba-WGR-CDML**       | Seq. mod + wavelet generative replay | Seed + no raw data stored |
# | 5 | **Mamba-CDML+LoRA**      | Seq. mod + frozen backbone after T1 | Seed + frozen base weights |
# | 6 | **Mamba-WGR-CDML+LoRA**  | WGR + frozen backbone after T1      | Seed + WGR + frozen base |
#
# ## Datasets (all three run in a single execution)
#
# | Dataset | Subjects | Channels | Window | Tasks |
# |---|---|---|---|---|
# | **Dataset 1** (Gait ID) | 118 | 6 | 128 | 4 x ~30 subjects |
# | **UCI-HAR** (Subject ID) | 30 | 6 | 128 | 4 x ~8 subjects |
# | **WISDM** (Subject ID) | 51 | 6 | 200 | 4 x ~13 subjects |
#
# ## Attacks (applied to every model)
#
# | Attack | Type | Threat model |
# |---|---|---|
# | **MIA**      | Membership Inference (Yeom et al. 2018, CE-loss threshold) | Black-box |
# | **FSI**      | Feature Space Inference (cosine k-NN re-ID)                 | White-box; embedding access |
# | **Backdoor** | Trojan sinusoidal trigger                                   | White-box; training access |
#
# ## No-seed attacker convention (phase11 / phase11B fix)
#
# For CDML-based models (CDML, WGR-CDML, CDML+LoRA, WGR-CDML+LoRA), the
# "no-seed" attacker does **not** zero out the CDML sequence. Instead, following
# the corrected phase11 / phase11B procedure, it applies a fixed, deliberately
# *wrong* key generated from `NO_SEED_GUESS = 1` (i.e. `set_task_sequence(task, 1)`).
# This models a realistic attacker who guesses a plausible-looking key they do
# not actually possess, rather than an attacker who can zero the embedding
# outright (a degenerate, unrealistically strong "attack" that also breaks the
# cosine-invariance trap discussed in the thesis). For FSI specifically, the
# gallery (k-NN reference database) is always built at the TRUE oracle key —
# only the probe/query side varies between 'oracle' and 'no_seed'.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, ConcatDataset, Subset
from sklearn.metrics import roc_curve, auc as sk_auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from functools import partial
import os, warnings
warnings.filterwarnings('ignore')

try:
    import pywt
    PYWT_AVAILABLE = True
    print('PyWavelets available — DWT decomposition enabled')
except ImportError:
    PYWT_AVAILABLE = False
    print('PyWavelets not found — FFT fallback will be used for WGR')

plt.rcParams.update({
    'figure.dpi': 120, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.3, 'font.size': 11,
})

if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
print(f'Device: {DEVICE}  |  PyTorch: {torch.__version__}')


# In[ ]:


# ── Reproducibility ───────────────────────────────────────────────────────
RANDOM_SEED    = 27
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── Training ──────────────────────────────────────────────────────────────
EPOCHS         = 50      # increase to 200+ for paper-accurate results
LR_INIT        = 1e-3
LR_DECAY       = 0.98
BATCH_SIZE     = 64
VAL_SPLIT      = 0.15

# ── Std baseline replay ─────────────────────────────────────────────────────
STD_REPLAY_FRAC = 0.15   # used by 'Mamba-Std-15%'; 'Mamba-Std-0%' uses 0.0

# ── CDML ──────────────────────────────────────────────────────────────────
CDML_SEED_BASE = 1000    # seed for Task 1; Task k uses CDML_SEED_BASE + (k-1)
NO_SEED_GUESS  = 1       # attacker's guessed key — fixed, deliberately wrong
                          # (phase11 / phase11B convention: NOT a zero/null sequence)

# ── LoRA ──────────────────────────────────────────────────────────────────
LORA_RANK      = 8
LORA_ALPHA     = 8

# ── WGR ───────────────────────────────────────────────────────────────────
WGR_WAVELET    = 'db4'
WGR_LEVEL      = 3
WGR_N_SYNTH    = 30      # synthetic windows per class per replay step
WGR_JITTER_STD = 0.05

# ── FB-Mamba backbone ─────────────────────────────────────────────────────
EMBED_DIM      = 128     # embedding / classification head input dim
D_MODEL        = 16      # Mamba internal state dimension
NUM_LAYERS     = 2       # number of FBMambaBlocks
D_STATE        = 2       # SSM state dimension (paper default)
D_CONV         = 2       # depthwise conv kernel size (paper default)
EXPAND         = 2       # inner dimension expansion factor (paper default)

# ── Attacks ───────────────────────────────────────────────────────────────
KNN_K          = 5       # k for FSI k-NN classifier (cosine metric)
MIA_N_SAMPLES  = 200      # members / non-members sampled per task for MIA
BD_POISON_FRAC = 0.15    # fraction of training samples poisoned
BD_TRIGGER_AMP = 0.5     # sinusoidal trigger amplitude
BD_TRIGGER_HZ  = 8       # trigger frequency (cycles per window)
BD_TRIGGER_CH  = 2       # channel index that receives the trigger (acc_z)

print('Global configuration set.')


# In[ ]:


# Update 'train_dir' / 'test_dir' / 'data_dir' to your local paths.
DATASET_CONFIGS = {
    'Dataset_1': {
        'name':        'Gait ID — Dataset 1',
        'loader':      'dataset1',
        'train_dir':   'Data/Dataset_1/train',
        'test_dir':    'Data/Dataset_1/test',
        'n_channels':  6,
        'window_size': 128,
        'n_classes':   118,
        'task_splits': {
            'Task 1': (1,   30),
            'Task 2': (31,  60),
            'Task 3': (61,  90),
            'Task 4': (91, 118),
        },
        'bd_target': 90,   # backdoor target label index (0-based after remapping)
    },
    'UCI_HAR': {
        'name':        'UCI-HAR (Subject ID)',
        'loader':      'ucihar',
        'data_dir':    'Data/UCI HAR Dataset',
        'n_channels':  6,
        'window_size': 128,
        'n_classes':   30,
        'task_splits': {
            'Task 1': (1,   8),
            'Task 2': (9,  17),
            'Task 3': (18, 24),
            'Task 4': (25, 30),
        },
        'bd_target': 27,
    },
    'WISDM': {
        'name':        'WISDM (Subject ID)',
        'loader':      'wisdm',
        'data_dir':    'Data/wisdm-dataset',
        'n_channels':  6,
        'window_size': 200,
        'n_classes':   51,
        'task_splits': {
            'Task 1': (1,  13),
            'Task 2': (40, 51),
            'Task 3': (27, 39),
            'Task 4': (14, 26),
        },
        'bd_target': 44,
    },
}
print(f'Registered datasets: {list(DATASET_CONFIGS.keys())}')


# In[ ]:


# ── Dataset 1 (Gait ID) ───────────────────────────────────────────────────
def load_dataset1(cfg):
    def _split(d, prefix):
        axes = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
        X = np.stack([np.loadtxt(f'{d}/Inertial_Signals/{prefix}_{a}.txt')
                      for a in axes], axis=1).astype(np.float32)
        y = np.loadtxt(f'{d}/y_{prefix}.txt', dtype=int)
        return X, y
    X_tr, y_tr = _split(cfg['train_dir'], 'train')
    X_te, y_te = _split(cfg['test_dir'],  'test')
    mu  = X_tr.mean(axis=(0, 2), keepdims=True)
    std = X_tr.std(axis=(0, 2),  keepdims=True) + 1e-8
    return (X_tr - mu) / std, y_tr, (X_te - mu) / std, y_te


# ── UCI-HAR (subject identification task) ────────────────────────────────
def load_ucihar(cfg):
    def _split(d, split):
        axes = ['total_acc_x', 'total_acc_y', 'total_acc_z',
                'body_gyro_x', 'body_gyro_y', 'body_gyro_z']
        X   = np.stack([
            np.loadtxt(f'{d}/{split}/Inertial Signals/{a}_{split}.txt')
            for a in axes], axis=1).astype(np.float32)
        y   = np.loadtxt(f'{d}/{split}/subject_{split}.txt', dtype=int)
        act = np.loadtxt(f'{d}/{split}/y_{split}.txt',       dtype=int)
        return X, y, act
    X_tr, y_tr, act_tr = _split(cfg['data_dir'], 'train')
    X_te, y_te, act_te = _split(cfg['data_dir'], 'test')
    # ↓ FIX: original split separates subjects entirely (activity-recognition design).
    # For subject-ID, pool everything and re-split stratified by subject, like WISDM.
    X_all   = np.concatenate([X_tr, X_te], axis=0)
    y_all   = np.concatenate([y_tr, y_te], axis=0)
    act_all = np.concatenate([act_tr, act_te], axis=0)
    # Walking-only filter (1=WALKING, 2=WALKING_UPSTAIRS, 3=WALKING_DOWNSTAIRS)
    walk_mask = np.isin(act_all, [1, 2, 3])
    X_all     = X_all[walk_mask]
    y_all     = y_all[walk_mask]
    print(f'  UCI HAR walking-only: {walk_mask.sum()} / {len(walk_mask)} windows '
          f'({100*walk_mask.mean():.1f}%)')
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all, test_size=0.2, random_state=RANDOM_SEED, stratify=y_all)
    mu  = X_tr.mean(axis=(0, 2), keepdims=True)
    std = X_tr.std(axis=(0, 2),  keepdims=True) + 1e-8
    return (X_tr - mu) / std, y_tr, (X_te - mu) / std, y_te


# ── WISDM (subject identification task) ──────────────────────────────────
def load_wisdm(cfg):
    W   = cfg['window_size']
    step = W // 2

    def _read_dir(sensor_dir, key):
        out = {}
        for fn in sorted(os.listdir(sensor_dir)):
            if not fn.endswith('.txt'):
                continue
            sid = int(fn.split('_')[1]) - 1599  # remap 1600→1, 1601→2, …
            rows = []
            with open(os.path.join(sensor_dir, fn), encoding='utf-8', errors='ignore') as fh:
                for line in fh:
                    line = line.strip().rstrip(';')
                    if not line or not line[0].isdigit():
                        continue
                    parts = line.split(',')
                    if len(parts) >= 6:
                        try:
                            rows.append([float(p.strip()) for p in parts[3:6]])
                        except ValueError:
                            pass
            if rows:
                out.setdefault(sid, {})[key] = np.array(rows, dtype=np.float32)
        return out

    accel = _read_dir(os.path.join(cfg['data_dir'], 'raw', 'phone', 'accel'), 'acc')
    gyro  = _read_dir(os.path.join(cfg['data_dir'], 'raw', 'phone', 'gyro'),  'gyro')
    for sid, sens in gyro.items():
        accel.setdefault(sid, {}).update(sens)

    X_all, y_all = [], []
    for sid in sorted(accel):
        s = accel[sid]
        if 'acc' not in s or 'gyro' not in s:
            continue
        n = min(len(s['acc']), len(s['gyro']))
        data = np.concatenate([s['acc'][:n], s['gyro'][:n]], axis=1)  # (T,6)
        for start in range(0, n - W + 1, step):
            X_all.append(data[start:start + W].T)  # (6,W)
            y_all.append(sid)

    X = np.stack(X_all).astype(np.float32)
    y = np.array(y_all, dtype=int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    mu  = X_tr.mean(axis=(0, 2), keepdims=True)
    std = X_tr.std(axis=(0, 2),  keepdims=True) + 1e-8
    return (X_tr - mu) / std, y_tr, (X_te - mu) / std, y_te


# ── Dispatcher ────────────────────────────────────────────────────────────
def load_raw_data(cfg):
    return {'dataset1': load_dataset1,
            'ucihar':   load_ucihar,
            'wisdm':    load_wisdm}[cfg['loader']](cfg)

print('Data loaders defined.')


# In[ ]:


def make_task_datasets(X_tr, y_tr, X_te, y_te, cfg):
    unique   = np.sort(np.unique(np.concatenate([y_tr, y_te])))
    l2i      = {lbl: idx for idx, lbl in enumerate(unique)}
    y_tr_i   = np.array([l2i[l] for l in y_tr])
    y_te_i   = np.array([l2i[l] for l in y_te])
    rng      = torch.Generator().manual_seed(RANDOM_SEED)
    task_data = {}
    for tname, (lo, hi) in cfg['task_splits'].items():
        mask_tr = (y_tr >= lo) & (y_tr <= hi)
        Xt = torch.tensor(X_tr[mask_tr])
        yt = torch.tensor(y_tr_i[mask_tr], dtype=torch.long)
        full_ds = TensorDataset(Xt, yt)
        n_val   = max(1, int(len(full_ds) * VAL_SPLIT))
        tr_ds, val_ds = random_split(full_ds, [len(full_ds) - n_val, n_val], generator=rng)
        mask_te = (y_te >= lo) & (y_te <= hi)
        te_ds   = TensorDataset(torch.tensor(X_te[mask_te]),
                                torch.tensor(y_te_i[mask_te], dtype=torch.long))
        task_data[tname] = {'train': tr_ds, 'val': val_ds, 'test': te_ds}
        print(f'  {tname}: {len(tr_ds)} train | {n_val} val | {len(te_ds)} test')
    return task_data, l2i

print('make_task_datasets defined.')


# In[ ]:


# ─────────────────────────────────────────────────────────────────────────
# FB-Mamba Core — pure PyTorch, no Triton/CUDA required
# Source: Wu et al., 2026 — fb_mamba_train_eval.py
# ─────────────────────────────────────────────────────────────────────────

def selective_scan(u, delta, A, B, C, D):
    # Sequential selective SSM scan (ZOH discretisation, Eq. 3-4)
    # u: (B,L,d_inner)  delta: (B,L,d_inner)  A: (d_inner,d_state)
    # B,C: (B,L,d_state)  D: (d_inner,)
    B_batch, L, d_inner = u.shape
    dt   = delta.unsqueeze(-1)                                       # (B,L,d,1)
    Aexp = torch.exp(dt * A.unsqueeze(0).unsqueeze(0))               # (B,L,d,N)
    Bu   = dt * B.unsqueeze(2) * u.unsqueeze(-1)                     # (B,L,d,N)
    h    = torch.zeros(B_batch, d_inner, A.shape[1],
                       device=u.device, dtype=u.dtype)
    ys   = []
    for t in range(L):
        h  = Aexp[:, t] * h + Bu[:, t]                              # (B,d,N)
        yt = (h * C[:, t].unsqueeze(1)).sum(-1)                      # (B,d)
        ys.append(yt)
    y = torch.stack(ys, dim=1)                                       # (B,L,d)
    return y + u * D.unsqueeze(0).unsqueeze(0)


class MambaBlock(nn.Module):
    # Single Mamba layer — Gu & Dao (2023), defaults: d_state=16, d_conv=4, expand=2
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_inner  = int(expand * d_model)
        self.d_state  = d_state
        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d   = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                  padding=d_conv - 1, groups=self.d_inner, bias=True)
        self.x_proj   = nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)
        self.dt_proj  = nn.Linear(self.d_inner, self.d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32
                         ).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log    = nn.Parameter(torch.log(A))
        self.D        = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm     = nn.LayerNorm(d_model)

    def forward(self, x):
        residual   = x
        B_b, L, _ = x.shape
        xz         = self.in_proj(x)
        x_b, z     = xz.chunk(2, dim=-1)
        xc = self.conv1d(x_b.transpose(1, 2))[..., :L].transpose(1, 2)
        xc = F.silu(xc)
        proj  = self.x_proj(xc)
        B_ssm = proj[..., :self.d_state]
        C_ssm = proj[..., self.d_state: 2 * self.d_state]
        delta = F.softplus(self.dt_proj(proj[..., 2 * self.d_state:]))
        A     = -torch.exp(self.A_log)
        y     = selective_scan(xc, delta, A, B_ssm, C_ssm, self.D)
        y     = y * F.silu(z)
        return self.norm(self.out_proj(y) + residual)


class FBMambaBlock(nn.Module):
    # Forward-Backward Mamba block (Wu et al. 2026, Eq. 9-13)
    # H_fused = alpha * Mamba_fwd(H) + (1-alpha) * flip(Mamba_bwd(flip(H)))
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.mamba_fwd     = MambaBlock(d_model, d_state, d_conv, expand)
        self.mamba_bwd     = MambaBlock(d_model, d_state, d_conv, expand)
        self.fusion_linear = nn.Linear(d_model * 2, d_model)  # W_alpha, b_alpha (Eq.11)
        self.out_proj      = nn.Linear(d_model, d_model)       # W_out, b_out (Eq.13)
        self.norm          = nn.LayerNorm(d_model)

    def forward(self, H):
        H_fwd   = self.mamba_fwd(H)
        H_bwd   = torch.flip(self.mamba_bwd(torch.flip(H, dims=[1])), dims=[1])
        alpha   = torch.sigmoid(self.fusion_linear(torch.cat([H_fwd, H_bwd], dim=-1)))
        H_fused = alpha * H_fwd + (1.0 - alpha) * H_bwd
        return self.norm(self.out_proj(H_fused))

print('selective_scan / MambaBlock / FBMambaBlock defined.')


# In[ ]:


class GaitMamba(nn.Module):
    # Mamba backbone for gait/HAR identification — used directly by the two
    # 'Std' (no CDML, no LoRA) baselines.
    # Input: (B, C, W) — batch x channels x window_size
    # Flow:  transpose -> input_proj -> FBMambaBlock*N -> avg_pool -> embedding -> classifier
    def __init__(self, n_channels=6, n_classes=118, embed_dim=EMBED_DIM,
                 d_model=D_MODEL, num_layers=NUM_LAYERS,
                 d_state=D_STATE, d_conv=D_CONV, expand=EXPAND):
        super().__init__()
        self.embed_dim  = embed_dim
        self.input_proj = nn.Linear(n_channels, d_model)
        self.blocks     = nn.ModuleList([
            FBMambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(num_layers)])
        self.embedding  = nn.Linear(d_model, embed_dim)
        self.classifier = nn.Linear(embed_dim, n_classes)

    def embed(self, x):
        h = x.transpose(1, 2)    # (B,W,C)
        h = self.input_proj(h)   # (B,W,d_model)
        for block in self.blocks:
            h = block(h)
        h = h.mean(dim=1)        # (B,d_model) — global average pool (Eq.14)
        return self.embedding(h) # (B,embed_dim)

    def forward(self, x):
        return self.classifier(self.embed(x))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GaitMamba_LoRA(nn.Module):
    # GaitMamba where embedding + classifier are LoRALinear.
    # After Task 1: Mamba blocks + input_proj frozen; only LoRA adapters updated.
    def __init__(self, n_channels=6, n_classes=118, embed_dim=EMBED_DIM,
                 lora_rank=LORA_RANK, lora_alpha=LORA_ALPHA,
                 d_model=D_MODEL, num_layers=NUM_LAYERS,
                 d_state=D_STATE, d_conv=D_CONV, expand=EXPAND):
        super().__init__()
        self.embed_dim  = embed_dim
        self.input_proj = nn.Linear(n_channels, d_model)
        self.blocks     = nn.ModuleList([
            FBMambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(num_layers)])
        self.embedding  = LoRALinear(d_model, embed_dim,  rank=lora_rank, alpha=lora_alpha)
        self.classifier = LoRALinear(embed_dim, n_classes, rank=lora_rank, alpha=lora_alpha)

    def embed(self, x):
        h = x.transpose(1, 2)
        h = self.input_proj(h)
        for block in self.blocks:
            h = block(h)
        h = h.mean(dim=1)
        return self.embedding(h)

    def forward(self, x):
        return self.classifier(self.embed(x))

    def save_lora(self, task_name):
        self.embedding.save_lora(task_name)
        self.classifier.save_lora(task_name)

    def load_lora(self, task_name):
        self.embedding.load_lora(task_name)
        self.classifier.load_lora(task_name)

    def reset_lora(self):
        self.embedding.reset_lora()
        self.classifier.reset_lora()

    def freeze_base_weights(self):
        for p in self.input_proj.parameters():
            p.requires_grad_(False)
        for blk in self.blocks:
            for p in blk.parameters():
                p.requires_grad_(False)
        self.embedding.freeze_base()
        self.classifier.freeze_base()

    def lora_parameters(self):
        return self.embedding.lora_parameters() + self.classifier.lora_parameters()

print('GaitMamba and GaitMamba_LoRA defined.')


# In[ ]:


def generate_cdml_sequence(embed_dim, seed):
    rng = np.random.default_rng(seed)
    return torch.tensor(
        np.where(rng.random(embed_dim) >= 0.5, 1.0, -1.0).astype(np.float32))


class CDMLLayer(nn.Module):
    # Code Division Modulation Layer: m = s_k ⊙ h  (Milani 2024, Eq. 1)
    def __init__(self, embed_dim, seed):
        super().__init__()
        self.register_buffer('sequence', generate_cdml_sequence(embed_dim, seed))

    def forward(self, h):
        return h * self.sequence


class LoRALinear(nn.Module):
    # Linear + Low-Rank Adaptation (Hu et al. 2021).
    # y = W·x + b  +  (B_k · A_k) · x * (alpha/rank)
    def __init__(self, in_features, out_features,
                 rank=LORA_RANK, alpha=LORA_ALPHA, bias=True):
        super().__init__()
        self.rank    = rank
        self.alpha   = alpha
        self.scaling = alpha / rank
        self.weight  = nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty(out_features, in_features),
                                     a=np.sqrt(5)))
        if bias:
            b = 1 / np.sqrt(in_features) if in_features > 0 else 0
            self.bias = nn.Parameter(torch.empty(out_features).uniform_(-b, b))
        else:
            self.register_parameter('bias', None)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        self._states: dict = {}

    def forward(self, x):
        base = F.linear(x, self.weight, self.bias)
        lora = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base + lora

    def save_lora(self, k):
        self._states[k] = (self.lora_A.data.cpu().clone(),
                           self.lora_B.data.cpu().clone())

    def load_lora(self, k):
        A, B = self._states[k]
        self.lora_A.data.copy_(A.to(self.lora_A.device))
        self.lora_B.data.copy_(B.to(self.lora_B.device))

    def reset_lora(self):
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def freeze_base(self):
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

    def lora_parameters(self):
        return [self.lora_A, self.lora_B]

print('CDMLLayer and LoRALinear defined.')


# In[ ]:


class GaitMamba_CDML(nn.Module):
    # GaitMamba + CDML — full backbone fine-tuning per task, no frozen weights.
    def __init__(self, n_channels=6, n_classes=118, embed_dim=EMBED_DIM,
                 seed=CDML_SEED_BASE, d_model=D_MODEL, num_layers=NUM_LAYERS,
                 d_state=D_STATE, d_conv=D_CONV, expand=EXPAND):
        super().__init__()
        self.embed_dim = embed_dim
        self.backbone  = GaitMamba(n_channels, n_classes, embed_dim,
                                   d_model, num_layers, d_state, d_conv, expand)
        self.cdml      = CDMLLayer(embed_dim, seed)
        self.seeds     = {}

    def embed_raw(self, x):
        return self.backbone.embed(x)

    def embed_modulated(self, x):
        return self.cdml(self.backbone.embed(x))

    def forward(self, x):
        return self.backbone.classifier(self.cdml(self.backbone.embed(x)))

    def set_task_sequence(self, task_name, seed):
        self.seeds[task_name] = seed
        self.cdml.sequence = generate_cdml_sequence(
            self.embed_dim, seed).to(next(self.parameters()).device)

    def zero_sequence(self):
        # Degenerate null key — kept for architectural completeness but NOT used
        # by the MIA / FSI attacks below (see NO_SEED_GUESS convention).
        self.cdml.sequence = torch.zeros(
            self.embed_dim, device=next(self.parameters()).device)


class GaitMamba_CDML_LoRA(nn.Module):
    # GaitMamba + CDML + LoRA — frozen backbone after Task 1, only adapters updated.
    # Architecture:
    #   (B,C,W) -> input_proj -> FBMambaBlock*N  [frozen after T1]
    #           -> avg_pool
    #           -> LoRALinear embedding  [base frozen after T1, adapter A_k/B_k per task]
    #           -> CDMLLayer             [per-task sequence s_k]
    #           -> LoRALinear classifier [base frozen after T1, adapter A_k/B_k per task]
    #           -> logits
    def __init__(self, n_channels=6, n_classes=118, embed_dim=EMBED_DIM,
                 seed=CDML_SEED_BASE, lora_rank=LORA_RANK, lora_alpha=LORA_ALPHA,
                 d_model=D_MODEL, num_layers=NUM_LAYERS,
                 d_state=D_STATE, d_conv=D_CONV, expand=EXPAND):
        super().__init__()
        self.embed_dim = embed_dim
        self.backbone  = GaitMamba_LoRA(n_channels, n_classes, embed_dim,
                                        lora_rank, lora_alpha,
                                        d_model, num_layers, d_state, d_conv, expand)
        self.cdml      = CDMLLayer(embed_dim, seed)
        self.seeds     = {}

    def embed_raw(self, x):
        return self.backbone.embed(x)

    def embed_modulated(self, x):
        return self.cdml(self.backbone.embed(x))

    def forward(self, x):
        return self.backbone.classifier(self.cdml(self.backbone.embed(x)))

    def set_task_sequence(self, task_name, seed):
        self.seeds[task_name] = seed
        self.cdml.sequence = generate_cdml_sequence(
            self.embed_dim, seed).to(next(self.parameters()).device)

    def zero_sequence(self):
        self.cdml.sequence = torch.zeros(
            self.embed_dim, device=next(self.parameters()).device)

    def save_lora(self, task_name):   self.backbone.save_lora(task_name)
    def load_lora(self, task_name):   self.backbone.load_lora(task_name)
    def reset_lora(self):             self.backbone.reset_lora()

    def freeze_base_weights(self):
        self.backbone.freeze_base_weights()
        n = sum(p.numel() for p in self.lora_parameters())
        print(f'  Backbone frozen. LoRA adapter params per task: {n:,}')

    def lora_parameters(self):
        return self.backbone.lora_parameters()

    def trainable_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

print('GaitMamba_CDML and GaitMamba_CDML_LoRA defined.')


# In[ ]:


def _fft_approximate(X_np, low_frac=0.125):
    F_ = np.fft.rfft(X_np, axis=-1)
    cut = max(1, int(F_.shape[-1] * low_frac))
    Fl  = F_.copy(); Fl[..., cut:] = 0
    Fh  = F_.copy(); Fh[..., :cut] = 0
    n   = X_np.shape[-1]
    return (np.fft.irfft(Fl, n=n, axis=-1),
            np.fft.irfft(Fh, n=n, axis=-1))


class WaveletReplayBuffer:
    # Per-class DWT statistics -> stochastic synthetic windows on demand.
    # No raw data stored: only wavelet coefficient mean/std per class.
    def __init__(self, wavelet=WGR_WAVELET, level=WGR_LEVEL,
                 n_synth=WGR_N_SYNTH, jitter_std=WGR_JITTER_STD, seed=RANDOM_SEED):
        self.wavelet = wavelet; self.level = level
        self.n_synth = n_synth; self.jitter = jitter_std
        self.rng     = np.random.default_rng(seed)
        self._tasks  = []

    def _to_numpy(self, ds):
        X = torch.stack([ds[i][0] for i in range(len(ds))]).numpy()
        y = torch.stack([ds[i][1] for i in range(len(ds))]).numpy()
        return X, y

    def add_task(self, train_ds, task_name):
        X_np, y_np = self._to_numpy(train_ds)
        N, C, T    = X_np.shape
        classes    = np.unique(y_np)
        if PYWT_AVAILABLE:
            coeffs_all = [[pywt.wavedec(X_np[n, c, :], self.wavelet, level=self.level)
                           for c in range(C)] for n in range(N)]
            class_stats = {}
            for cls in classes:
                mask   = (y_np == cls)
                cA_arr = np.array([[coeffs_all[i][c][0] for c in range(C)]
                                   for i in range(N) if mask[i]])
                flat   = cA_arr.reshape(cA_arr.shape[0], -1)
                class_stats[int(cls)] = {'mu': flat.mean(0), 'sigma': flat.std(0) + 1e-6}
            detail_stats = []
            for lv in range(1, self.level + 1):
                cD = np.array([[coeffs_all[i][c][lv] for c in range(C)]
                               for i in range(N)])
                detail_stats.append({'mu': cD.mean(0), 'sigma': cD.std(0) + 1e-6})
            coeff_lens = [len(a) for a in coeffs_all[0][0]]
        else:
            Xl, Xh = _fft_approximate(X_np)
            class_stats = {
                int(cls): {'mu':    Xl[y_np == cls].reshape(sum(y_np == cls), -1).mean(0),
                           'sigma': Xl[y_np == cls].reshape(sum(y_np == cls), -1).std(0) + 1e-6}
                for cls in classes}
            detail_stats = [{'mu': Xh.mean(0), 'sigma': Xh.std(0) + 1e-6}]
            coeff_lens   = None
        self._tasks.append({'task_name': task_name, 'class_stats': class_stats,
                            'detail_stats': detail_stats, 'coeff_lens': coeff_lens,
                            'C': C, 'T': T, 'classes': list(classes.astype(int))})
        print(f'  WGR buffer: stored statistics for {task_name} ({len(classes)} classes)')

    def synthesize_task(self, task_idx):
        t = self._tasks[task_idx]
        C, T = t['C'], t['T']
        Xs, ys = [], []
        if PYWT_AVAILABLE:
            for cls, stats in t['class_stats'].items():
                for _ in range(self.n_synth):
                    cA_flat = self.rng.normal(stats['mu'], stats['sigma'])
                    cA      = cA_flat.reshape(C, -1)
                    window  = np.zeros((C, T), dtype=np.float32)
                    for c in range(C):
                        coeffs_c = [cA[c].astype(float)]
                        for ds in t['detail_stats']:
                            coeffs_c.append(self.rng.normal(ds['mu'][c], ds['sigma'][c]).astype(float))
                        rec = pywt.waverec(coeffs_c, self.wavelet)
                        window[c] = rec[:T] if len(rec) >= T else np.pad(rec.astype(np.float32), (0, T - len(rec)))
                    window += self.rng.normal(0, self.jitter, window.shape).astype(np.float32)
                    Xs.append(window); ys.append(cls)
        else:
            det = t['detail_stats'][0]
            for cls, stats in t['class_stats'].items():
                for _ in range(self.n_synth):
                    xl = self.rng.normal(stats['mu'], stats['sigma']).reshape(C, -1)
                    xl = np.repeat(xl, 4, axis=-1)[:, :T]
                    xh = self.rng.normal(det['mu'], det['sigma']).reshape(C, -1)[:, :T]
                    if xh.shape[-1] < T:
                        xh = np.pad(xh, ((0, 0), (0, T - xh.shape[-1])))
                    window = (xl + xh).astype(np.float32)
                    window += self.rng.normal(0, self.jitter, window.shape).astype(np.float32)
                    Xs.append(window); ys.append(cls)
        return TensorDataset(torch.tensor(np.stack(Xs)),
                             torch.tensor(ys, dtype=torch.long))

    def get_replay_dataset(self):
        if not self._tasks:
            return None
        return ConcatDataset([self.synthesize_task(i) for i in range(len(self._tasks))])

print('WaveletReplayBuffer defined.')


# In[ ]:


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        correct += (model(X_b).argmax(1) == y_b).sum().item()
        total   += len(y_b)
    return correct / total if total > 0 else 0.0


def _run_epoch(model, loader, optimizer, criterion, device):
    model.train()
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        criterion(model(X_b), y_b).backward()
        optimizer.step()


def _eval_all_tasks(model, task_data, task_names, step_idx, acc_matrix,
                    device, is_lora):
    curr = task_names[step_idx]
    curr_seed = model.seeds.get(curr, CDML_SEED_BASE + step_idx)
    for ei, et in enumerate(task_names[:step_idx + 1]):
        model.set_task_sequence(et, model.seeds.get(et, CDML_SEED_BASE + ei))
        if is_lora:
            model.load_lora(et)
        tl = DataLoader(task_data[et]['test'], batch_size=256)
        acc_matrix[step_idx, ei] = evaluate(model, tl, device)
    # restore current task state
    model.set_task_sequence(curr, curr_seed)
    if is_lora:
        model.load_lora(curr)


def _make_model(ModelClass, cfg, **kw):
    return ModelClass(n_channels=cfg['n_channels'],
                      n_classes=cfg['n_classes'],
                      embed_dim=cfg.get('embed_dim', EMBED_DIM), **kw)

print('Training utilities defined.')


# In[ ]:


def train_mamba_std(task_data, task_names, cfg, device, replay_frac=0.0,
                     epochs=None, verbose_every=25, label='Mamba-Std'):
    # Standard sequential fine-tuning baseline (no CDML, no LoRA).
    # replay_frac=0.0  -> naive fine-tuning, no rehearsal ('Mamba-Std-0%')
    # replay_frac=0.15 -> raw-data rehearsal buffer ('Mamba-Std-15%')
    if epochs is None: epochs = EPOCHS
    n_tasks    = len(task_names)
    model      = _make_model(GaitMamba, cfg).to(device)
    criterion  = nn.CrossEntropyLoss()
    acc_matrix = np.full((n_tasks, n_tasks), np.nan)
    torch.manual_seed(RANDOM_SEED)

    past_train_datasets = []

    for step_idx, task_name in enumerate(task_names):
        print(f'[{label}] Step {step_idx + 1}/{n_tasks}: {task_name}')

        datasets_to_mix = [task_data[task_name]['train']]
        if replay_frac > 0:
            for past_ds in past_train_datasets:
                n_replay = max(1, int(len(past_ds) * replay_frac))
                indices  = torch.randperm(len(past_ds))[:n_replay].tolist()
                datasets_to_mix.append(Subset(past_ds, indices))
            if len(datasets_to_mix) > 1:
                print(f'  + replay from {len(datasets_to_mix) - 1} past task(s) '
                      f'(frac={replay_frac})')

        loader    = DataLoader(ConcatDataset(datasets_to_mix),
                               batch_size=BATCH_SIZE, shuffle=True)
        optimizer = optim.Adam(model.parameters(), lr=LR_INIT)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_DECAY)
        for epoch in range(1, epochs + 1):
            _run_epoch(model, loader, optimizer, criterion, device)
            scheduler.step()
            if epoch % verbose_every == 0 or epoch == 1:
                vl  = DataLoader(task_data[task_name]['val'], batch_size=256)
                print(f'  Epoch {epoch:>3}/{epochs}  val={evaluate(model, vl, device):.3f}')

        past_train_datasets.append(task_data[task_name]['train'])

        for ei, et in enumerate(task_names[:step_idx + 1]):
            tl = DataLoader(task_data[et]['test'], batch_size=256)
            acc_matrix[step_idx, ei] = evaluate(model, tl, device)
        row = '  '.join([f'T{j+1}:{acc_matrix[step_idx,j]*100:.1f}%'
                         for j in range(step_idx + 1)])
        print(f'  -> {row}')
    return model, acc_matrix


def train_mamba_cdml(task_data, task_names, cfg, device,
                     epochs=None, verbose_every=25, label='Mamba-CDML'):
    # Full backbone fine-tuned per task, CDML modulation, no replay.
    if epochs is None: epochs = EPOCHS
    n_tasks    = len(task_names)
    model      = _make_model(GaitMamba_CDML, cfg).to(device)
    criterion  = nn.CrossEntropyLoss()
    acc_matrix = np.full((n_tasks, n_tasks), np.nan)
    torch.manual_seed(RANDOM_SEED)

    for step_idx, task_name in enumerate(task_names):
        seed_k = CDML_SEED_BASE + step_idx
        model.set_task_sequence(task_name, seed_k)
        print(f'[{label}] Step {step_idx + 1}/{n_tasks}: {task_name}')
        loader    = DataLoader(task_data[task_name]['train'],
                               batch_size=BATCH_SIZE, shuffle=True)
        optimizer = optim.Adam(model.parameters(), lr=LR_INIT)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_DECAY)
        for epoch in range(1, epochs + 1):
            model.set_task_sequence(task_name, seed_k)
            _run_epoch(model, loader, optimizer, criterion, device)
            scheduler.step()
            if epoch % verbose_every == 0 or epoch == 1:
                model.set_task_sequence(task_name, seed_k)
                vl  = DataLoader(task_data[task_name]['val'], batch_size=256)
                print(f'  Epoch {epoch:>3}/{epochs}  val={evaluate(model, vl, device):.3f}')
        _eval_all_tasks(model, task_data, task_names, step_idx, acc_matrix, device, is_lora=False)
        row = '  '.join([f'T{j+1}:{acc_matrix[step_idx,j]*100:.1f}%'
                         for j in range(step_idx + 1)])
        print(f'  -> {row}')
    return model, acc_matrix


def train_mamba_cdml_lora(task_data, task_names, cfg, device,
                          epochs=None, verbose_every=25, label='Mamba-CDML+LoRA'):
    # Backbone frozen after T1, only LoRA adapters updated.
    if epochs is None: epochs = EPOCHS
    n_tasks    = len(task_names)
    model      = _make_model(GaitMamba_CDML_LoRA, cfg).to(device)
    criterion  = nn.CrossEntropyLoss()
    acc_matrix = np.full((n_tasks, n_tasks), np.nan)
    torch.manual_seed(RANDOM_SEED)

    for step_idx, task_name in enumerate(task_names):
        seed_k = CDML_SEED_BASE + step_idx
        model.set_task_sequence(task_name, seed_k)
        model.reset_lora()
        print(f'[{label}] Step {step_idx + 1}/{n_tasks}: {task_name}')
        if step_idx == 0:
            optimizer = optim.Adam(model.parameters(), lr=LR_INIT)
            print(f'  Full model: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} params')
        else:
            model.freeze_base_weights()
            optimizer = optim.Adam(model.lora_parameters(), lr=LR_INIT)
        loader    = DataLoader(task_data[task_name]['train'],
                               batch_size=BATCH_SIZE, shuffle=True)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_DECAY)
        for epoch in range(1, epochs + 1):
            model.set_task_sequence(task_name, seed_k)
            _run_epoch(model, loader, optimizer, criterion, device)
            scheduler.step()
            if epoch % verbose_every == 0 or epoch == 1:
                model.set_task_sequence(task_name, seed_k)
                vl  = DataLoader(task_data[task_name]['val'], batch_size=256)
                print(f'  Epoch {epoch:>3}/{epochs}  val={evaluate(model, vl, device):.3f}')
        model.save_lora(task_name)
        _eval_all_tasks(model, task_data, task_names, step_idx, acc_matrix, device, is_lora=True)
        row = '  '.join([f'T{j+1}:{acc_matrix[step_idx,j]*100:.1f}%'
                         for j in range(step_idx + 1)])
        print(f'  -> {row}')
    return model, acc_matrix


def train_mamba_wgr_cdml(task_data, task_names, cfg, device,
                          epochs=None, verbose_every=25, label='Mamba-WGR-CDML'):
    # CDML + wavelet synthetic replay, no raw data stored, no LoRA.
    if epochs is None: epochs = EPOCHS
    n_tasks    = len(task_names)
    model      = _make_model(GaitMamba_CDML, cfg).to(device)
    criterion  = nn.CrossEntropyLoss()
    acc_matrix = np.full((n_tasks, n_tasks), np.nan)
    wgr_buf    = WaveletReplayBuffer(seed=RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    for step_idx, task_name in enumerate(task_names):
        seed_k  = CDML_SEED_BASE + step_idx
        model.set_task_sequence(task_name, seed_k)
        print(f'[{label}] Step {step_idx + 1}/{n_tasks}: {task_name}')
        datasets = [task_data[task_name]['train']]
        if step_idx > 0:
            synth = wgr_buf.get_replay_dataset()
            if synth is not None:
                datasets.append(synth)
                print(f'  + {len(synth)} synthetic replay samples')
        loader    = DataLoader(ConcatDataset(datasets), batch_size=BATCH_SIZE, shuffle=True)
        optimizer = optim.Adam(model.parameters(), lr=LR_INIT)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_DECAY)
        for epoch in range(1, epochs + 1):
            model.set_task_sequence(task_name, seed_k)
            _run_epoch(model, loader, optimizer, criterion, device)
            scheduler.step()
            if epoch % verbose_every == 0 or epoch == 1:
                model.set_task_sequence(task_name, seed_k)
                vl  = DataLoader(task_data[task_name]['val'], batch_size=256)
                print(f'  Epoch {epoch:>3}/{epochs}  val={evaluate(model, vl, device):.3f}')
        _eval_all_tasks(model, task_data, task_names, step_idx, acc_matrix, device, is_lora=False)
        row = '  '.join([f'T{j+1}:{acc_matrix[step_idx,j]*100:.1f}%'
                         for j in range(step_idx + 1)])
        print(f'  -> {row}')
        wgr_buf.add_task(task_data[task_name]['train'], task_name)
    return model, acc_matrix


def train_mamba_wgr_cdml_lora(task_data, task_names, cfg, device,
                               epochs=None, verbose_every=25, label='Mamba-WGR-CDML+LoRA'):
    # WGR + frozen backbone after T1 + LoRA adapters.
    if epochs is None: epochs = EPOCHS
    n_tasks    = len(task_names)
    model      = _make_model(GaitMamba_CDML_LoRA, cfg).to(device)
    criterion  = nn.CrossEntropyLoss()
    acc_matrix = np.full((n_tasks, n_tasks), np.nan)
    wgr_buf    = WaveletReplayBuffer(seed=RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    for step_idx, task_name in enumerate(task_names):
        seed_k = CDML_SEED_BASE + step_idx
        model.set_task_sequence(task_name, seed_k)
        model.reset_lora()
        print(f'[{label}] Step {step_idx + 1}/{n_tasks}: {task_name}')
        datasets = [task_data[task_name]['train']]
        if step_idx > 0:
            synth = wgr_buf.get_replay_dataset()
            if synth is not None:
                datasets.append(synth)
                print(f'  + {len(synth)} synthetic replay samples')
        if step_idx == 0:
            optimizer = optim.Adam(model.parameters(), lr=LR_INIT)
            print(f'  Full model: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} params')
        else:
            model.freeze_base_weights()
            optimizer = optim.Adam(model.lora_parameters(), lr=LR_INIT)
        loader    = DataLoader(ConcatDataset(datasets), batch_size=BATCH_SIZE, shuffle=True)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_DECAY)
        for epoch in range(1, epochs + 1):
            model.set_task_sequence(task_name, seed_k)
            _run_epoch(model, loader, optimizer, criterion, device)
            scheduler.step()
            if epoch % verbose_every == 0 or epoch == 1:
                model.set_task_sequence(task_name, seed_k)
                vl  = DataLoader(task_data[task_name]['val'], batch_size=256)
                print(f'  Epoch {epoch:>3}/{epochs}  val={evaluate(model, vl, device):.3f}')
        model.save_lora(task_name)
        _eval_all_tasks(model, task_data, task_names, step_idx, acc_matrix, device, is_lora=True)
        row = '  '.join([f'T{j+1}:{acc_matrix[step_idx,j]*100:.1f}%'
                         for j in range(step_idx + 1)])
        print(f'  -> {row}')
        wgr_buf.add_task(task_data[task_name]['train'], task_name)
    return model, acc_matrix

print('All six training strategies defined.')


# In[ ]:


# ── Attack — Membership Inference Attack (MIA) ───────────────────────────
# Threshold-based MIA (Yeom et al. 2018).
# Score: s(x,y) = -CE(f(x), y)  — lower loss = more likely a training member.
# Members    = randomly sampled training windows per task.
# Non-members = randomly sampled test windows from the SAME task
#               (same subjects, unseen windows — hardest setting for the attacker).
#
# cdml_mode (only meaningful when is_cdml=True):
#   'oracle'  -> attacker knows the true per-task CDML seed (upper bound)
#   'no_seed' -> attacker guesses a fixed WRONG seed (NO_SEED_GUESS = 1),
#                per the corrected phase11 / phase11B procedure — NOT a
#                zero/null sequence.
def run_mia(model, task_data, task_names, device,
            is_cdml=False, is_lora=False, cdml_mode='none', n_samples=MIA_N_SAMPLES):
    results = {}
    for t_idx, task_name in enumerate(task_names):
        if is_lora:
            model.load_lora(task_name)
        if is_cdml:
            if cdml_mode == 'oracle':
                model.set_task_sequence(task_name, CDML_SEED_BASE + t_idx)
            elif cdml_mode == 'no_seed':
                model.set_task_sequence(task_name, NO_SEED_GUESS)

        train_ds = task_data[task_name]['train']
        test_ds  = task_data[task_name]['test']

        rng     = np.random.default_rng(RANDOM_SEED + 77 + t_idx)
        n_mem   = min(n_samples, len(train_ds))
        n_non   = min(n_samples, len(test_ds))
        mem_idx = rng.choice(len(train_ds), n_mem, replace=False)
        non_idx = rng.choice(len(test_ds),  n_non, replace=False)

        model.eval()
        with torch.no_grad():
            X_mem = torch.stack([train_ds[int(i)][0] for i in mem_idx]).to(device)
            y_mem = torch.stack([train_ds[int(i)][1] for i in mem_idx]).to(device)
            losses_mem = nn.CrossEntropyLoss(reduction='none')(
                model(X_mem), y_mem).cpu().float().numpy()

            X_non = torch.stack([test_ds[int(i)][0] for i in non_idx]).to(device)
            y_non = torch.stack([test_ds[int(i)][1] for i in non_idx]).to(device)
            losses_non = nn.CrossEntropyLoss(reduction='none')(
                model(X_non), y_non).cpu().float().numpy()

        m_arr  = -losses_mem    # higher = lower loss = more likely member
        nm_arr = -losses_non
        scores = np.concatenate([m_arr, nm_arr])
        labels = np.concatenate([np.ones(len(m_arr)), np.zeros(len(nm_arr))])
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc     = sk_auc(fpr, tpr)
        fnr         = 1 - tpr
        eer_idx     = np.nanargmin(np.abs(fpr - fnr))
        eer         = float(np.mean([fpr[eer_idx], fnr[eer_idx]]))
        results[task_name] = {
            'auc': roc_auc, 'eer': eer,
            'fpr': fpr,     'tpr': tpr,
            'm_scores': m_arr, 'nm_scores': nm_arr,
        }
    return results

print('MIA defined.')


# In[ ]:


# ── Attack — Feature Space Inference (FSI) ───────────────────────────────
# Corrected procedure (phase11 / phase11B / thesis "FSI cosine invariance
# trap" fix): the k-NN gallery is ALWAYS built at the true oracle key. Only
# the probe/query side varies. This models a correctly-keyed reference
# database (leaked or legitimately-computed enrollment templates) versus an
# attacker who must supply their own query key.
#
# query_mode: 'raw'     -> Std models, no CDML at all
#             'oracle'  -> attacker also knows the true key (upper bound)
#             'no_seed' -> attacker guesses NO_SEED_GUESS = 1, mismatched
#                          against the gallery's true key
@torch.no_grad()
def extract_embeddings(model, dataset, device, seq_mode='raw'):
    model.eval()
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    all_h, all_y = [], []
    for X_b, y_b in loader:
        X_b = X_b.to(device)
        if seq_mode == 'raw':
            h = model.backbone.embed(X_b) if hasattr(model, 'backbone') else model.embed(X_b)
        else:
            h = model.embed_modulated(X_b) if hasattr(model, 'embed_modulated') else model.embed(X_b)
        all_h.append(h.cpu().float().numpy())
        all_y.append(y_b.numpy())
    return np.concatenate(all_h), np.concatenate(all_y)


def run_feature_probe(model, task_data, task_names, device,
                       is_cdml=False, is_lora=False, query_mode='oracle', k=KNN_K):
    if is_cdml:
        tr_h_list, tr_y_list = [], []
        for t_idx, t_name in enumerate(task_names):
            if is_lora:
                model.load_lora(t_name)
            model.set_task_sequence(t_name, CDML_SEED_BASE + t_idx)
            h, y = extract_embeddings(model, task_data[t_name]['train'], device, 'oracle')
            tr_h_list.append(h); tr_y_list.append(y)
        tr_h = np.concatenate(tr_h_list); tr_y = np.concatenate(tr_y_list)
    else:
        all_tr = ConcatDataset([task_data[t]['train'] for t in task_names])
        tr_h, tr_y = extract_embeddings(model, all_tr, device, 'raw')

    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine', n_jobs=-1)
    knn.fit(tr_h, tr_y)

    results = {}
    for t_idx, task_name in enumerate(task_names):
        if is_cdml:
            if is_lora:
                model.load_lora(task_name)
            if query_mode == 'oracle':
                model.set_task_sequence(task_name, CDML_SEED_BASE + t_idx)
            elif query_mode == 'no_seed':
                model.set_task_sequence(task_name, NO_SEED_GUESS)
        te_h, te_y = extract_embeddings(model, task_data[task_name]['test'], device,
                                        'raw' if not is_cdml else query_mode)
        preds = knn.predict(te_h)
        top1  = float((preds == te_y).mean())
        conf  = float(knn.predict_proba(te_h).max(axis=1).mean())
        # 'acc' kept as the primary key (used by summary/plots); 'chance' vs.
        # the full gallery label space (matches the phase11 convention).
        results[task_name] = {'acc': top1, 'top1_acc': top1, 'mean_conf': conf}
    return results

print('FSI (run_feature_probe) defined.')


# In[ ]:


# ── Attack — Backdoor (sinusoidal trigger) ────────────────────────────────
def add_trigger(X, amp=BD_TRIGGER_AMP, hz=BD_TRIGGER_HZ, ch=BD_TRIGGER_CH):
    X_t = X.clone()
    W   = X_t.shape[-1]
    t   = torch.arange(W, dtype=torch.float32, device=X.device)
    X_t[:, ch, :] += amp * torch.sin(2 * np.pi * hz * t / W)
    return X_t


def run_backdoor(ModelClass, task_data, task_names, cfg, device,
                 epochs=None, label='', is_cdml=False):
    # Trains a FRESH model of the same class/architecture on a poisoned
    # version of the last task only (single-task fit, mirrors phase21's
    # backdoor procedure). For CDML-based classes the model is trained and
    # evaluated at the true oracle key for that task.
    if epochs is None: epochs = EPOCHS
    target_idx = cfg.get('bd_target', 0)
    last_task  = task_names[-1]
    train_ds   = task_data[last_task]['train']
    test_ds    = task_data[last_task]['test']

    # Build poisoned training set for the last task
    n_poison    = max(1, int(len(train_ds) * BD_POISON_FRAC))
    rng         = np.random.default_rng(RANDOM_SEED)
    poison_idx  = set(rng.choice(len(train_ds), n_poison, replace=False).tolist())
    Xp, yp      = [], []
    for i in range(len(train_ds)):
        x, y = train_ds[i]
        if i in poison_idx:
            x = add_trigger(x.unsqueeze(0)).squeeze(0)
            yp.append(target_idx)
        else:
            yp.append(y.item())
        Xp.append(x)
    poisoned_ds  = TensorDataset(torch.stack(Xp), torch.tensor(yp, dtype=torch.long))

    # Train a fresh model on poisoned last task
    model_p   = _make_model(ModelClass, cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    if is_cdml:
        seed_k = CDML_SEED_BASE + (len(task_names) - 1)
        model_p.set_task_sequence(last_task, seed_k)
    optimizer = optim.Adam(model_p.parameters(), lr=LR_INIT)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_DECAY)
    loader    = DataLoader(poisoned_ds, batch_size=BATCH_SIZE, shuffle=True)
    for epoch in range(1, epochs + 1):
        if is_cdml:
            model_p.set_task_sequence(last_task, seed_k)
        _run_epoch(model_p, loader, optimizer, criterion, device)
        scheduler.step()

    # Clean accuracy on test set
    if is_cdml:
        model_p.set_task_sequence(last_task, seed_k)
    clean_acc = evaluate(model_p, DataLoader(test_ds, batch_size=256), device)

    # Attack success rate: triggered inputs classified as target
    model_p.eval()
    X_all     = torch.stack([test_ds[i][0] for i in range(len(test_ds))])
    X_trig    = add_trigger(X_all).to(device)
    with torch.no_grad():
        preds = model_p(X_trig).argmax(1).cpu().numpy()
    asr = float((preds == target_idx).mean())
    print(f'  [{label}] Backdoor: clean_acc={clean_acc:.3f}  ASR={asr:.3f}')
    return {'clean_acc': clean_acc, 'asr': asr, 'target': target_idx}

print('Backdoor attack defined.')


# In[ ]:


def compute_metrics(acc_matrix):
    # Average accuracy: mean over all tasks at end of training
    avg_acc = float(np.nanmean(acc_matrix[-1, :]))
    # BWT: mean drop in per-task accuracy after learning subsequent tasks
    n = acc_matrix.shape[0]
    bwt_vals = [acc_matrix[-1, i] - acc_matrix[i, i]
                for i in range(n - 1)
                if not np.isnan(acc_matrix[-1, i]) and not np.isnan(acc_matrix[i, i])]
    bwt = float(np.mean(bwt_vals)) if bwt_vals else 0.0
    return avg_acc, bwt


# Model registry: (label, train_fn, ModelClass_for_backdoor, is_lora, is_cdml)
STRATEGY_REGISTRY = [
    ('Mamba-Std-0%',
        partial(train_mamba_std, replay_frac=0.0),          GaitMamba,           False, False),
    ('Mamba-Std-15%',
        partial(train_mamba_std, replay_frac=STD_REPLAY_FRAC), GaitMamba,        False, False),
    ('Mamba-CDML',
        train_mamba_cdml,          GaitMamba_CDML,      False, True),
    ('Mamba-WGR-CDML',
        train_mamba_wgr_cdml,      GaitMamba_CDML,      False, True),
    ('Mamba-CDML+LoRA',
        train_mamba_cdml_lora,     GaitMamba_CDML_LoRA, True,  True),
    ('Mamba-WGR-CDML+LoRA',
        train_mamba_wgr_cdml_lora, GaitMamba_CDML_LoRA, True,  True),
]


def run_experiments(ds_name, cfg, device, epochs=None):
    print(f'\n{"=" * 70}')
    print(f'  DATASET: {cfg["name"]}')
    print(f'{"=" * 70}')

    # Load & prepare data
    print('\n[1/5] Loading data...')
    try:
        X_tr, y_tr, X_te, y_te = load_raw_data(cfg)
        print(f'  Train: {X_tr.shape}  Test: {X_te.shape}')
    except Exception as e:
        print(f'  [!] Cannot load {ds_name}: {e}')
        return None

    print('[2/5] Building task datasets...')
    task_data, l2i = make_task_datasets(X_tr, y_tr, X_te, y_te, cfg)
    task_names     = list(cfg['task_splits'].keys())

    results = {}

    # Train all six strategies
    print('[3/5] Training strategies...')
    for label, train_fn, _, is_lora, is_cdml in STRATEGY_REGISTRY:
        print(f'\n  --- {label} ---')
        model, acc_mat = train_fn(task_data, task_names, cfg, device,
                                  epochs=epochs, label=label)
        avg_acc, bwt   = compute_metrics(acc_mat)
        results[label] = {'model': model, 'acc_matrix': acc_mat,
                          'avg_acc': avg_acc, 'bwt': bwt,
                          'is_lora': is_lora, 'is_cdml': is_cdml}

    # Attack 1 — MIA
    print('\n[4a/5] Running MIA...')
    for label, _, _, is_lora, is_cdml in STRATEGY_REGISTRY:
        model = results[label]['model']
        if is_cdml:
            for mode in ('no_seed', 'oracle'):
                mia_res = run_mia(model, task_data, task_names, device,
                                  is_cdml=True, is_lora=is_lora, cdml_mode=mode)
                results[label][f'mia_{mode}'] = mia_res
                avg_auc = np.mean([r['auc'] for r in mia_res.values()])
                print(f'  {label} [{mode}]  avg AUC={avg_auc:.3f}')
            # Restore final-task state
            if is_lora:
                model.load_lora(task_names[-1])
            model.set_task_sequence(task_names[-1], CDML_SEED_BASE + len(task_names) - 1)
        else:
            mia_res = run_mia(model, task_data, task_names, device,
                              is_cdml=False, is_lora=False, cdml_mode='none')
            # No attacker-key distinction for Std models — store under both
            # keys so downstream plotting/summary code can treat all six
            # strategies uniformly.
            results[label]['mia_no_seed'] = mia_res
            results[label]['mia_oracle']  = mia_res
            avg_auc = np.mean([r['auc'] for r in mia_res.values()])
            print(f'  {label}  avg AUC={avg_auc:.3f}')

    # Attack 2 — FSI
    print('\n[4b/5] Running FSI...')
    for label, _, _, is_lora, is_cdml in STRATEGY_REGISTRY:
        model = results[label]['model']
        if is_cdml:
            for mode in ('no_seed', 'oracle'):
                fsi_res = run_feature_probe(model, task_data, task_names, device,
                                            is_cdml=True, is_lora=is_lora, query_mode=mode)
                results[label][f'fsi_{mode}'] = fsi_res
                avg_acc_fsi = np.mean([r['acc'] for r in fsi_res.values()])
                print(f'  {label} [{mode}]  avg k-NN acc={avg_acc_fsi:.3f}')
            if is_lora:
                model.load_lora(task_names[-1])
            model.set_task_sequence(task_names[-1], CDML_SEED_BASE + len(task_names) - 1)
        else:
            fsi_res = run_feature_probe(model, task_data, task_names, device,
                                        is_cdml=False, is_lora=False, query_mode='raw')
            results[label]['fsi_no_seed'] = fsi_res
            results[label]['fsi_oracle']  = fsi_res
            avg_acc_fsi = np.mean([r['acc'] for r in fsi_res.values()])
            print(f'  {label}  avg k-NN acc={avg_acc_fsi:.3f}')

    # Attack 3 — Backdoor
    print('\n[5/5] Running Backdoor...')
    for label, _, ModelClass, _, is_cdml in STRATEGY_REGISTRY:
        bd = run_backdoor(ModelClass, task_data, task_names, cfg, device,
                          epochs=epochs, label=label, is_cdml=is_cdml)
        results[label]['backdoor'] = bd

    # Summary
    print(f'\n{"─" * 55}')
    print(f'  Summary — {cfg["name"]}')
    print(f'  {"Strategy":<26} {"Avg Acc":>9} {"BWT":>8}')
    print(f'  {"─" * 45}')
    for label, *_ in STRATEGY_REGISTRY:
        r = results[label]
        print(f'  {label:<26} {r["avg_acc"]*100:>8.2f}% {r["bwt"]*100:>7.2f}%')

    results['_meta'] = {'task_data': task_data, 'task_names': task_names,
                        'l2i': l2i, 'cfg': cfg}
    return results

print('run_experiments() defined.')


# In[ ]:


# Set ACTIVE_DATASETS to the subset you want to run.
ACTIVE_DATASETS = ['Dataset_1', 'UCI_HAR', 'WISDM']

ALL_RESULTS = {}
for ds_name in ACTIVE_DATASETS:
    cfg = DATASET_CONFIGS[ds_name]
    res = run_experiments(ds_name, cfg, DEVICE, epochs=EPOCHS)
    if res is not None:
        ALL_RESULTS[ds_name] = res

print(f'\nFinished {len(ALL_RESULTS)}/{len(ACTIVE_DATASETS)} datasets.')


# In[ ]:


MODEL_COLORS = {
    'Mamba-Std-0%':          '#E74C3C',
    'Mamba-Std-15%':         '#E67E22',
    'Mamba-CDML':            '#3498DB',
    'Mamba-WGR-CDML':        '#27AE60',
    'Mamba-CDML+LoRA':       '#1ABC9C',
    'Mamba-WGR-CDML+LoRA':   '#9B59B6',
}
STRATEGIES = [s[0] for s in STRATEGY_REGISTRY]


def plot_forgetting_matrices(ds_results, ds_name):
    cfg        = ds_results['_meta']['cfg']
    task_names = ds_results['_meta']['task_names']
    n_tasks    = len(task_names)
    n_models   = len(STRATEGIES)

    fig, axes = plt.subplots(1, n_models, figsize=(4.6 * n_models, 4.6))
    fig.suptitle(f'{cfg["name"]} — Accuracy matrices\n'
                 f'Row = after step S, Col = task T accuracy',
                 fontsize=12, fontweight='bold')

    for ax, label in zip(axes, STRATEGIES):
        mat    = ds_results[label]['acc_matrix']
        masked = np.ma.masked_invalid(mat * 100)
        im     = ax.imshow(masked, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xticks(range(n_tasks))
        ax.set_xticklabels([f'T{i+1}' for i in range(n_tasks)])
        ax.set_yticks(range(n_tasks))
        ax.set_yticklabels([f'After S{i+1}' for i in range(n_tasks)])
        ax.set_title(label, color=MODEL_COLORS[label], fontsize=9, fontweight='bold')
        for i in range(n_tasks):
            for j in range(n_tasks):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f'{v*100:.0f}', ha='center', va='center',
                            fontsize=8, color='black' if v > 0.3 else 'white')
    plt.tight_layout()
    plt.savefig(f'fig_{ds_name}_matrices.png', bbox_inches='tight')
    plt.show()


def plot_accuracy_summary(ds_results, ds_name):
    cfg        = ds_results['_meta']['cfg']
    task_names = ds_results['_meta']['task_names']
    n_tasks    = len(task_names)
    steps      = range(1, n_tasks + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(f'{cfg["name"]} — Performance summary', fontsize=12, fontweight='bold')

    # T1 forgetting curve
    ax = axes[0]
    for label in STRATEGIES:
        mat = ds_results[label]['acc_matrix']
        t1  = [mat[s, 0] * 100 for s in range(n_tasks) if not np.isnan(mat[s, 0])]
        ax.plot(list(steps)[:len(t1)], t1, color=MODEL_COLORS[label],
                marker='o', lw=2, label=label)
    ax.set_title('Task 1 accuracy over time\n(lower = more forgetting)')
    ax.set_xticks(list(steps))
    ax.set_xticklabels([f'After T{s}' for s in steps], rotation=15)
    ax.set_ylabel('Accuracy (%)')
    ax.legend(fontsize=7)

    # Average accuracy across seen tasks
    ax = axes[1]
    for label in STRATEGIES:
        mat  = ds_results[label]['acc_matrix']
        avgs = [np.nanmean(mat[s, :s + 1]) * 100 for s in range(n_tasks)]
        ax.plot(list(steps), avgs, color=MODEL_COLORS[label],
                marker='s', lw=2, label=label)
    ax.set_title('Average accuracy across seen tasks\n(higher = better CL)')
    ax.set_xticks(list(steps))
    ax.set_xticklabels([f'After T{s}' for s in steps], rotation=15)
    ax.set_ylabel('Avg Accuracy (%)')
    ax.legend(fontsize=7)

    # Final Avg Acc + BWT bar chart
    ax   = axes[2]
    x    = np.arange(len(STRATEGIES))
    avgs = [ds_results[s]['avg_acc'] * 100 for s in STRATEGIES]
    bwts = [ds_results[s]['bwt'] * 100 for s in STRATEGIES]
    bars = ax.bar(x - 0.2, avgs, 0.35, label='Avg Acc (%)',
                  color=[MODEL_COLORS[s] for s in STRATEGIES], alpha=0.85)
    ax.bar(x + 0.2, bwts, 0.35, label='BWT (%)',
           color=[MODEL_COLORS[s] for s in STRATEGIES], alpha=0.45, hatch='//')
    for bar, v in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5, f'{v:.1f}',
                ha='center', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('Mamba-', '') for s in STRATEGIES], rotation=25, fontsize=8)
    ax.set_title('Final Avg Acc & BWT\n(solid=Acc, hatch=BWT)')
    ax.legend(fontsize=8)
    ax.set_ylabel('%')
    plt.tight_layout()
    plt.savefig(f'fig_{ds_name}_accuracy.png', bbox_inches='tight')
    plt.show()


def plot_attacks(ds_results, ds_name):
    cfg        = ds_results['_meta']['cfg']
    task_names = ds_results['_meta']['task_names']
    task_x     = [f'T{i+1}' for i in range(len(task_names))]

    fig = plt.figure(figsize=(18, 9.5))
    fig.suptitle(f'{cfg["name"]} — Privacy attacks (MIA / FSI / Backdoor)',
                 fontsize=13, fontweight='bold')
    gs  = gridspec.GridSpec(2, 4, figure=fig)

    # FSI k-NN — no_seed
    ax = fig.add_subplot(gs[0, 0])
    for label in STRATEGIES:
        vals = [ds_results[label]['fsi_no_seed'][t]['acc'] for t in task_names]
        ax.plot(task_x, vals, color=MODEL_COLORS[label], marker='^', lw=2, label=label)
    ax.axhline(1.0 / cfg['n_classes'], ls=':', color='gray', alpha=0.7,
               label=f'Chance (~{100/cfg["n_classes"]:.1f}%)')
    ax.set_title('FSI — k-NN acc (no-seed, seed=1)\n(lower = more privacy)')
    ax.set_ylabel('k-NN Acc'); ax.set_ylim(0, 1.05); ax.legend(fontsize=6)

    # FSI k-NN — oracle
    ax = fig.add_subplot(gs[0, 1])
    for label in STRATEGIES:
        vals = [ds_results[label]['fsi_oracle'][t]['acc'] for t in task_names]
        ax.plot(task_x, vals, color=MODEL_COLORS[label], marker='^', lw=2, label=label)
    ax.set_title('FSI — k-NN acc (oracle)\n(lower = more privacy)')
    ax.set_ylabel('k-NN Acc'); ax.set_ylim(0, 1.05); ax.legend(fontsize=6)

    # Backdoor: Clean Acc vs ASR grouped bar
    ax     = fig.add_subplot(gs[0, 2:])
    x      = np.arange(len(STRATEGIES))
    cleans = [ds_results[s]['backdoor']['clean_acc'] * 100 for s in STRATEGIES]
    asrs   = [ds_results[s]['backdoor']['asr'] * 100 for s in STRATEGIES]
    bars1  = ax.bar(x - 0.18, cleans, 0.34, label='Clean acc (%)',
                    color=[MODEL_COLORS[s] for s in STRATEGIES], alpha=0.85)
    bars2  = ax.bar(x + 0.18, asrs, 0.34, label='ASR (%)', color='#E74C3C', alpha=0.7)
    for bar, v in zip(bars1, cleans):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5, f'{v:.1f}',
                ha='center', fontsize=7)
    for bar, v in zip(bars2, asrs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5, f'{v:.1f}',
                ha='center', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('Mamba-', '') for s in STRATEGIES], rotation=20, fontsize=8)
    ax.set_title('Backdoor — Clean accuracy vs Attack Success Rate (ASR)\n'
                 '(lower ASR with maintained clean acc = better defence)')
    ax.set_ylabel('%'); ax.set_ylim(0, 115); ax.legend(fontsize=8)

    # MIA AUC — no_seed
    ax = fig.add_subplot(gs[1, 0])
    ax.axhline(0.5, ls='--', color='gray', alpha=0.6, label='Random (0.5)')
    for label in STRATEGIES:
        vals = [ds_results[label]['mia_no_seed'][t]['auc'] for t in task_names]
        ax.plot(task_x, vals, color=MODEL_COLORS[label], marker='o', lw=2, label=label)
    ax.set_title('MIA — AUC (no-seed, seed=1)'); ax.set_ylabel('AUC')
    ax.set_ylim(0.4, 1.05); ax.legend(fontsize=6)

    # MIA AUC — oracle
    ax = fig.add_subplot(gs[1, 1])
    ax.axhline(0.5, ls='--', color='gray', alpha=0.6)
    for label in STRATEGIES:
        vals = [ds_results[label]['mia_oracle'][t]['auc'] for t in task_names]
        ax.plot(task_x, vals, color=MODEL_COLORS[label], marker='o', lw=2, label=label)
    ax.set_title('MIA — AUC (oracle)'); ax.set_ylabel('AUC')
    ax.set_ylim(0.4, 1.05); ax.legend(fontsize=6)

    # MIA EER — no_seed
    ax = fig.add_subplot(gs[1, 2])
    for label in STRATEGIES:
        vals = [ds_results[label]['mia_no_seed'][t]['eer'] for t in task_names]
        ax.plot(task_x, vals, color=MODEL_COLORS[label], marker='s', lw=2, label=label)
    ax.set_title('MIA — EER (no-seed)\n(higher = harder for attacker)')
    ax.set_ylabel('EER'); ax.set_ylim(0, 0.6); ax.legend(fontsize=6)

    # MIA EER — oracle
    ax = fig.add_subplot(gs[1, 3])
    for label in STRATEGIES:
        vals = [ds_results[label]['mia_oracle'][t]['eer'] for t in task_names]
        ax.plot(task_x, vals, color=MODEL_COLORS[label], marker='s', lw=2, label=label)
    ax.set_title('MIA — EER (oracle)\n(higher = harder for attacker)')
    ax.set_ylabel('EER'); ax.set_ylim(0, 0.6); ax.legend(fontsize=6)

    plt.tight_layout()
    plt.savefig(f'fig_{ds_name}_attacks.png', bbox_inches='tight')
    plt.show()

print('Visualization functions defined.')


# In[ ]:


for ds_name, ds_res in ALL_RESULTS.items():
    if ds_res is None:
        continue
    print(f'\n{"=" * 60}')
    print(f'  Plots: {ds_res["_meta"]["cfg"]["name"]}')
    print(f'{"=" * 60}')
    plot_forgetting_matrices(ds_res, ds_name)
    plot_accuracy_summary(ds_res, ds_name)
    plot_attacks(ds_res, ds_name)


# In[ ]:


print('\n' + '=' * 85)
print('  CROSS-DATASET SUMMARY — FB-Mamba Six-Model Security Evaluation')
print('=' * 85)
print(f'  {"Dataset":<22} {"Strategy":<22} {"Avg Acc":>9} {"BWT":>8} '
      f'{"MIA AUC":>9} {"FSI Acc":>9} {"BD ASR":>8}')
print(f'  {"─" * 82}')

for ds_name, ds_res in ALL_RESULTS.items():
    if ds_res is None:
        continue
    task_names = ds_res['_meta']['task_names']
    ds_label   = ds_res['_meta']['cfg']['name']
    for i, label in enumerate(STRATEGIES):
        r        = ds_res[label]
        avg_mia  = np.mean([r['mia_no_seed'][t]['auc'] for t in task_names])
        avg_fsi  = np.mean([r['fsi_no_seed'][t]['acc'] for t in task_names])
        bd_asr   = r['backdoor']['asr']
        col_ds   = ds_label if i == 0 else ''
        print(f'  {col_ds:<22} {label:<22} {r["avg_acc"]*100:>8.2f}% '
              f'{r["bwt"]*100:>7.2f}%  {avg_mia:>8.3f}  {avg_fsi:>8.3f} '
              f'{bd_asr*100:>7.1f}%')
    print(f'  {"─" * 82}')

print()
print('Column guide:')
print('  Avg Acc  — mean task accuracy after completing all 4 tasks')
print('  BWT      — backward transfer (negative = forgetting)')
print('  MIA AUC  — membership inference AUC, no-seed attacker (seed=1) for')
print('             CDML models / single-run AUC for Std models')
print('             (0.5 = random; lower = more private)')
print('  FSI Acc  — k-NN re-identification accuracy, no-seed attacker (seed=1)')
print('             (lower = more private)')
print('  BD ASR   — backdoor attack success rate (lower = more robust)')