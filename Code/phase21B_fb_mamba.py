#!/usr/bin/env python
# coding: utf-8

# # Phase 21b — FB-Mamba Standard Continual Learning (15% Replay)
# 
# **Architecture:** FB-Mamba (Wu et al., 2026)
# **Strategy:** Standard fine-tuning with 15% Replay buffer (Baseline)
# **Evaluation:** Accuracy and Privacy Inference across 3 datasets
# 
# ---

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, ConcatDataset
from sklearn.metrics import roc_curve, auc as sk_auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import os, warnings
warnings.filterwarnings('ignore')

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

# ── Reproducibility ───────────────────────────────────────────────────────
RANDOM_SEED    = 27
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── Training ──────────────────────────────────────────────────────────────
EPOCHS         = 50      
LR_INIT        = 1e-3
LR_DECAY       = 0.98
BATCH_SIZE     = 64
VAL_SPLIT      = 0.15
REPLAY_FRAC    = 0.15    # 15% replay of past data

# ── FB-Mamba backbone ─────────────────────────────────────────────────────
EMBED_DIM      = 128     
D_MODEL        = 16      
NUM_LAYERS     = 2       
D_STATE        = 2       
D_CONV         = 2       
EXPAND         = 2       

# ── Attacks ───────────────────────────────────────────────────────────────
IIA_N_QUERIES  = 20      
KNN_K          = 5       
BD_POISON_FRAC = 0.15    
BD_TRIGGER_AMP = 0.5     
BD_TRIGGER_HZ  = 8       
BD_TRIGGER_CH  = 2       

print('Global configuration set.')

# ── Datasets Config ───────────────────────────────────────────────────────
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
        'bd_target': 90,   
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


# ── Data Loaders ──────────────────────────────────────────────────────────
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

def load_ucihar(cfg):
    def _split(d, split):
        axes = ['total_acc_x', 'total_acc_y', 'total_acc_z',
                'body_gyro_x', 'body_gyro_y', 'body_gyro_z']
        X   = np.stack([np.loadtxt(f'{d}/{split}/Inertial Signals/{a}_{split}.txt') for a in axes], axis=1).astype(np.float32)
        y   = np.loadtxt(f'{d}/{split}/subject_{split}.txt', dtype=int)
        act = np.loadtxt(f'{d}/{split}/y_{split}.txt',       dtype=int)
        return X, y, act
    X_tr, y_tr, act_tr = _split(cfg['data_dir'], 'train')
    X_te, y_te, act_te = _split(cfg['data_dir'], 'test')
    
    X_all   = np.concatenate([X_tr, X_te], axis=0)
    y_all   = np.concatenate([y_tr, y_te], axis=0)
    act_all = np.concatenate([act_tr, act_te], axis=0)
    
    walk_mask = np.isin(act_all, [1, 2, 3])
    X_all     = X_all[walk_mask]
    y_all     = y_all[walk_mask]
    
    X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.2, random_state=RANDOM_SEED, stratify=y_all)
    mu  = X_tr.mean(axis=(0, 2), keepdims=True)
    std = X_tr.std(axis=(0, 2),  keepdims=True) + 1e-8
    return (X_tr - mu) / std, y_tr, (X_te - mu) / std, y_te

def load_wisdm(cfg):
    W   = cfg['window_size']
    step = W // 2

    def _read_dir(sensor_dir, key):
        out = {}
        for fn in sorted(os.listdir(sensor_dir)):
            if not fn.endswith('.txt'):
                continue
            sid = int(fn.split('_')[1]) - 1599 
            rows = []
            with open(os.path.join(sensor_dir, fn), encoding='utf-8', errors='ignore') as fh:
                for line in fh:
                    line = line.strip().rstrip(';')
                    if not line or not line[0].isdigit():
                        continue
                    parts = line.split(',')
                    if len(parts) >= 6:
                        try: rows.append([float(p.strip()) for p in parts[3:6]])
                        except ValueError: pass
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
        if 'acc' not in s or 'gyro' not in s: continue
        n = min(len(s['acc']), len(s['gyro']))
        data = np.concatenate([s['acc'][:n], s['gyro'][:n]], axis=1)
        for start in range(0, n - W + 1, step):
            X_all.append(data[start:start + W].T)
            y_all.append(sid)

    X = np.stack(X_all).astype(np.float32)
    y = np.array(y_all, dtype=int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    mu  = X_tr.mean(axis=(0, 2), keepdims=True)
    std = X_tr.std(axis=(0, 2),  keepdims=True) + 1e-8
    return (X_tr - mu) / std, y_tr, (X_te - mu) / std, y_te

def load_raw_data(cfg):
    return {'dataset1': load_dataset1, 'ucihar': load_ucihar, 'wisdm': load_wisdm}[cfg['loader']](cfg)

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
    return task_data, l2i


# ── FBMamba Backbone Core ──────────────────────────────────────────────────
def selective_scan(u, delta, A, B, C, D):
    B_batch, L, d_inner = u.shape
    dt   = delta.unsqueeze(-1)
    Aexp = torch.exp(dt * A.unsqueeze(0).unsqueeze(0))
    Bu   = dt * B.unsqueeze(2) * u.unsqueeze(-1)
    h    = torch.zeros(B_batch, d_inner, A.shape[1], device=u.device, dtype=u.dtype)
    ys   = []
    for t in range(L):
        h  = Aexp[:, t] * h + Bu[:, t]
        yt = (h * C[:, t].unsqueeze(1)).sum(-1)
        ys.append(yt)
    y = torch.stack(ys, dim=1)
    return y + u * D.unsqueeze(0).unsqueeze(0)

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_inner  = int(expand * d_model)
        self.d_state  = d_state
        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d   = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                  padding=d_conv - 1, groups=self.d_inner, bias=True)
        self.x_proj   = nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)
        self.dt_proj  = nn.Linear(self.d_inner, self.d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
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
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.mamba_fwd     = MambaBlock(d_model, d_state, d_conv, expand)
        self.mamba_bwd     = MambaBlock(d_model, d_state, d_conv, expand)
        self.fusion_linear = nn.Linear(d_model * 2, d_model)
        self.out_proj      = nn.Linear(d_model, d_model)
        self.norm          = nn.LayerNorm(d_model)

    def forward(self, H):
        H_fwd   = self.mamba_fwd(H)
        H_bwd   = torch.flip(self.mamba_bwd(torch.flip(H, dims=[1])), dims=[1])
        alpha   = torch.sigmoid(self.fusion_linear(torch.cat([H_fwd, H_bwd], dim=-1)))
        H_fused = alpha * H_fwd + (1.0 - alpha) * H_bwd
        return self.norm(self.out_proj(H_fused))

class GaitMamba(nn.Module):
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
        h = x.transpose(1, 2)
        h = self.input_proj(h)
        for block in self.blocks:
            h = block(h)
        h = h.mean(dim=1)
        return self.embedding(h)

    def forward(self, x):
        return self.classifier(self.embed(x))


# ── Training & Evaluators ─────────────────────────────────────────────────
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

def train_mamba_std_15(task_data, task_names, cfg, device, epochs=None, verbose_every=25, label='Mamba-Std-15%'):
    if epochs is None: epochs = EPOCHS
    n_tasks    = len(task_names)
    model      = GaitMamba(n_channels=cfg['n_channels'], n_classes=cfg['n_classes']).to(device)
    criterion  = nn.CrossEntropyLoss()
    acc_matrix = np.full((n_tasks, n_tasks), np.nan)
    torch.manual_seed(RANDOM_SEED)

    past_train_datasets = []

    for step_idx, task_name in enumerate(task_names):
        print(f'[{label}] Step {step_idx + 1}/{n_tasks}: {task_name}')
        
        # Build 15% replay loader
        datasets_to_mix = [task_data[task_name]['train']]
        for past_ds in past_train_datasets:
            n_replay = max(1, int(len(past_ds) * REPLAY_FRAC))
            indices  = torch.randperm(len(past_ds))[:n_replay].tolist()
            datasets_to_mix.append(torch.utils.data.Subset(past_ds, indices))
            
        combined_ds = ConcatDataset(datasets_to_mix)
        loader      = DataLoader(combined_ds, batch_size=BATCH_SIZE, shuffle=True)
        
        optimizer = optim.Adam(model.parameters(), lr=LR_INIT)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_DECAY)
        
        for epoch in range(1, epochs + 1):
            _run_epoch(model, loader, optimizer, criterion, device)
            scheduler.step()
            if epoch % verbose_every == 0 or epoch == 1:
                vl = DataLoader(task_data[task_name]['val'], batch_size=256)
                print(f'  Epoch {epoch:>3}/{epochs}  val={evaluate(model, vl, device):.3f}')
                
        past_train_datasets.append(task_data[task_name]['train'])
        
        for ei, et in enumerate(task_names[:step_idx + 1]):
            tl = DataLoader(task_data[et]['test'], batch_size=256)
            acc_matrix[step_idx, ei] = evaluate(model, tl, device)
            
        row = '  '.join([f'T{j+1}:{acc_matrix[step_idx,j]*100:.1f}%' for j in range(step_idx + 1)])
        print(f'  -> {row}')
        
    return model, acc_matrix


# ── Attacks ───────────────────────────────────────────────────────────────
@torch.no_grad()
def iia_score(model, X_windows, subj_idx, device):
    model.eval()
    X_windows = X_windows.to(device)
    y_true    = torch.full((len(X_windows),), subj_idx, dtype=torch.long, device=device)
    losses    = nn.CrossEntropyLoss(reduction='none')(model(X_windows), y_true).cpu().float().numpy()
    return float(-losses.mean())

def run_iia(model, task_data, task_names, task_splits, l2i, device, n_queries=IIA_N_QUERIES):
    results = {}
    for task_name in task_names:
        lo, hi      = task_splits[task_name]
        task_subjs  = [l2i[s] for s in range(lo, hi + 1) if s in l2i]
        test_ds     = task_data[task_name]['test']

        member_scores = []
        for si in task_subjs:
            indices = [i for i in range(len(test_ds)) if test_ds[i][1].item() == si]
            if len(indices) == 0: continue
            X_mem = torch.stack([test_ds[i][0] for i in indices])
            member_scores.append(iia_score(model, X_mem[:n_queries], si, device))

        nonmember_scores = []
        for other in [t for t in task_names if t != task_name]:
            lo2, hi2  = task_splits[other]
            oth_subjs = [l2i[s] for s in range(lo2, hi2 + 1) if s in l2i]
            oth_ds    = task_data[other]['test']
            
            for si in oth_subjs:
                indices = [i for i in range(len(oth_ds)) if oth_ds[i][1].item() == si]
                if len(indices) == 0: continue
                X_nm = torch.stack([oth_ds[i][0] for i in indices])
                nonmember_scores.append(iia_score(model, X_nm[:n_queries], si, device))

        m_arr  = np.array(member_scores)
        nm_arr = np.array(nonmember_scores[:len(m_arr)])
        if len(m_arr) < 2 or len(nm_arr) < 2:
            results[task_name] = {'auc': 0.5, 'eer': 0.5}
            continue
            
        labels = np.concatenate([np.ones(len(m_arr)),  np.zeros(len(nm_arr))])
        scores = np.concatenate([m_arr, nm_arr])
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc     = sk_auc(fpr, tpr)
        fnr         = 1 - tpr
        eer_idx     = np.nanargmin(np.abs(fpr - fnr))
        eer         = float(np.mean([fpr[eer_idx], fnr[eer_idx]]))
        results[task_name] = {'auc': roc_auc, 'eer': eer}
    return results

def run_mia(model, task_data, task_names, device, n_samples=200):
    results = {}
    for t_idx, task_name in enumerate(task_names):
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
            losses_mem = nn.CrossEntropyLoss(reduction='none')(model(X_mem), y_mem).cpu().float().numpy()

            X_non = torch.stack([test_ds[int(i)][0] for i in non_idx]).to(device)
            y_non = torch.stack([test_ds[int(i)][1] for i in non_idx]).to(device)
            losses_non = nn.CrossEntropyLoss(reduction='none')(model(X_non), y_non).cpu().float().numpy()

        m_arr  = -losses_mem
        nm_arr = -losses_non
        scores = np.concatenate([m_arr, nm_arr])
        labels = np.concatenate([np.ones(len(m_arr)), np.zeros(len(nm_arr))])
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc     = sk_auc(fpr, tpr)
        fnr         = 1 - tpr
        eer_idx     = np.nanargmin(np.abs(fpr - fnr))
        eer         = float(np.mean([fpr[eer_idx], fnr[eer_idx]]))
        results[task_name] = {'auc': roc_auc, 'eer': eer}
    return results

def run_fsi(model, task_data, task_names, cfg, device):
    results = {}
    for task_name in task_names:
        if len(task_data[task_name]['test']) == 0:
            results[task_name] = {'acc': 1.0 / max(1, cfg['n_classes']), 'chance': 1.0 / max(1, cfg['n_classes'])}
            continue
        
        def get_embs(ds):
            Xs, ys = [], []
            for Xb, yb in DataLoader(ds, batch_size=256):
                model.eval()
                with torch.no_grad():
                    Xs.append(model.embed(Xb.to(device)).cpu().numpy())
                ys.append(yb.numpy())
            return np.concatenate(Xs), np.concatenate(ys)

        X_tr_e, y_tr_e = get_embs(task_data[task_name]['train'])
        X_te_e, y_te_e = get_embs(task_data[task_name]['test'])
        n_cls = len(np.unique(y_tr_e))
        k     = min(KNN_K, max(1, len(X_tr_e) // max(1, n_cls) - 1))
        
        try:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_tr_e, y_tr_e)
            acc = knn.score(X_te_e, y_te_e)
        except Exception:
            acc = 1.0 / max(1, n_cls)
            
        results[task_name] = {'acc': acc, 'chance': 1.0 / max(1, n_cls)}
    return results

def add_trigger(X, amp=BD_TRIGGER_AMP, hz=BD_TRIGGER_HZ, ch=BD_TRIGGER_CH):
    X_t = X.clone()
    W   = X_t.shape[-1]
    t   = torch.arange(W, dtype=torch.float32, device=X.device)
    X_t[:, ch, :] += amp * torch.sin(2 * np.pi * hz * t / W)
    return X_t

def run_backdoor(task_data, task_names, cfg, device, epochs=None, label=''):
    if epochs is None: epochs = EPOCHS
    target_idx = cfg.get('bd_target', 0)
    last_task  = task_names[-1]
    train_ds   = task_data[last_task]['train']
    test_ds    = task_data[last_task]['test']

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

    model_p   = GaitMamba(n_channels=cfg['n_channels'], n_classes=cfg['n_classes']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_p.parameters(), lr=LR_INIT)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_DECAY)
    loader    = DataLoader(poisoned_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    for epoch in range(1, epochs + 1):
        _run_epoch(model_p, loader, optimizer, criterion, device)
        scheduler.step()

    clean_acc = evaluate(model_p, DataLoader(test_ds, batch_size=256), device)

    model_p.eval()
    X_all  = torch.stack([test_ds[i][0] for i in range(len(test_ds))])
    X_trig = add_trigger(X_all).to(device)
    with torch.no_grad():
        preds = model_p(X_trig).argmax(1).cpu().numpy()
        
    asr = float((preds == target_idx).mean())
    print(f'  [{label}] Backdoor: clean_acc={clean_acc:.3f}  ASR={asr:.3f}')
    return {'clean_acc': clean_acc, 'asr': asr, 'target': target_idx}


# ── Execution Pipeline ────────────────────────────────────────────────────
def run_experiments(ds_name, cfg, device, epochs=None):
    print(f'\n{"=" * 70}')
    print(f'  DATASET: {cfg["name"]}')
    print(f'{"=" * 70}')

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
    label = 'Mamba-Std-15%'
    
    print(f'\n[3/5] Training {label}...')
    model, acc_mat = train_mamba_std_15(task_data, task_names, cfg, device, epochs=epochs, label=label)
    
    avg_acc = float(np.nanmean(acc_mat[-1, :]))
    bwt_vals = [acc_mat[-1, i] - acc_mat[i, i] for i in range(len(task_names) - 1) if not np.isnan(acc_mat[-1, i])]
    bwt = float(np.mean(bwt_vals)) if bwt_vals else 0.0
    results[label] = {'model': model, 'acc_matrix': acc_mat, 'avg_acc': avg_acc, 'bwt': bwt}

    print('\n[4/5] Running Privacy Attacks...')
    iia_res = run_iia(model, task_data, task_names, cfg['task_splits'], l2i, device)
    results[label]['iia'] = iia_res
    print(f'  IIA avg AUC={np.mean([r["auc"] for r in iia_res.values()]):.3f}')

    mia_res = run_mia(model, task_data, task_names, device)
    results[label]['mia'] = mia_res
    print(f'  MIA avg AUC={np.mean([r["auc"] for r in mia_res.values()]):.3f}')

    fsi_res = run_fsi(model, task_data, task_names, cfg, device)
    results[label]['fsi'] = fsi_res
    print(f'  FSI avg k-NN acc={np.mean([r["acc"] for r in fsi_res.values()]):.3f}')

    print('\n[5/5] Running Backdoor Attack...')
    bd = run_backdoor(task_data, task_names, cfg, device, epochs=epochs, label=label)
    results[label]['backdoor'] = bd

    print(f'\n{"─" * 55}')
    print(f'  Summary — {cfg["name"]}')
    print(f'  {"Strategy":<26} {"Avg Acc":>9} {"BWT":>8}')
    print(f'  {"─" * 45}')
    r = results[label]
    print(f'  {label:<26} {r["avg_acc"]*100:>8.2f}% {r["bwt"]*100:>7.2f}%')

    results['_meta'] = {'task_data': task_data, 'task_names': task_names, 'l2i': l2i, 'cfg': cfg}
    return results

def plot_results(ds_results, ds_name):
    cfg        = ds_results['_meta']['cfg']
    task_names = ds_results['_meta']['task_names']
    n_tasks    = len(task_names)
    label      = 'Mamba-Std-15%'
    
    # 1. Accuracy Matrix Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    mat = ds_results[label]['acc_matrix']
    masked = np.ma.masked_invalid(mat * 100)
    im = ax.imshow(masked, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(n_tasks))
    ax.set_xticklabels([f'T{i+1}' for i in range(n_tasks)])
    ax.set_yticks(range(n_tasks))
    ax.set_yticklabels([f'After S{i+1}' for i in range(n_tasks)])
    ax.set_title(f'{cfg["name"]} — {label} Accuracy')
    for i in range(n_tasks):
        for j in range(n_tasks):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f'{v*100:.1f}', ha='center', va='center',
                        color='black' if v > 0.3 else 'white')
    plt.tight_layout()
    plt.savefig(f'fig_{ds_name}_{label}_matrix.png')
    plt.show()

    # 2. Attacks Plot
    task_x = [f'T{i+1}' for i in range(len(task_names))]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'{cfg["name"]} — {label} Privacy Attacks', fontsize=12, fontweight='bold')
    
    axes[0].plot(task_x, [ds_results[label]['iia'][t]['auc'] for t in task_names], marker='o', label='IIA AUC')
    axes[0].plot(task_x, [ds_results[label]['mia'][t]['auc'] for t in task_names], marker='s', label='MIA AUC')
    axes[0].axhline(0.5, ls='--', color='gray', label='Random (0.5)')
    axes[0].set_title('Membership Inference (AUC)')
    axes[0].set_ylim(0.4, 1.05)
    axes[0].legend()

    axes[1].plot(task_x, [ds_results[label]['fsi'][t]['acc'] for t in task_names], marker='^', color='orange', label='FSI k-NN')
    ch = ds_results[label]['fsi'][task_names[0]]['chance']
    axes[1].axhline(ch, ls=':', color='gray', label=f'Chance ({ch:.2f})')
    axes[1].set_title('Feature Space Inference (k-NN Acc)')
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    
    cleans = ds_results[label]['backdoor']['clean_acc'] * 100
    asrs   = ds_results[label]['backdoor']['asr'] * 100
    bars = axes[2].bar(['Clean Acc', 'ASR'], [cleans, asrs], color=['#3498DB', '#E74C3C'])
    for bar in bars:
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{bar.get_height():.1f}%', ha='center')
    axes[2].set_title('Backdoor Vulnerability')
    axes[2].set_ylim(0, 115)
    
    plt.tight_layout()
    plt.savefig(f'fig_{ds_name}_{label}_attacks.png')
    plt.show()

# ── Main Run ──────────────────────────────────────────────────────────────
ACTIVE_DATASETS = ['Dataset_1', 'UCI_HAR', 'WISDM']
ALL_RESULTS = {}

for ds_name in ACTIVE_DATASETS:
    cfg = DATASET_CONFIGS[ds_name]
    res = run_experiments(ds_name, cfg, DEVICE, epochs=EPOCHS)
    if res is not None:
        ALL_RESULTS[ds_name] = res
        plot_results(res, ds_name)

print('\n' + '=' * 75)
print('  CROSS-DATASET SUMMARY — FB-Mamba Standard 15% Replay')
print('=' * 75)
print(f'  {"Dataset":<22} {"Strategy":<26} {"Avg Acc":>9} {"BWT":>8} '
      f'{"IIA AUC":>9} {"FSI Acc":>9} {"BD ASR":>8}')
print(f'  {"─" * 72}')

for ds_name, ds_res in ALL_RESULTS.items():
    if ds_res is None: continue
    task_names = ds_res['_meta']['task_names']
    ds_label   = ds_res['_meta']['cfg']['name']
    label      = 'Mamba-Std-15%'
    r          = ds_res[label]
    
    avg_iia = np.mean([r['iia'][t]['auc'] for t in task_names])
    avg_fsi = np.mean([r['fsi'][t]['acc'] for t in task_names])
    bd_asr  = r['backdoor']['asr']
    
    print(f'  {ds_label:<22} {label:<26} {r["avg_acc"]*100:>8.2f}% '
          f'{r["bwt"]*100:>7.2f}%  {avg_iia:>8.3f}  {avg_fsi:>8.3f} '
          f'{bd_asr*100:>7.1f}%')
print(f'  {"─" * 72}')