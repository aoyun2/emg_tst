from __future__ import annotations

import math
import json
import copy
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from emg_tst.data import StandardScaler
from emg_tst.masking import stateful_variable_mask
from emg_tst.model import TSTEncoder, TSTPretrainDenoiser, TSTRegressor

# ==========================================
# Hard-coded training config (NO CLI FLAGS)
# ==========================================
SAMPLES_FILE = Path("samples_dataset.npy")   # produced by split_to_samples.py
SAVE_DIR = Path("checkpoints")
RUN_NAME = datetime.now().strftime("tst_%Y%m%d_%H%M%S")

SEED = 7
K_FOLDS = 5

# Model
D_MODEL = 128
N_HEADS = 8
D_FF = 256
N_LAYERS = 3
DROPOUT = 0.15

# Training
BATCH_SIZE = 64
LR = 3e-4
WEIGHT_DECAY = 1e-4
EPOCHS_PRETRAIN = 80
EPOCHS_FINETUNE = 100

# Masked reconstruction pretraining (stateful Markov masking)
MASK_R = 0.15
MASK_LM = 3

# Validation split (used to pick the best checkpoint; test is only evaluated once at the end).
# Prefer splitting by file_id to avoid temporal leakage between adjacent windows.
VAL_FRAC_FALLBACK = 0.1

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

def kfold_indices(n: int, k: int, seed: int):
    """Random k-fold split (fallback for single-file datasets)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    out = []
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i], axis=0)
        out.append((train_idx.astype(np.int64), test_idx.astype(np.int64)))
    return out


def lofo_indices(file_ids: np.ndarray):
    """Leave-One-File-Out splits. Each fold holds out one recording file.
    Prevents temporal leakage between consecutive windows of the same recording."""
    unique_files = np.unique(file_ids)
    out = []
    for held_out in unique_files:
        test_idx = np.where(file_ids == held_out)[0].astype(np.int64)
        train_idx = np.where(file_ids != held_out)[0].astype(np.int64)
        out.append((train_idx, test_idx))
    return out

class SamplesArrayDataset(torch.utils.data.Dataset):
    """In-memory samples dataset with optional normalization and subset indices.
    Returns (x, y_scalar, y_seq) where y_seq is the full angle trajectory."""
    def __init__(self, X_all: np.ndarray, y_all: np.ndarray, y_seq_all: np.ndarray,
                 indices: np.ndarray, mean: np.ndarray, std: np.ndarray,
                 feature_cols: np.ndarray | None = None, augment: bool = False):
        self.X_all = X_all
        self.y_all = y_all
        self.y_seq_all = y_seq_all
        self.indices = indices
        self.mean = mean
        self.std = std
        self.feature_cols = feature_cols  # None = all features
        self.augment = augment

    def __len__(self):
        return int(self.indices.shape[0])

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        x = self.X_all[idx].astype(np.float32)  # (W, F)
        x = (x - self.mean[None, :]) / self.std[None, :]
        if self.feature_cols is not None:
            x = x[:, self.feature_cols]
        if self.augment:
            # Gaussian noise: std=0.03 in normalised units (~3% of 1 std deviation).
            # Applied to all features. Amplitude jitter is intentionally excluded because
            # it would corrupt the quaternion features (last 4), which must remain unit-norm.
            x = x + np.random.randn(*x.shape).astype(np.float32) * 0.03
        # Normalize labels to [0, 1].  Knee angles are in [0°, 180°] by construction.
        # Without this, MSE gradients scale as ~137^2 ≈ 18 000 per sample; combined with
        # grad-norm clipping at 1.0, the output head bias can only shift ~3e-4 °/step and
        # never reaches the target mean in a reasonable number of epochs.
        LABEL_SCALE = 180.0
        y = np.float32(self.y_all[idx]) / LABEL_SCALE
        y_seq = self.y_seq_all[idx].astype(np.float32) / LABEL_SCALE  # (W,) in [0,1]
        return torch.from_numpy(x).float(), torch.tensor([y], dtype=torch.float32), torch.from_numpy(y_seq).float()


def _split_train_val(train_idx: np.ndarray, file_ids: np.ndarray, *, seed: int, val_frac: float = VAL_FRAC_FALLBACK):
    """Split a training index array into (train_idx2, val_idx).

    Primary mode: hold out an entire recording file (session) for validation.
    Fallback: random window split when there is only one training file.
    """
    train_idx = np.asarray(train_idx, dtype=np.int64).reshape(-1)
    if train_idx.size < 2:
        # Degenerate: force a minimal split.
        return train_idx[:1], train_idx[:0]

    rng = np.random.default_rng(int(seed))
    train_files = np.unique(file_ids[train_idx])

    if train_files.size >= 2:
        val_file = int(rng.choice(train_files))
        val_mask = file_ids[train_idx] == val_file
        val_idx = train_idx[val_mask]
        train_idx2 = train_idx[~val_mask]
    else:
        perm = rng.permutation(train_idx)
        n_val = int(max(1, round(float(val_frac) * perm.size)))
        val_idx = perm[:n_val]
        train_idx2 = perm[n_val:]

    if train_idx2.size < 1:
        # Ensure non-empty train split.
        perm = rng.permutation(train_idx)
        val_idx = perm[:1]
        train_idx2 = perm[1:]

    return train_idx2.astype(np.int64), val_idx.astype(np.int64)


def _file_names_for_ids(file_ids_sel: np.ndarray, file_names: np.ndarray) -> list[str]:
    out: list[str] = []
    uniq = np.unique(np.asarray(file_ids_sel, dtype=np.int64).reshape(-1))
    for fid in uniq.tolist():
        fid = int(fid)
        if 0 <= fid < int(len(file_names)):
            out.append(str(file_names[fid]))
        else:
            out.append(f"file_{fid}")
    return out


def _make_pretrain_dataset(ds_train: torch.utils.data.Dataset):
    class _Pre(torch.utils.data.Dataset):
        def __init__(self, base): self.base = base
        def __len__(self): return len(self.base)
        def __getitem__(self, i):
            x, _, _ = self.base[i]
            return x
    return _Pre(ds_train)


def train_one_fold(
    *,
    X_all: np.ndarray,
    y_all: np.ndarray,
    y_seq_all: np.ndarray,
    file_ids: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray | None = None,
    test_idx: np.ndarray,
    feature_cols: np.ndarray,
    device: torch.device,
    run_dir: Path | None,
    fold_i: int,
    label: str,
    data_meta: dict,
    quiet: bool = False,
    early_stop_patience: int = 0,
    epochs_pretrain: int = EPOCHS_PRETRAIN,
    epochs_finetune: int = EPOCHS_FINETUNE,
) -> dict:
    """Train one fold using an internal train/val split for model selection."""
    n_vars = int(len(feature_cols))
    seq_len = int(X_all.shape[1])

    # Train/val split for checkpoint selection (avoid peeking at test).
    if val_idx is None:
        train_idx2, val_idx = _split_train_val(train_idx, file_ids, seed=SEED + fold_i * 1000)
    else:
        train_idx2 = np.asarray(train_idx, dtype=np.int64).reshape(-1)
        val_idx = np.asarray(val_idx, dtype=np.int64).reshape(-1)

    # Per-recording EMG normalization.
    # Session-level amplitude offsets (electrode placement, skin impedance, fatigue)
    # are the main source of cross-session error.  Z-score each recording's EMG features
    # independently so the model sees comparable activation scales across sessions.
    # Quaternion features (last 4 columns) are unit-norm by construction — skip them.
    n_quat = 4
    n_raw_feat = int(X_all.shape[2]) - n_quat  # number of EMG/spectral features

    # Compute per-file stats using training windows only.
    file_mu: dict[int, np.ndarray] = {}
    file_sd: dict[int, np.ndarray] = {}
    for fid in np.unique(file_ids[train_idx2]).tolist():
        fidx = train_idx2[file_ids[train_idx2] == fid]
        x_ref = X_all[fidx, :, :n_raw_feat].reshape(-1, n_raw_feat)
        file_mu[int(fid)] = x_ref.mean(axis=0).astype(np.float32)
        file_sd[int(fid)] = np.maximum(x_ref.std(axis=0), 1e-6).astype(np.float32)

    # For held-out (test/val) files: use the file's own stats.  In a prosthetics
    # deployment you always run a short calibration pass before live use, so this
    # is a realistic assumption, not data leakage.
    for fid in np.unique(file_ids).tolist():
        if int(fid) not in file_mu:
            fidx = np.where(file_ids == fid)[0]
            x_ref = X_all[fidx, :, :n_raw_feat].reshape(-1, n_raw_feat)
            file_mu[int(fid)] = x_ref.mean(axis=0).astype(np.float32)
            file_sd[int(fid)] = np.maximum(x_ref.std(axis=0), 1e-6).astype(np.float32)

    # Build per-file-normalised copy of X_all (EMG channels only; quat unchanged).
    X_norm = X_all.copy()
    for i in range(int(X_all.shape[0])):
        fid = int(file_ids[i])
        X_norm[i, :, :n_raw_feat] = (
            (X_all[i, :, :n_raw_feat] - file_mu[fid]) / file_sd[fid]
        )

    # Fit global scaler on the per-file-normalised training windows.
    # EMG features are now ~N(0,1) per session; the global scaler handles residual
    # cross-session variance and normalises the quaternion features.
    scaler = StandardScaler.fit([X_norm[train_idx2].reshape(-1, X_all.shape[2])])

    ds_train = SamplesArrayDataset(X_norm, y_all, y_seq_all, train_idx2, scaler.mean_, scaler.std_, feature_cols, augment=True)
    ds_val   = SamplesArrayDataset(X_norm, y_all, y_seq_all, val_idx,    scaler.mean_, scaler.std_, feature_cols)
    ds_test  = SamplesArrayDataset(X_norm, y_all, y_seq_all, test_idx,   scaler.mean_, scaler.std_, feature_cols)

    # Pretraining uses clean (non-augmented) inputs — the masked reconstruction task
    # is already a strong regulariser; adding noise on top makes it unnecessarily hard.
    ds_pre_base = SamplesArrayDataset(X_norm, y_all, y_seq_all, train_idx2, scaler.mean_, scaler.std_, feature_cols, augment=False)
    ds_pre = _make_pretrain_dataset(ds_pre_base)

    bs = min(BATCH_SIZE, len(ds_train))
    dl_pre = DataLoader(ds_pre, batch_size=bs, shuffle=True, drop_last=len(ds_pre) > bs)
    dl_train = DataLoader(ds_train, batch_size=bs, shuffle=True, drop_last=len(ds_train) > bs)
    dl_val = DataLoader(ds_val, batch_size=min(BATCH_SIZE, len(ds_val)), shuffle=False)
    dl_test = DataLoader(ds_test, batch_size=min(BATCH_SIZE, len(ds_test)), shuffle=False)

    if not quiet:
        print(f"window={seq_len} vars={n_vars} train={len(ds_train)} val={len(ds_val)} test={len(ds_test)}")

    encoder = TSTEncoder(
        n_vars=n_vars, seq_len=seq_len,
        d_model=D_MODEL, n_heads=N_HEADS,
        d_ff=D_FF, n_layers=N_LAYERS,
        dropout=DROPOUT
    ).to(device)

    # -------- PRETRAIN --------
    pre = TSTPretrainDenoiser(encoder).to(device)
    opt = torch.optim.RAdam(pre.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    pre_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=int(epochs_pretrain), eta_min=LR * 0.01
    )

    if not quiet:
        print("[Pretrain]")
    best_recon_mse = float("inf")
    best_pre_state = None
    epoch_iter = tqdm(range(1, int(epochs_pretrain) + 1), desc="Pretrain", unit="epoch", disable=quiet)
    for epoch in epoch_iter:
        pre.train()
        num_sum = 0.0
        den_sum = 0.0
        for xb in dl_pre:
            xb = xb.to(device)
            # Generate mask on CPU — DirectML boolean advanced indexing
            # (state[bool_mask] = val) is unreliable and corrupts the mask.
            mask = stateful_variable_mask(
                batch_size=xb.shape[0],
                seq_len=xb.shape[1],
                n_vars=xb.shape[2],
                r=MASK_R,
                lm=MASK_LM,
                device="cpu",
            ).to(device)
            xb_masked = xb * mask
            pred = pre(xb_masked)

            inv = 1.0 - mask
            se = (pred - xb) ** 2
            num = (se * inv).sum()
            den = inv.sum().clamp_min(1.0)
            loss = num / den

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pre.parameters(), 1.0)
            opt.step()

            num_sum += float(num.detach().item())
            den_sum += float(den.detach().item())

        pre_scheduler.step()
        recon_mse = num_sum / max(den_sum, 1e-8)
        if recon_mse < best_recon_mse:
            best_recon_mse = recon_mse
            best_pre_state = {k: v.cpu().clone() for k, v in pre.state_dict().items()}
        epoch_iter.set_postfix(recon_mse=f"{recon_mse:.6f}", best=f"{best_recon_mse:.6f}")

    # Restore best pretrain weights before handing encoder to finetuning.
    if best_pre_state is not None:
        pre.load_state_dict(best_pre_state)
        pre.to(device)

    # -------- FINETUNE --------
    reg = TSTRegressor(encoder, out_dim=1).to(device)

    # Initialize the output head bias to the training-set label mean (normalized).
    # Default PyTorch init predicts ~0°; training labels are ~100–180°, giving
    # RMSE ≈ 140° at epoch 1.  Starting from the mean drops this to std(y) ≈ 30–40°
    # immediately and dramatically speeds up convergence.
    train_label_mean_norm = float(np.mean(y_all[train_idx2]) / 180.0)
    with torch.no_grad():
        reg.head.bias.fill_(train_label_mean_norm)

    # Separate LR groups: higher LR for the randomly-initialised head, lower for
    # the pretrained encoder.  Gradient clipping at norm=1.0 across ~425K params
    # means the 129-param head gets almost no update budget at the shared LR.
    LR_HEAD = LR * 10.0
    opt2 = torch.optim.RAdam([
        {"params": reg.encoder.parameters(), "lr": LR},
        {"params": reg.head.parameters(),    "lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=int(epochs_finetune), eta_min=LR * 0.01)
    mse = torch.nn.MSELoss()

    if not quiet:
        print("[Finetune]")
    best_val_rmse = float("inf")
    best_val_seq_rmse = float("inf")
    best_epoch = 0
    best_state = None
    best_patience_left = int(early_stop_patience)
    last_val_metrics = {"rmse": float("inf"), "seq_rmse": float("inf")}

    ft_iter = tqdm(range(1, int(epochs_finetune) + 1), desc="Finetune", unit="epoch", disable=quiet)
    for epoch in ft_iter:
        reg.train()
        total = 0.0
        n = 0
        for batch in dl_train:
            xb, _yb_scalar, yb_seq = batch
            xb = xb.to(device)
            yb_seq = yb_seq.to(device)
            out = reg(xb)               # [B,T,1]
            pred_seq = out[:, :, 0]      # [B,T]
            loss = mse(pred_seq, yb_seq)

            opt2.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reg.parameters(), 1.0)
            opt2.step()

            total += float(loss.detach().item()) * yb_seq.numel()
            n += yb_seq.numel()

        scheduler.step()

        val_metrics = eval_regressor(reg, dl_val, device=device) if len(ds_val) > 0 else {"rmse": float("inf"), "seq_rmse": float("inf")}
        # train_rmse: labels are normalized to [0,1]; multiply by 180 to report degrees
        last_val_metrics = dict(val_metrics)
        train_rmse_deg = math.sqrt(total / max(n, 1)) * 180.0
        train_rmse = train_rmse_deg
        ft_iter.set_postfix(
            train=f"{train_rmse:.1f}",
            val=f"{val_metrics['rmse']:.1f}",
            vseq=f"{val_metrics['seq_rmse']:.1f}",
        )

        improved = float(val_metrics["rmse"]) < float(best_val_rmse) - 1e-8
        if improved:
            best_val_rmse = float(val_metrics["rmse"])
            best_val_seq_rmse = float(val_metrics["seq_rmse"])
            best_epoch = int(epoch)
            best_state = {k: v.cpu().clone() for k, v in reg.state_dict().items()}
            best_patience_left = int(early_stop_patience)
        elif early_stop_patience > 0:
            best_patience_left -= 1
            if best_patience_left <= 0:
                if not quiet:
                    print(f"Early stop at epoch {epoch} (no val RMSE improvement for {early_stop_patience} epochs)")
                break

    # Evaluate test once using the best-by-validation weights.
    last_state = {k: v.cpu().clone() for k, v in reg.state_dict().items()}
    if best_state is not None:
        reg.load_state_dict(best_state, strict=True)
    test_metrics = eval_regressor(reg, dl_test, device=device)

    out = {
        "fold": int(fold_i),
        "n_train": int(len(ds_train)),
        "n_val": int(len(ds_val)),
        "n_test": int(len(ds_test)),
        "epochs_pretrain": int(epochs_pretrain),
        "epochs_finetune": int(epochs_finetune),
        "best_val_rmse": float(best_val_rmse),
        "best_val_seq_rmse": float(best_val_seq_rmse),
        "best_epoch": int(best_epoch),
        "final_val_rmse": float(last_val_metrics.get("rmse", float("inf"))),
        "test_rmse": float(test_metrics["rmse"]),
        "test_seq_rmse": float(test_metrics["seq_rmse"]),
        "test_mae": float(test_metrics["mae"]),
        # Backward compatibility: older tooling expects metrics.json.best_rmse.
        # This is the test RMSE at the checkpoint selected by validation RMSE.
        "best_rmse": float(test_metrics["rmse"]),
    }

    file_names = np.asarray(data_meta.get("file_names", np.array([])))
    train_file_ids = np.unique(np.asarray(file_ids[train_idx2], dtype=np.int64).reshape(-1)).astype(np.int64)
    val_file_ids = np.unique(np.asarray(file_ids[val_idx], dtype=np.int64).reshape(-1)).astype(np.int64)
    test_file_ids = np.unique(np.asarray(file_ids[test_idx], dtype=np.int64).reshape(-1)).astype(np.int64)
    split_manifest = {
        "fold": int(fold_i),
        "split_strategy": str(data_meta.get("split_strategy", "unknown")),
        "seed": int(data_meta.get("seed", SEED)),
        "samples_file": str(data_meta.get("samples_file", SAMPLES_FILE)),
        "test_indices": np.asarray(test_idx, dtype=np.int64).reshape(-1).tolist(),
        "test_file_ids": test_file_ids.tolist(),
        "test_file_names": _file_names_for_ids(test_file_ids, file_names),
        "val_file_ids": val_file_ids.tolist(),
        "val_file_names": _file_names_for_ids(val_file_ids, file_names),
        "train_file_ids": train_file_ids.tolist(),
        "train_file_names": _file_names_for_ids(train_file_ids, file_names),
    }
    out["held_out_file_ids"] = list(split_manifest["test_file_ids"])
    out["held_out_file_names"] = list(split_manifest["test_file_names"])

    if run_dir is not None:
        fold_dir = run_dir / f"fold_{fold_i:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Save best checkpoint (by validation RMSE).
        save_checkpoint(
            fold_dir / "reg_best.pt",
            reg=reg, encoder=encoder, scaler=scaler,
            extra={
                "stage": "finetune_best_val",
                "fold": int(fold_i),
                "epoch": int(best_epoch),
                "best_val_rmse": float(best_val_rmse),
                "best_val_seq_rmse": float(best_val_seq_rmse),
                "test_rmse": float(test_metrics["rmse"]),
                "test_seq_rmse": float(test_metrics["seq_rmse"]),
                "label_shift": int(data_meta.get("label_shift", 0)),
                "feature_cols": feature_cols.tolist(),
                "split_manifest": split_manifest,
            },
        )

        # Save last checkpoint (for debugging).
        reg.load_state_dict(last_state, strict=True)
        save_checkpoint(
            fold_dir / "reg_last.pt",
            reg=reg, encoder=encoder, scaler=scaler,
            extra={
                "stage": "finetune_last",
                "fold": int(fold_i),
                "epoch": int(epoch),
                "best_val_rmse": float(best_val_rmse),
                "label_shift": int(data_meta.get("label_shift", 0)),
                "feature_cols": feature_cols.tolist(),
                "split_manifest": split_manifest,
            },
        )

        with open(fold_dir / "split_manifest.json", "w", encoding="utf-8") as f:
            json.dump(split_manifest, f, indent=2)

        with open(fold_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

    return out


@torch.no_grad()
def eval_regressor(model: TSTRegressor, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    se_sum_last = 0.0
    ae_sum_last = 0.0
    n_last = 0
    se_sum_seq = 0.0
    ae_sum_seq = 0.0
    n_seq = 0
    for batch in loader:
        xb, yb_scalar, yb_seq = batch
        xb = xb.to(device)
        yb_scalar = yb_scalar.to(device)
        yb_seq = yb_seq.to(device)
        out = model(xb)            # [B,T,1]
        LABEL_SCALE = 180.0
        # Last timestep — convert from normalized [0,1] back to degrees
        pred_last = out[:, -1, 0] * LABEL_SCALE
        y_last = yb_scalar[:, 0] * LABEL_SCALE
        diff_last = pred_last - y_last
        se_sum_last += float((diff_last * diff_last).sum().item())
        ae_sum_last += float(diff_last.abs().sum().item())
        n_last += y_last.numel()
        # Full sequence — convert from normalized [0,1] back to degrees
        pred_seq = out[:, :, 0] * LABEL_SCALE
        yb_seq_deg = yb_seq * LABEL_SCALE
        diff_seq = pred_seq - yb_seq_deg
        se_sum_seq += float((diff_seq * diff_seq).sum().item())
        ae_sum_seq += float(diff_seq.abs().sum().item())
        n_seq += yb_seq_deg.numel()
    return {
        "rmse": math.sqrt(se_sum_last / max(n_last, 1)),
        "mae": ae_sum_last / max(n_last, 1),
        "seq_rmse": math.sqrt(se_sum_seq / max(n_seq, 1)),
        "seq_mae": ae_sum_seq / max(n_seq, 1),
    }

def save_checkpoint(path: Path, *, reg: TSTRegressor, encoder: TSTEncoder, scaler: StandardScaler, extra: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Move state dict to CPU before saving — DirectML tensors can't be serialized by torch.save.
    cpu_state = {k: v.cpu() for k, v in reg.state_dict().items()}
    ckpt = {
        "reg_state_dict": cpu_state,
        "model_cfg": {
            "n_vars": int(encoder.n_vars),
            "seq_len": int(encoder.seq_len),
            "d_model": int(D_MODEL),
            "n_heads": int(N_HEADS),
            "d_ff": int(D_FF),
            "n_layers": int(N_LAYERS),
            "dropout": float(DROPOUT),
        },
        "task_cfg": {
            "predict_last_only": False,
            "label_shift": int(extra.get("label_shift", 0)),
            # Model outputs are in [0,1]; multiply by label_scale to get degrees.
            "label_scale": 180.0,
        },
        "scaler": {
            "mean": scaler.mean_,
            "std": scaler.std_,
        },
        "extra": extra,
    }
    torch.save(ckpt, path)

def _train_cv(X_all, y_all, y_seq_all, file_ids, folds, feature_cols, device, run_dir, label, data, quiet=False):
    """Train + CV with a given feature subset. Returns list of fold metrics."""
    n_vars = len(feature_cols)
    seq_len = X_all.shape[1]
    n_folds = len(folds)
    all_fold_metrics = []
    cv_manifest = {
        "label": str(label),
        "split_strategy": str(data.get("split_strategy", "unknown")),
        "seed": int(data.get("seed", SEED)),
        "samples_file": str(data.get("samples_file", SAMPLES_FILE)),
        "n_folds": int(n_folds),
        "folds": [],
    }

    for fold_i, (train_idx, test_idx) in enumerate(folds, start=1):
        if not quiet:
            print(f"\n========== {label} | Fold {fold_i}/{n_folds} ==========")

        fold_metrics = train_one_fold(
            X_all=X_all,
            y_all=y_all,
            y_seq_all=y_seq_all,
            file_ids=file_ids,
            train_idx=train_idx,
            test_idx=test_idx,
            feature_cols=feature_cols,
            device=device,
            run_dir=run_dir,
            fold_i=fold_i,
            label=label,
            data_meta=data,
            quiet=quiet,
            early_stop_patience=0,
        )

        # Keep backward-compatible keys: "best_rmse" is the test RMSE at the best validation epoch.
        all_fold_metrics.append({
            "fold": int(fold_i),
            "best_rmse": float(fold_metrics["test_rmse"]),
            "best_val_rmse": float(fold_metrics["best_val_rmse"]),
            "final_rmse": float(fold_metrics["test_rmse"]),
            "final_seq_rmse": float(fold_metrics["test_seq_rmse"]),
            "final_mae": float(fold_metrics["test_mae"]),
            "held_out_file_ids": list(fold_metrics.get("held_out_file_ids", [])),
            "held_out_file_names": list(fold_metrics.get("held_out_file_names", [])),
        })
        cv_manifest["folds"].append({
            "fold": int(fold_i),
            "held_out_file_ids": list(fold_metrics.get("held_out_file_ids", [])),
            "held_out_file_names": list(fold_metrics.get("held_out_file_names", [])),
            "checkpoint": (str(run_dir / f"fold_{fold_i:02d}" / "reg_best.pt") if run_dir is not None else None),
            "split_manifest": (str(run_dir / f"fold_{fold_i:02d}" / "split_manifest.json") if run_dir is not None else None),
        })

    if run_dir is not None:
        with open(run_dir / "cv_manifest.json", "w", encoding="utf-8") as f:
            json.dump(cv_manifest, f, indent=2)
    return all_fold_metrics


@torch.no_grad()
def permutation_importance(reg, ds_test, feature_cols, device, n_repeats=5):
    """Compute permutation importance: how much RMSE increases when each feature is shuffled.
    Higher = more important."""
    reg.eval()
    n_vars = len(feature_cols)

    # Get baseline RMSE
    dl = DataLoader(ds_test, batch_size=min(BATCH_SIZE, len(ds_test)), shuffle=False)
    baseline = eval_regressor(reg, dl, device)["seq_rmse"]

    # For each feature, shuffle it and measure RMSE increase
    importance = np.zeros(n_vars, dtype=np.float64)

    for f_i in tqdm(range(n_vars), desc="Perm. importance", unit="feat"):
        deltas = []
        for _ in range(n_repeats):
            # Build shuffled dataset
            X_shuf = ds_test.X_all.copy()
            perm = np.random.permutation(X_shuf.shape[0])
            # Shuffle feature f_i across all samples at all timesteps
            orig_col = feature_cols[f_i]
            X_shuf[:, :, orig_col] = X_shuf[perm, :, orig_col]

            ds_shuf = SamplesArrayDataset(
                X_shuf, ds_test.y_all, ds_test.y_seq_all,
                ds_test.indices, ds_test.mean, ds_test.std,
                feature_cols
            )
            dl_shuf = DataLoader(ds_shuf, batch_size=min(BATCH_SIZE, len(ds_shuf)), shuffle=False)
            shuf_rmse = eval_regressor(reg, dl_shuf, device)["seq_rmse"]
            deltas.append(shuf_rmse - baseline)
        importance[f_i] = np.mean(deltas)

    return importance, baseline


def main():
    set_seed(SEED)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        try:
            import torch_directml
            device = torch_directml.device()
            print("Using DirectML (AMD GPU)")
        except ImportError:
            device = torch.device("cpu")
            print("DirectML not found, using CPU. Install with: pip install torch-directml")

    if not SAMPLES_FILE.exists():
        raise SystemExit(f"Missing {SAMPLES_FILE}. Run: python split_to_samples.py")

    data = np.load(SAMPLES_FILE, allow_pickle=True).item()
    X_all = data["X"].astype(np.float32)     # (N, W, F)
    y_all = data["y"].astype(np.float32)     # (N,)
    y_seq_all = data["y_seq"].astype(np.float32)  # (N, W)
    file_ids = data["file_id"]                # (N,) int32
    file_names = data.get("file_names", np.array([]))
    data["samples_file"] = str(SAMPLES_FILE)
    data["seed"] = int(SEED)

    n_all = int(X_all.shape[0])
    seq_len = int(X_all.shape[1])
    n_vars = int(X_all.shape[2])
    n_files = len(np.unique(file_ids))

    print(f"Dataset: {n_all} samples, window={seq_len}, features={n_vars}, files={n_files}")

    # Enforce quaternion thigh features (no scalar thigh_angle mode).
    thigh_mode = str(data.get("thigh_mode", ""))
    thigh_n = int(data.get("thigh_n_features", 0))
    if thigh_mode != "quat" or thigh_n != 4:
        raise SystemExit(
            "This pipeline requires thigh quaternion features (thigh_quat_wxyz, 4 dims). "
            f"Got thigh_mode={thigh_mode!r}, thigh_n_features={thigh_n}. "
            "Rebuild samples_dataset.npy from recordings produced by rigtest_gui.py."
        )

    # LOFO when multiple files, random k-fold fallback for single file
    if n_files >= 2:
        folds = lofo_indices(file_ids)
        data["split_strategy"] = "lofo"
        print(f"Cross-validation: Leave-One-File-Out ({n_files} folds)")
        for i, fid in enumerate(np.unique(file_ids)):
            name = file_names[fid] if fid < len(file_names) else f"file_{fid}"
            n_win = int((file_ids == fid).sum())
            print(f"  Fold {i+1}: hold out {name} ({n_win} windows)")
    else:
        folds = kfold_indices(n_all, K_FOLDS, seed=SEED)
        data["split_strategy"] = "kfold"
        print(f"Cross-validation: Random {K_FOLDS}-fold (single file, LOFO not possible)")

    # Feature column indices
    all_cols = np.arange(n_vars)
    thigh_col = np.arange(n_vars - thigh_n, n_vars)  # last thigh_n columns
    emg_cols = np.arange(n_vars - thigh_n)           # everything except thigh features

    # ==========================================
    # ABLATION STUDY: compare feature subsets
    # ==========================================
    print("\n" + "="*60)
    print("ABLATION STUDY: Does EMG help beyond thigh orientation alone?")
    print("="*60)

    results = {}

    # 1) All features (EMG + thigh)
    print("\n>>> [1/3] ALL FEATURES (EMG + thigh)")
    run_dir_all = SAVE_DIR / (RUN_NAME + "_all")
    run_dir_all.mkdir(parents=True, exist_ok=True)
    metrics_all = _train_cv(X_all, y_all, y_seq_all, file_ids, folds, all_cols, device, run_dir_all, "ALL", data)
    rmse_all = np.mean([m["best_rmse"] for m in metrics_all])
    results["all"] = rmse_all

    # 2) Thigh only
    print("\n>>> [2/3] THIGH ONLY (baseline)")
    run_dir_thigh = SAVE_DIR / (RUN_NAME + "_thigh_only")
    run_dir_thigh.mkdir(parents=True, exist_ok=True)
    metrics_thigh = _train_cv(X_all, y_all, y_seq_all, file_ids, folds, thigh_col, device, run_dir_thigh, "THIGH-ONLY", data)
    rmse_thigh = np.mean([m["best_rmse"] for m in metrics_thigh])
    results["thigh_only"] = rmse_thigh

    # 3) EMG only
    print("\n>>> [3/3] EMG ONLY (no thigh features)")
    run_dir_emg = SAVE_DIR / (RUN_NAME + "_emg_only")
    run_dir_emg.mkdir(parents=True, exist_ok=True)
    metrics_emg = _train_cv(X_all, y_all, y_seq_all, file_ids, folds, emg_cols, device, run_dir_emg, "EMG-ONLY", data)
    rmse_emg = np.mean([m["best_rmse"] for m in metrics_emg])
    results["emg_only"] = rmse_emg

    # ==========================================
    # ABLATION SUMMARY
    # ==========================================
    print("\n" + "="*60)
    print("ABLATION RESULTS (mean best RMSE across folds)")
    print("="*60)
    print(f"  All features (EMG + thigh): {rmse_all:8.3f} deg")
    print(f"  Thigh only:                 {rmse_thigh:8.3f} deg")
    print(f"  EMG only:                   {rmse_emg:8.3f} deg")
    print("-"*60)

    emg_improvement = rmse_thigh - rmse_all
    if emg_improvement > 0.5:
        print(f"  OK: EMG helps: adding EMG reduces RMSE by {emg_improvement:.2f} deg")
    elif emg_improvement > 0:
        print(f"  NOTE: EMG helps marginally: {emg_improvement:.2f} deg improvement")
    else:
        print(f"  WARNING: EMG not helping: thigh-only is {-emg_improvement:.2f} deg better")
        print(f"    Possible causes: not enough data, poor sensor contact, or EMG too noisy")

    thigh_contribution = rmse_emg - rmse_all
    print(f"  Thigh contribution:         {thigh_contribution:.2f} deg improvement over EMG-only")

    # ==========================================
    # PERMUTATION IMPORTANCE (on best full model, fold 1)
    # ==========================================
    print("\n" + "="*60)
    print("PERMUTATION IMPORTANCE (fold 1, all features)")
    print("  = RMSE increase when each feature is shuffled")
    print("="*60)

    set_seed(SEED)
    train_idx, test_idx = folds[0]

    # Load best model from fold 1
    ckpt_path = run_dir_all / "fold_01" / "reg_best.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt["model_cfg"]
        scaler_obj = ckpt.get("scaler", {})
        mean = np.asarray(scaler_obj.get("mean", None), dtype=np.float32)
        std = np.asarray(scaler_obj.get("std", None), dtype=np.float32)
        if mean.size == 0 or std.size == 0:
            # Fallback if older checkpoints exist.
            scaler = StandardScaler.fit([X_all[train_idx].reshape(-1, n_vars)])
            mean, std = scaler.mean_, scaler.std_

        feature_cols_pi = np.asarray(ckpt.get("extra", {}).get("feature_cols", all_cols.tolist()), dtype=np.int64)
        # Rebuild per-file-normalised X for fold 1 so PI uses the same preprocessing as training.
        n_quat_pi = 4
        n_raw_pi = int(X_all.shape[2]) - n_quat_pi
        file_mu_pi: dict[int, np.ndarray] = {}
        file_sd_pi: dict[int, np.ndarray] = {}
        for fid in np.unique(file_ids[train_idx]).tolist():
            fidx = train_idx[file_ids[train_idx] == fid]
            x_ref = X_all[fidx, :, :n_raw_pi].reshape(-1, n_raw_pi)
            file_mu_pi[int(fid)] = x_ref.mean(axis=0).astype(np.float32)
            file_sd_pi[int(fid)] = np.maximum(x_ref.std(axis=0), 1e-6).astype(np.float32)
        for fid in np.unique(file_ids).tolist():
            if int(fid) not in file_mu_pi:
                fidx = np.where(file_ids == fid)[0]
                x_ref = X_all[fidx, :, :n_raw_pi].reshape(-1, n_raw_pi)
                file_mu_pi[int(fid)] = x_ref.mean(axis=0).astype(np.float32)
                file_sd_pi[int(fid)] = np.maximum(x_ref.std(axis=0), 1e-6).astype(np.float32)
        X_norm_pi = X_all.copy()
        for i in range(int(X_all.shape[0])):
            fid = int(file_ids[i])
            X_norm_pi[i, :, :n_raw_pi] = (X_all[i, :, :n_raw_pi] - file_mu_pi[fid]) / file_sd_pi[fid]
        ds_test = SamplesArrayDataset(X_norm_pi, y_all, y_seq_all, test_idx, mean, std, feature_cols_pi)
        encoder = TSTEncoder(
            n_vars=int(cfg["n_vars"]), seq_len=int(cfg["seq_len"]),
            d_model=int(cfg["d_model"]), n_heads=int(cfg["n_heads"]),
            d_ff=int(cfg["d_ff"]), n_layers=int(cfg["n_layers"]),
            dropout=float(cfg["dropout"])
        ).to(device)
        reg = TSTRegressor(encoder, out_dim=1).to(device)
        reg.load_state_dict(ckpt["reg_state_dict"])

        importance, baseline = permutation_importance(reg, ds_test, feature_cols_pi, device, n_repeats=5)

        # Build feature names
        n_raw = int(data.get("n_raw_features", 0))
        has_raw = bool(data.get("has_raw_emg", False))
        feat_names = []
        if has_raw and n_raw > 0:
            from emg_tst.data import N_FFT_BANDS
            per_sensor = ["RMS", "MAV", "WL", "ZC", "SSC"] + [f"fft{b}" for b in range(N_FFT_BANDS)]
            for s in range(3):
                for fn in per_sensor:
                    feat_names.append(f"s{s+1}_{fn}")
        else:
            n_ch = int(data.get("n_channels", 16))
            for s in range(3):
                for c in range(n_ch):
                    feat_names.append(f"s{s+1}_spectr{c}")
        thigh_mode = str(data.get("thigh_mode", "angle"))
        thigh_n = int(data.get("thigh_n_features", 1))
        if thigh_mode == "quat" or thigh_n == 4:
            feat_names.extend(["thigh_q_w", "thigh_q_x", "thigh_q_y", "thigh_q_z"])
        else:
            feat_names.append("thigh_angle")

        # Sort by importance
        order = np.argsort(importance)[::-1]
        print(f"\n  Baseline seq_RMSE: {baseline:.3f} deg\n")
        print(f"  {'Feature':<20s} {'dRMSE (deg)':>10s}  {'Impact':>10s}")
        print(f"  {'-'*20} {'-'*10}  {'-'*10}")
        for rank, fi in enumerate(order):
            delta = importance[fi]
            # `importance` is indexed by the selected feature columns; map back to original feature id.
            col = int(feature_cols_pi[fi]) if fi < int(feature_cols_pi.size) else int(fi)
            name = feat_names[col] if col < len(feat_names) else f"col_{col}"
            bar = "#" * max(1, int(delta / max(importance.max(), 1e-6) * 20))
            print(f"  {name:<20s} {delta:>+10.3f}  {bar}")
            if rank >= 19:  # top 20
                break
    else:
        print("  (skipped - no checkpoint found)")

    # Save ablation summary
    summary = {
        "ablation": results,
        "emg_improvement_degrees": float(emg_improvement),
        "folds_all": metrics_all,
        "folds_thigh": metrics_thigh,
        "folds_emg": metrics_emg,
        "config": {
            "samples_file": str(SAMPLES_FILE),
            "seed": int(SEED),
            "model": {"d_model": D_MODEL, "n_heads": N_HEADS, "d_ff": D_FF, "n_layers": N_LAYERS, "dropout": DROPOUT},
            "train": {"batch_size": BATCH_SIZE, "lr": LR, "epochs_pretrain": EPOCHS_PRETRAIN, "epochs_finetune": EPOCHS_FINETUNE},
        }
    }
    summary_path = SAVE_DIR / RUN_NAME / "ablation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved ablation summary to: {summary_path}")

if __name__ == "__main__":
    main()