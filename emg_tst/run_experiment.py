from __future__ import annotations

import math
import json
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
DROPOUT = 0.1

# Training
BATCH_SIZE = 16       # reduced for 5-second (1000-step) windows to fit in GPU memory
LR = 3e-4
EPOCHS_PRETRAIN = 40
EPOCHS_FINETUNE = 20

# Masked reconstruction pretraining (stateful Markov masking)
MASK_R = 0.15
MASK_LM = 3

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
                 feature_cols: np.ndarray | None = None):
        self.X_all = X_all
        self.y_all = y_all
        self.y_seq_all = y_seq_all
        self.indices = indices
        self.mean = mean
        self.std = std
        self.feature_cols = feature_cols  # None = all features

    def __len__(self):
        return int(self.indices.shape[0])

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        x = self.X_all[idx].astype(np.float32)  # (W, F)
        x = (x - self.mean[None, :]) / self.std[None, :]
        if self.feature_cols is not None:
            x = x[:, self.feature_cols]
        y = np.float32(self.y_all[idx])
        y_seq = self.y_seq_all[idx].astype(np.float32)  # (W,)
        return torch.from_numpy(x).float(), torch.tensor([y], dtype=torch.float32), torch.from_numpy(y_seq).float()

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
        # Last timestep
        pred_last = out[:, -1, 0]
        y_last = yb_scalar[:, 0]
        diff_last = pred_last - y_last
        se_sum_last += float((diff_last * diff_last).sum().item())
        ae_sum_last += float(diff_last.abs().sum().item())
        n_last += y_last.numel()
        # Full sequence
        pred_seq = out[:, :, 0]    # [B,T]
        diff_seq = pred_seq - yb_seq
        se_sum_seq += float((diff_seq * diff_seq).sum().item())
        ae_sum_seq += float(diff_seq.abs().sum().item())
        n_seq += yb_seq.numel()
    return {
        "rmse": math.sqrt(se_sum_last / max(n_last, 1)),
        "mae": ae_sum_last / max(n_last, 1),
        "seq_rmse": math.sqrt(se_sum_seq / max(n_seq, 1)),
        "seq_mae": ae_sum_seq / max(n_seq, 1),
    }

def save_checkpoint(path: Path, *, reg: TSTRegressor, encoder: TSTEncoder, scaler: StandardScaler, extra: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "reg_state_dict": reg.state_dict(),
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
        },
        "scaler": {
            "mean": scaler.mean_,
            "std": scaler.std_,
        },
        "extra": extra,
    }
    torch.save(ckpt, path)

def _train_cv(X_all, y_all, y_seq_all, folds, feature_cols, device, run_dir, label, data, quiet=False):
    """Train + CV with a given feature subset. Returns list of fold metrics."""
    n_vars = len(feature_cols)
    seq_len = X_all.shape[1]
    n_folds = len(folds)
    all_fold_metrics = []

    for fold_i, (train_idx, test_idx) in enumerate(folds, start=1):
        if not quiet:
            print(f"\n========== {label} | Fold {fold_i}/{n_folds} ==========")

        # Fit scaler on ALL features, but dataset selects feature_cols
        scaler = StandardScaler.fit([X_all[train_idx].reshape(-1, X_all.shape[2])])

        ds_train = SamplesArrayDataset(X_all, y_all, y_seq_all, train_idx, scaler.mean_, scaler.std_, feature_cols)
        ds_test  = SamplesArrayDataset(X_all, y_all, y_seq_all, test_idx,  scaler.mean_, scaler.std_, feature_cols)

        class _Pre(torch.utils.data.Dataset):
            def __init__(self, base): self.base = base
            def __len__(self): return len(self.base)
            def __getitem__(self, i):
                x, _, _ = self.base[i]
                return x

        ds_pre = _Pre(ds_train)

        bs = min(BATCH_SIZE, len(ds_train))
        dl_pre = DataLoader(ds_pre, batch_size=bs, shuffle=True, drop_last=len(ds_pre) > bs)
        dl_train = DataLoader(ds_train, batch_size=bs, shuffle=True, drop_last=len(ds_train) > bs)
        dl_test = DataLoader(ds_test, batch_size=min(BATCH_SIZE, len(ds_test)), shuffle=False)

        if not quiet:
            print(f"window={seq_len} vars={n_vars} train={len(ds_train)} test={len(ds_test)}")

        # Fresh model per fold
        encoder = TSTEncoder(
            n_vars=n_vars, seq_len=seq_len,
            d_model=D_MODEL, n_heads=N_HEADS,
            d_ff=D_FF, n_layers=N_LAYERS,
            dropout=DROPOUT
        ).to(device)

        # -------- PRETRAIN --------
        pre = TSTPretrainDenoiser(encoder).to(device)
        opt = torch.optim.RAdam(pre.parameters(), lr=LR)

        if not quiet:
            print("[Pretrain]")
        epoch_iter = tqdm(range(1, EPOCHS_PRETRAIN + 1), desc="Pretrain", unit="epoch", disable=quiet)
        for epoch in epoch_iter:
            pre.train()
            num_sum = 0.0
            den_sum = 0.0
            for xb in dl_pre:
                xb = xb.to(device)
                mask = stateful_variable_mask(
                    batch_size=xb.shape[0],
                    seq_len=xb.shape[1],
                    n_vars=xb.shape[2],
                    r=MASK_R,
                    lm=MASK_LM,
                    device=device,
                )
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

            recon_mse = num_sum / max(den_sum, 1e-8)
            epoch_iter.set_postfix(recon_mse=f"{recon_mse:.6f}")

        # -------- FINETUNE --------
        reg = TSTRegressor(encoder, out_dim=1).to(device)
        opt2 = torch.optim.RAdam(reg.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=EPOCHS_FINETUNE, eta_min=LR * 0.01)
        mse = torch.nn.MSELoss()

        if not quiet:
            print("[Finetune]")
        best_rmse = float("inf")
        fold_dir = run_dir / f"fold_{fold_i:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        ft_iter = tqdm(range(1, EPOCHS_FINETUNE + 1), desc="Finetune", unit="epoch", disable=quiet)
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

            metrics = eval_regressor(reg, dl_test, device=device)
            train_rmse = math.sqrt(total / max(n, 1))
            ft_iter.set_postfix(
                train=f"{train_rmse:.3f}",
                test=f"{metrics['rmse']:.3f}",
                seq=f"{metrics['seq_rmse']:.3f}",
            )

            if metrics["rmse"] < best_rmse:
                best_rmse = metrics["rmse"]
                save_checkpoint(
                    fold_dir / "reg_best.pt",
                    reg=reg, encoder=encoder, scaler=scaler,
                    extra={"stage":"finetune_best", "fold": fold_i, "epoch": epoch, "test_rmse": float(metrics["rmse"]),
                           "label_shift": int(data.get("label_shift", 0)), "feature_cols": feature_cols.tolist()}
                )

        # Always save last
        save_checkpoint(
            fold_dir / "reg_last.pt",
            reg=reg, encoder=encoder, scaler=scaler,
            extra={"stage":"finetune_last", "fold": fold_i, "epoch": EPOCHS_FINETUNE, "best_rmse": float(best_rmse),
                   "label_shift": int(data.get("label_shift", 0)), "feature_cols": feature_cols.tolist()}
        )

        # Write fold metrics
        with open(fold_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump({
                "fold": fold_i,
                "best_rmse": float(best_rmse),
                "final_test_rmse": float(metrics["rmse"]),
                "final_test_seq_rmse": float(metrics["seq_rmse"]),
                "final_test_mae": float(metrics["mae"]),
            }, f, indent=2)

        all_fold_metrics.append({
            "fold": fold_i,
            "best_rmse": float(best_rmse),
            "final_rmse": float(metrics["rmse"]),
            "final_seq_rmse": float(metrics["seq_rmse"]),
            "final_mae": float(metrics["mae"]),
        })

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not SAMPLES_FILE.exists():
        raise SystemExit(f"Missing {SAMPLES_FILE}. Run: python split_to_samples.py")

    data = np.load(SAMPLES_FILE, allow_pickle=True).item()
    X_all = data["X"].astype(np.float32)     # (N, W, F)
    y_all = data["y"].astype(np.float32)     # (N,)
    y_seq_all = data["y_seq"].astype(np.float32)  # (N, W)
    file_ids = data["file_id"]                # (N,) int32
    file_names = data.get("file_names", np.array([]))

    n_all = int(X_all.shape[0])
    seq_len = int(X_all.shape[1])
    n_vars = int(X_all.shape[2])
    n_files = len(np.unique(file_ids))

    print(f"Dataset: {n_all} samples, window={seq_len}, features={n_vars}, files={n_files}")

    # LOFO when multiple files, random k-fold fallback for single file
    if n_files >= 2:
        folds = lofo_indices(file_ids)
        print(f"Cross-validation: Leave-One-File-Out ({n_files} folds)")
        for i, fid in enumerate(np.unique(file_ids)):
            name = file_names[fid] if fid < len(file_names) else f"file_{fid}"
            n_win = int((file_ids == fid).sum())
            print(f"  Fold {i+1}: hold out {name} ({n_win} windows)")
    else:
        folds = kfold_indices(n_all, K_FOLDS, seed=SEED)
        print(f"Cross-validation: Random {K_FOLDS}-fold (single file, LOFO not possible)")

    # Feature column indices
    all_cols = np.arange(n_vars)
    thigh_col = np.array([n_vars - 1])          # last column
    emg_cols = np.arange(n_vars - 1)            # everything except last

    # ==========================================
    # ABLATION STUDY: compare feature subsets
    # ==========================================
    print("\n" + "="*60)
    print("ABLATION STUDY: Does EMG help beyond thigh angle alone?")
    print("="*60)

    results = {}

    # 1) All features (EMG + thigh)
    print("\n>>> [1/3] ALL FEATURES (EMG + thigh)")
    run_dir_all = SAVE_DIR / (RUN_NAME + "_all")
    run_dir_all.mkdir(parents=True, exist_ok=True)
    metrics_all = _train_cv(X_all, y_all, y_seq_all, folds, all_cols, device, run_dir_all, "ALL", data)
    rmse_all = np.mean([m["best_rmse"] for m in metrics_all])
    results["all"] = rmse_all

    # 2) Thigh angle only
    print("\n>>> [2/3] THIGH ANGLE ONLY (baseline)")
    run_dir_thigh = SAVE_DIR / (RUN_NAME + "_thigh_only")
    run_dir_thigh.mkdir(parents=True, exist_ok=True)
    metrics_thigh = _train_cv(X_all, y_all, y_seq_all, folds, thigh_col, device, run_dir_thigh, "THIGH-ONLY", data)
    rmse_thigh = np.mean([m["best_rmse"] for m in metrics_thigh])
    results["thigh_only"] = rmse_thigh

    # 3) EMG only
    print("\n>>> [3/3] EMG ONLY (no thigh angle)")
    run_dir_emg = SAVE_DIR / (RUN_NAME + "_emg_only")
    run_dir_emg.mkdir(parents=True, exist_ok=True)
    metrics_emg = _train_cv(X_all, y_all, y_seq_all, folds, emg_cols, device, run_dir_emg, "EMG-ONLY", data)
    rmse_emg = np.mean([m["best_rmse"] for m in metrics_emg])
    results["emg_only"] = rmse_emg

    # ==========================================
    # ABLATION SUMMARY
    # ==========================================
    print("\n" + "="*60)
    print("ABLATION RESULTS (mean best RMSE across folds)")
    print("="*60)
    print(f"  All features (EMG + thigh): {rmse_all:8.3f}°")
    print(f"  Thigh angle only:           {rmse_thigh:8.3f}°")
    print(f"  EMG only:                   {rmse_emg:8.3f}°")
    print("-"*60)

    emg_improvement = rmse_thigh - rmse_all
    if emg_improvement > 0.5:
        print(f"  ✓ EMG HELPS: adding EMG reduces RMSE by {emg_improvement:.2f}°")
    elif emg_improvement > 0:
        print(f"  ~ EMG helps marginally: {emg_improvement:.2f}° improvement")
    else:
        print(f"  ✗ EMG NOT HELPING: thigh-only is {-emg_improvement:.2f}° better")
        print(f"    Possible causes: not enough data, poor sensor contact, or EMG too noisy")

    thigh_contribution = rmse_emg - rmse_all
    print(f"  Thigh angle contribution:   {thigh_contribution:.2f}° improvement over EMG-only")

    # ==========================================
    # PERMUTATION IMPORTANCE (on best full model, fold 1)
    # ==========================================
    print("\n" + "="*60)
    print("PERMUTATION IMPORTANCE (fold 1, all features)")
    print("  = RMSE increase when each feature is shuffled")
    print("="*60)

    set_seed(SEED)
    train_idx, test_idx = folds[0]
    scaler = StandardScaler.fit([X_all[train_idx].reshape(-1, n_vars)])
    ds_test = SamplesArrayDataset(X_all, y_all, y_seq_all, test_idx, scaler.mean_, scaler.std_, all_cols)

    # Load best model from fold 1
    ckpt_path = run_dir_all / "fold_01" / "reg_best.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt["model_cfg"]
        encoder = TSTEncoder(
            n_vars=int(cfg["n_vars"]), seq_len=int(cfg["seq_len"]),
            d_model=int(cfg["d_model"]), n_heads=int(cfg["n_heads"]),
            d_ff=int(cfg["d_ff"]), n_layers=int(cfg["n_layers"]),
            dropout=float(cfg["dropout"])
        ).to(device)
        reg = TSTRegressor(encoder, out_dim=1).to(device)
        reg.load_state_dict(ckpt["reg_state_dict"])

        importance, baseline = permutation_importance(reg, ds_test, all_cols, device, n_repeats=5)

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
        feat_names.append("thigh_angle")

        # Sort by importance
        order = np.argsort(importance)[::-1]
        print(f"\n  Baseline seq_RMSE: {baseline:.3f}°\n")
        print(f"  {'Feature':<20s} {'ΔRMSE (°)':>10s}  {'Impact':>10s}")
        print(f"  {'-'*20} {'-'*10}  {'-'*10}")
        for rank, fi in enumerate(order):
            delta = importance[fi]
            name = feat_names[fi] if fi < len(feat_names) else f"col_{fi}"
            bar = "█" * max(1, int(delta / max(importance.max(), 1e-6) * 20))
            print(f"  {name:<20s} {delta:>+10.3f}  {bar}")
            if rank >= 19:  # top 20
                break
    else:
        print("  (skipped — no checkpoint found)")

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
