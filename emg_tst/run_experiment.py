from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from emg_tst.data import load_recording
from emg_tst.model import SensorFusionTransformer, rolling_last_step_predict

# ==========================================
# Hard-coded training config (NO CLI FLAGS)
# ==========================================
DATA_GLOB = "data*.npy"
SAVE_DIR = Path("checkpoints")
RUN_NAME = datetime.now().strftime("tst_%Y%m%d_%H%M%S")

SEED = 7
K_FOLDS = 5

# Windowing / task
SOURCE_WINDOW = 200
CONTEXT_WINDOW = 100
TRAIN_STRIDE = 1
EVAL_STRIDE = 1
LABEL_SHIFT = 0
VAL_FRAC_FALLBACK = 0.1
LABEL_SCALE = 180.0

# Model
D_MODEL = 96
N_HEADS = 6
D_FF = 192
N_LAYERS = 3
DROPOUT = 0.20
CAUSAL_ATTENTION = False

# Training
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 20
EARLY_STOP_PATIENCE = 5
TRAIN_NOISE_STD = 0.02
TRAIN_DROP_IMU_PROB = 0.30
PRED_BATCH_SIZE = 512


def set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))


def kfold_indices(n: int, k: int, seed: int):
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(int(n))
    folds = np.array_split(perm, int(k))
    out = []
    for i in range(int(k)):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(int(k)) if j != i], axis=0)
        out.append((train_idx.astype(np.int64), test_idx.astype(np.int64)))
    return out


def lofo_indices(file_ids: np.ndarray):
    file_ids = np.asarray(file_ids, dtype=np.int64).reshape(-1)
    unique_files = np.unique(file_ids)
    out = []
    for held_out in unique_files.tolist():
        test_idx = np.where(file_ids == int(held_out))[0].astype(np.int64)
        train_idx = np.where(file_ids != int(held_out))[0].astype(np.int64)
        out.append((train_idx, test_idx))
    return out


@dataclass(frozen=True)
class RecordingCorpus:
    paths: tuple[Path, ...]
    file_names: tuple[str, ...]
    X_list: tuple[np.ndarray, ...]
    y_list: tuple[np.ndarray, ...]
    n_features: int
    n_emg: int
    n_imu: int
    sample_hz: float
    meta: dict


class LazyWindowDataset(Dataset):
    def __init__(
        self,
        *,
        X_list: tuple[np.ndarray, ...],
        y_list: tuple[np.ndarray, ...],
        windows: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        feature_cols: np.ndarray,
        context_window: int,
        n_emg_selected: int,
        n_imu_selected: int,
        augment: bool = False,
        noise_std: float = 0.0,
        drop_imu_prob: float = 0.0,
    ):
        self.X_list = X_list
        self.y_list = y_list
        self.windows = np.asarray(windows, dtype=np.int64).reshape(-1, 2)
        self.mean = np.asarray(mean, dtype=np.float32).reshape(-1)
        self.std = np.asarray(std, dtype=np.float32).reshape(-1)
        self.feature_cols = np.asarray(feature_cols, dtype=np.int64).reshape(-1)
        self.context_window = int(context_window)
        self.n_emg_selected = int(max(0, n_emg_selected))
        self.n_imu_selected = int(max(0, n_imu_selected))
        self.augment = bool(augment)
        self.noise_std = float(max(0.0, noise_std))
        self.drop_imu_prob = float(max(0.0, drop_imu_prob))

    def __len__(self) -> int:
        return int(self.windows.shape[0])

    def __getitem__(self, i: int):
        file_i, start = self.windows[int(i)].tolist()
        start_i = int(start)
        stop_i = start_i + int(SOURCE_WINDOW)
        x = self.X_list[int(file_i)][start_i:stop_i].astype(np.float32)
        y = float(self.y_list[int(file_i)][stop_i - 1]) / float(LABEL_SCALE)
        x = (x - self.mean[None, :]) / self.std[None, :]
        x = x[:, self.feature_cols]
        if self.context_window < int(x.shape[0]):
            x = x[-self.context_window :]
        if self.augment:
            if self.noise_std > 0.0:
                x = x + np.random.randn(*x.shape).astype(np.float32) * self.noise_std
            if self.n_emg_selected > 0 and self.n_imu_selected > 0 and float(np.random.rand()) < self.drop_imu_prob:
                x[:, self.n_emg_selected :] = 0.0
        return torch.from_numpy(x).float(), torch.tensor([y], dtype=torch.float32)


def _discover_recording_paths() -> list[Path]:
    paths = sorted(Path(".").glob(DATA_GLOB))
    out = []
    for p in paths:
        if p.name == "samples_dataset.npy" or "samples_" in p.name:
            continue
        out.append(p)
    return out


def _assert_recording_layout_compatible(first_meta: dict, meta: dict, path: Path) -> None:
    if int(meta.get("n_features", -1)) != int(first_meta.get("n_features", -2)):
        raise RuntimeError(
            f"Feature count mismatch: {path.name} has F={int(meta.get('n_features', -1))}, "
            f"expected F={int(first_meta.get('n_features', -1))}."
        )
    if int(meta.get("n_angular_velocity_features", 0)) != int(first_meta.get("n_angular_velocity_features", 0)):
        raise RuntimeError(
            f"Angular-velocity layout mismatch in {path.name}: "
            f"{int(meta.get('n_angular_velocity_features', 0))} vs {int(first_meta.get('n_angular_velocity_features', 0))}."
        )
    if str(meta.get("thigh_mode", "")) != str(first_meta.get("thigh_mode", "")) or int(meta.get("thigh_n_features", 0)) != int(first_meta.get("thigh_n_features", 0)):
        raise RuntimeError(
            f"Thigh feature mismatch in {path.name}: thigh_mode={meta.get('thigh_mode')!r}, "
            f"thigh_n_features={int(meta.get('thigh_n_features', 0))}."
        )


def load_recording_corpus() -> RecordingCorpus:
    paths = _discover_recording_paths()
    if not paths:
        raise SystemExit(f"No recordings found matching {DATA_GLOB}. Expected data0.npy, data1.npy, ...")

    X_raw_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    first_meta: dict | None = None
    for p in paths:
        X, y, meta = load_recording(p)
        if first_meta is None:
            first_meta = dict(meta)
        else:
            _assert_recording_layout_compatible(first_meta, meta, p)
        X_raw_list.append(np.asarray(X, dtype=np.float32))
        y_list.append(np.asarray(y, dtype=np.float32))

    assert first_meta is not None
    n_features = int(first_meta["n_features"])
    n_imu = int(first_meta.get("n_angular_velocity_features", 0)) + int(first_meta.get("thigh_n_features", 0))
    n_emg = int(n_features - n_imu)
    if n_emg < 0:
        raise RuntimeError(f"Invalid feature layout: n_features={n_features}, n_imu={n_imu}")

    X_norm_list: list[np.ndarray] = []
    for X in X_raw_list:
        Xi = np.asarray(X, dtype=np.float32).copy()
        if n_emg > 0:
            blk = Xi[:, :n_emg].astype(np.float64)
            mu = blk.mean(axis=0)
            sd = np.maximum(blk.std(axis=0), 1e-6)
            Xi[:, :n_emg] = ((blk - mu) / sd).astype(np.float32)
        X_norm_list.append(Xi.astype(np.float32))

    return RecordingCorpus(
        paths=tuple(paths),
        file_names=tuple(p.name for p in paths),
        X_list=tuple(X_norm_list),
        y_list=tuple(y_list),
        n_features=int(n_features),
        n_emg=int(n_emg),
        n_imu=int(n_imu),
        sample_hz=float(first_meta.get("effective_hz", 200.0)),
        meta=dict(first_meta),
    )


def build_windows_for_files(
    corpus: RecordingCorpus,
    file_ids: tuple[int, ...] | list[int] | np.ndarray,
    *,
    stride: int,
) -> np.ndarray:
    rows: list[tuple[int, int]] = []
    for fid in [int(x) for x in file_ids]:
        T = int(corpus.y_list[int(fid)].shape[0])
        if T < int(SOURCE_WINDOW):
            continue
        for start in range(0, T - int(SOURCE_WINDOW) + 1, int(stride)):
            rows.append((int(fid), int(start)))
    if not rows:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(rows, dtype=np.int64)


def _split_train_val_files(train_file_ids: tuple[int, ...], *, seed: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    files = tuple(sorted(int(x) for x in train_file_ids))
    if len(files) >= 2:
        rng = np.random.default_rng(int(seed))
        val_file = int(rng.choice(np.asarray(files, dtype=np.int64)))
        train_files = tuple(fid for fid in files if fid != val_file)
        return train_files, (val_file,)
    return files, ()


def build_fold_windows(
    corpus: RecordingCorpus,
    *,
    train_file_ids: tuple[int, ...],
    test_file_ids: tuple[int, ...],
    seed: int,
    train_stride: int = TRAIN_STRIDE,
    eval_stride: int = EVAL_STRIDE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...], tuple[int, ...]]:
    train_files, val_files = _split_train_val_files(tuple(train_file_ids), seed=int(seed))
    train_windows = build_windows_for_files(corpus, train_files, stride=int(train_stride))
    test_windows = build_windows_for_files(corpus, test_file_ids, stride=int(eval_stride))
    val_windows = build_windows_for_files(corpus, val_files, stride=int(eval_stride))

    if val_windows.size < 1 and train_windows.shape[0] >= 2:
        rng = np.random.default_rng(int(seed))
        perm = rng.permutation(train_windows.shape[0])
        n_val = int(max(1, round(float(VAL_FRAC_FALLBACK) * float(train_windows.shape[0]))))
        val_sel = np.sort(perm[:n_val])
        train_sel = np.sort(perm[n_val:])
        val_windows = train_windows[val_sel].astype(np.int64)
        train_windows = train_windows[train_sel].astype(np.int64)

    return train_windows, val_windows, test_windows, train_files, val_files


def _fit_global_scaler_from_files(corpus: RecordingCorpus, file_ids: tuple[int, ...], *, n_features: int) -> tuple[np.ndarray, np.ndarray]:
    parts = [corpus.X_list[int(fid)] for fid in file_ids]
    if not parts:
        raise RuntimeError("Cannot fit scaler with no training files.")
    flat = np.concatenate(parts, axis=0).astype(np.float32)
    mean = flat.mean(axis=0).astype(np.float32)
    std = np.maximum(flat.std(axis=0), 1e-6).astype(np.float32)
    if int(mean.size) != int(n_features) or int(std.size) != int(n_features):
        raise RuntimeError("Scaler feature count mismatch.")
    return mean, std


def _training_label_mean_norm(corpus: RecordingCorpus, train_windows: np.ndarray) -> float:
    vals = []
    for fid, start in np.asarray(train_windows, dtype=np.int64).reshape(-1, 2).tolist():
        stop = int(start) + int(SOURCE_WINDOW)
        vals.append(float(corpus.y_list[int(fid)][stop - 1]) / float(LABEL_SCALE))
    if not vals:
        return 0.5
    return float(np.mean(np.asarray(vals, dtype=np.float64)))


@torch.no_grad()
def eval_last_step(model: SensorFusionTransformer, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    se = 0.0
    ae = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        diff = (pred - yb[:, 0]) * float(LABEL_SCALE)
        se += float((diff * diff).sum().item())
        ae += float(diff.abs().sum().item())
        n += int(diff.numel())
    return {
        "rmse": math.sqrt(se / max(n, 1)),
        "mae": ae / max(n, 1),
    }


@torch.no_grad()
def eval_recording_rollout(
    model: SensorFusionTransformer,
    *,
    corpus: RecordingCorpus,
    file_ids: tuple[int, ...],
    mean: np.ndarray,
    std: np.ndarray,
    feature_cols: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    se = 0.0
    ae = 0.0
    n = 0
    for fid in [int(x) for x in file_ids]:
        x = corpus.X_list[int(fid)].astype(np.float32)
        xn = (x - mean[None, :]) / std[None, :]
        xs = xn[:, feature_cols]
        pred = rolling_last_step_predict(model, xs, batch_size=int(PRED_BATCH_SIZE)).numpy().astype(np.float32)
        gt = corpus.y_list[int(fid)].astype(np.float32)
        m = int(min(pred.size, gt.size))
        if m < 1:
            continue
        diff = (pred[:m] * float(LABEL_SCALE)) - gt[:m]
        se += float(np.sum(diff * diff))
        ae += float(np.sum(np.abs(diff)))
        n += int(m)
    return {
        "rmse": math.sqrt(se / max(n, 1)),
        "mae": ae / max(n, 1),
    }


def _save_checkpoint(
    path: Path,
    *,
    model: SensorFusionTransformer,
    mean: np.ndarray,
    std: np.ndarray,
    feature_cols: np.ndarray,
    n_emg_norm: int,
    extra: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    ckpt = {
        "model_state_dict": cpu_state,
        "reg_state_dict": cpu_state,
        "model_cfg": {
            "model_type": "sensor_fusion_last_step_transformer",
            "n_vars": int(feature_cols.size),
            "n_emg_vars": int(model.n_emg_vars),
            "n_imu_vars": int(model.n_imu_vars),
            "seq_len": int(model.seq_len),
            "d_model": int(model.d_model),
            "n_heads": int(model.n_heads),
            "d_ff": int(model.d_ff),
            "n_layers": int(model.n_layers),
            "dropout": float(model.dropout),
            "causal": bool(model.causal),
        },
        "task_cfg": {
            "predict_last_only": True,
            "label_shift": int(LABEL_SHIFT),
            "label_scale": float(LABEL_SCALE),
            "source_window": int(SOURCE_WINDOW),
            "context_window": int(CONTEXT_WINDOW),
            "train_stride": int(extra.get("train_stride", TRAIN_STRIDE)),
        },
        "scaler": {
            "mean": np.asarray(mean, dtype=np.float32),
            "std": np.asarray(std, dtype=np.float32),
        },
        "extra": dict(extra) | {
            "feature_cols": np.asarray(feature_cols, dtype=np.int64).tolist(),
            "n_emg_norm": int(n_emg_norm),
        },
    }
    torch.save(ckpt, path)


def train_one_fold(
    *,
    corpus: RecordingCorpus,
    train_windows: np.ndarray,
    val_windows: np.ndarray | None,
    test_windows: np.ndarray,
    train_file_ids: tuple[int, ...],
    val_file_ids: tuple[int, ...],
    test_file_ids: tuple[int, ...],
    feature_cols: np.ndarray,
    n_emg_selected: int,
    n_imu_selected: int,
    device: torch.device,
    run_dir: Path | None,
    fold_i: int,
    label: str,
    quiet: bool = False,
    epochs: int = EPOCHS,
    early_stop_patience: int = EARLY_STOP_PATIENCE,
    train_stride: int = TRAIN_STRIDE,
) -> dict:
    feature_cols = np.asarray(feature_cols, dtype=np.int64).reshape(-1)
    train_windows = np.asarray(train_windows, dtype=np.int64).reshape(-1, 2)
    test_windows = np.asarray(test_windows, dtype=np.int64).reshape(-1, 2)
    val_windows_arr = np.empty((0, 2), dtype=np.int64) if val_windows is None else np.asarray(val_windows, dtype=np.int64).reshape(-1, 2)

    train_files_for_scaler = tuple(train_file_ids) if train_file_ids else tuple(sorted(set(train_windows[:, 0].tolist())))
    mean, std = _fit_global_scaler_from_files(corpus, train_files_for_scaler, n_features=corpus.n_features)

    ds_train = LazyWindowDataset(
        X_list=corpus.X_list,
        y_list=corpus.y_list,
        windows=train_windows,
        mean=mean,
        std=std,
        feature_cols=feature_cols,
        context_window=int(CONTEXT_WINDOW),
        n_emg_selected=int(n_emg_selected),
        n_imu_selected=int(n_imu_selected),
        augment=True,
        noise_std=float(TRAIN_NOISE_STD),
        drop_imu_prob=(float(TRAIN_DROP_IMU_PROB) if (n_emg_selected > 0 and n_imu_selected > 0) else 0.0),
    )
    ds_val = LazyWindowDataset(
        X_list=corpus.X_list,
        y_list=corpus.y_list,
        windows=val_windows_arr,
        mean=mean,
        std=std,
        feature_cols=feature_cols,
        context_window=int(CONTEXT_WINDOW),
        n_emg_selected=int(n_emg_selected),
        n_imu_selected=int(n_imu_selected),
    )
    ds_test = LazyWindowDataset(
        X_list=corpus.X_list,
        y_list=corpus.y_list,
        windows=test_windows,
        mean=mean,
        std=std,
        feature_cols=feature_cols,
        context_window=int(CONTEXT_WINDOW),
        n_emg_selected=int(n_emg_selected),
        n_imu_selected=int(n_imu_selected),
    )

    if len(ds_train) < 1:
        raise RuntimeError("Training fold has no windows.")
    if len(ds_test) < 1:
        raise RuntimeError("Test fold has no windows.")

    dl_train = DataLoader(ds_train, batch_size=min(int(BATCH_SIZE), len(ds_train)), shuffle=True, drop_last=len(ds_train) > int(BATCH_SIZE))
    dl_val = DataLoader(ds_val, batch_size=min(int(PRED_BATCH_SIZE), max(len(ds_val), 1)), shuffle=False) if len(ds_val) > 0 else None
    dl_test = DataLoader(ds_test, batch_size=min(int(PRED_BATCH_SIZE), len(ds_test)), shuffle=False)

    model = SensorFusionTransformer(
        n_emg_vars=int(n_emg_selected),
        n_imu_vars=int(n_imu_selected),
        seq_len=int(CONTEXT_WINDOW),
        d_model=int(D_MODEL),
        n_heads=int(N_HEADS),
        d_ff=int(D_FF),
        n_layers=int(N_LAYERS),
        dropout=float(DROPOUT),
        causal=bool(CAUSAL_ATTENTION),
    ).to(device)

    with torch.no_grad():
        model.head.bias.fill_(_training_label_mean_norm(corpus, train_windows))

    opt = torch.optim.Adam(model.parameters(), lr=float(LR), weight_decay=float(WEIGHT_DECAY))
    mse = nn.MSELoss()
    best_val_rmse = float("inf")
    best_epoch = 0
    best_state = None
    patience_left = int(early_stop_patience)
    last_val_rmse = float("inf")

    split_manifest = {
        "fold": int(fold_i),
        "split_strategy": "lofo" if len(corpus.file_names) >= 2 else "kfold",
        "seed": int(SEED),
        "recording_glob": str(DATA_GLOB),
        "train_file_ids": [int(x) for x in train_file_ids],
        "train_file_names": [str(corpus.file_names[int(x)]) for x in train_file_ids],
        "val_file_ids": [int(x) for x in val_file_ids],
        "val_file_names": [str(corpus.file_names[int(x)]) for x in val_file_ids],
        "test_file_ids": [int(x) for x in test_file_ids],
        "test_file_names": [str(corpus.file_names[int(x)]) for x in test_file_ids],
        "test_indices": [],
    }

    if not quiet:
        print(
            f"fold={int(fold_i)} label={label} train={len(ds_train)} val={len(ds_val)} "
            f"test={len(ds_test)} vars={int(feature_cols.size)}"
        )

    it = tqdm(
        range(1, int(epochs) + 1),
        desc=f"Finetune-{label}-F{int(fold_i):02d}",
        unit="epoch",
        disable=quiet,
        dynamic_ncols=True,
    )
    for epoch in it:
        model.train()
        se_train = 0.0
        n_train = 0
        for xb, yb in dl_train:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = mse(pred, yb[:, 0])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            diff = (pred.detach() - yb[:, 0]) * float(LABEL_SCALE)
            se_train += float((diff * diff).sum().item())
            n_train += int(diff.numel())

        train_rmse = math.sqrt(se_train / max(n_train, 1))
        if dl_val is not None:
            val_metrics = eval_last_step(model, dl_val, device=device)
            last_val_rmse = float(val_metrics["rmse"])
        else:
            val_metrics = {"rmse": float("inf"), "mae": float("inf")}
            last_val_rmse = float("inf")

        improved = float(val_metrics["rmse"]) < float(best_val_rmse) - 1e-8
        if improved:
            best_val_rmse = float(val_metrics["rmse"])
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = int(early_stop_patience)
            if run_dir is not None:
                fold_dir = run_dir / f"fold_{int(fold_i):02d}"
                _save_checkpoint(
                    fold_dir / "reg_best.pt",
                    model=model,
                    mean=mean,
                    std=std,
                    feature_cols=feature_cols,
                    n_emg_norm=int(corpus.n_emg),
                    extra={
                        "stage": "best_val",
                        "fold": int(fold_i),
                        "epoch": int(epoch),
                        "best_val_rmse": float(best_val_rmse),
                        "train_stride": int(train_stride),
                        "label": str(label),
                        "split_manifest": split_manifest,
                    },
                )
                with open(fold_dir / "split_manifest.json", "w", encoding="utf-8") as f:
                    json.dump(split_manifest, f, indent=2)
        elif int(early_stop_patience) > 0:
            patience_left -= 1
            if patience_left <= 0:
                if not quiet:
                    print(f"Early stop at epoch {int(epoch)} (no val improvement for {int(early_stop_patience)} epochs)")
                break

        it.set_postfix_str(f"tr={train_rmse:.1f} vl={val_metrics['rmse']:.1f} bv={best_val_rmse:.1f}@{best_epoch}")

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    test_metrics = eval_last_step(model, dl_test, device=device)
    rollout_metrics = eval_recording_rollout(
        model,
        corpus=corpus,
        file_ids=tuple(test_file_ids),
        mean=mean,
        std=std,
        feature_cols=feature_cols,
        device=device,
    )

    out = {
        "fold": int(fold_i),
        "label": str(label),
        "n_train": int(len(ds_train)),
        "n_val": int(len(ds_val)),
        "n_test": int(len(ds_test)),
        "best_val_rmse": float(best_val_rmse),
        "best_epoch": int(best_epoch),
        "final_val_rmse": float(last_val_rmse),
        "test_rmse": float(test_metrics["rmse"]),
        "test_seq_rmse": float(rollout_metrics["rmse"]),
        "test_mae": float(test_metrics["mae"]),
        "test_seq_mae": float(rollout_metrics["mae"]),
        "best_rmse": float(test_metrics["rmse"]),
        "held_out_file_ids": [int(x) for x in test_file_ids],
        "held_out_file_names": [str(corpus.file_names[int(x)]) for x in test_file_ids],
    }

    if run_dir is not None:
        fold_dir = run_dir / f"fold_{int(fold_i):02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        _save_checkpoint(
            fold_dir / "reg_last.pt",
            model=model,
            mean=mean,
            std=std,
            feature_cols=feature_cols,
            n_emg_norm=int(corpus.n_emg),
            extra={
                "stage": "last",
                "fold": int(fold_i),
                "epoch": int(best_epoch),
                "best_val_rmse": float(best_val_rmse),
                "train_stride": int(train_stride),
                "label": str(label),
                "split_manifest": split_manifest,
            },
        )
        with open(fold_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        with open(fold_dir / "split_manifest.json", "w", encoding="utf-8") as f:
            json.dump(split_manifest, f, indent=2)

    return out


def _train_cv(
    *,
    corpus: RecordingCorpus,
    run_dir: Path,
    label: str,
    feature_cols: np.ndarray,
    n_emg_selected: int,
    n_imu_selected: int,
    device: torch.device,
) -> list[dict]:
    n_files = int(len(corpus.file_names))
    if n_files < 2:
        raise RuntimeError("The transformer training pipeline requires at least two recording files for LOFO evaluation.")

    all_fold_metrics: list[dict] = []
    cv_manifest = {
        "label": str(label),
        "split_strategy": "lofo",
        "seed": int(SEED),
        "recording_glob": str(DATA_GLOB),
        "n_folds": int(n_files),
        "folds": [],
    }

    file_splits = [
        (
            tuple(fid for fid in range(n_files) if fid != held_out),
            (int(held_out),),
        )
        for held_out in range(n_files)
    ]

    for fold_i, (train_file_ids, test_file_ids) in enumerate(file_splits, start=1):
        print(f"\n========== {label} | Fold {int(fold_i)}/{len(file_splits)} ==========")
        train_windows, val_windows, test_windows, train_files_final, val_files = build_fold_windows(
            corpus,
            train_file_ids=tuple(train_file_ids),
            test_file_ids=tuple(test_file_ids),
            seed=int(SEED + fold_i * 1000),
            train_stride=int(TRAIN_STRIDE),
            eval_stride=int(EVAL_STRIDE),
        )
        metrics = train_one_fold(
            corpus=corpus,
            train_windows=train_windows,
            val_windows=val_windows,
            test_windows=test_windows,
            train_file_ids=tuple(train_files_final),
            val_file_ids=tuple(val_files),
            test_file_ids=tuple(test_file_ids),
            feature_cols=np.asarray(feature_cols, dtype=np.int64),
            n_emg_selected=int(n_emg_selected),
            n_imu_selected=int(n_imu_selected),
            device=device,
            run_dir=run_dir,
            fold_i=int(fold_i),
            label=str(label),
            quiet=False,
            epochs=int(EPOCHS),
            early_stop_patience=int(EARLY_STOP_PATIENCE),
            train_stride=int(TRAIN_STRIDE),
        )
        all_fold_metrics.append(metrics)
        cv_manifest["folds"].append(
            {
                "fold": int(fold_i),
                "test_file_ids": [int(x) for x in test_file_ids],
                "test_file_names": [str(corpus.file_names[int(x)]) for x in test_file_ids],
                "test_indices": [],
                "checkpoint": str(run_dir / f"fold_{int(fold_i):02d}" / "reg_best.pt"),
                "split_manifest": str(run_dir / f"fold_{int(fold_i):02d}" / "split_manifest.json"),
            }
        )

    with open(run_dir / "cv_manifest.json", "w", encoding="utf-8") as f:
        json.dump(cv_manifest, f, indent=2)
    return all_fold_metrics


def _ablation_configs(corpus: RecordingCorpus) -> list[tuple[str, np.ndarray, int, int, str]]:
    all_cols = np.arange(corpus.n_features, dtype=np.int64)
    emg_cols = np.arange(0, corpus.n_emg, dtype=np.int64)
    imu_cols = np.arange(corpus.n_features - corpus.n_imu, corpus.n_features, dtype=np.int64)
    return [
        ("all", all_cols, int(corpus.n_emg), int(corpus.n_imu), "ALL"),
        ("thigh_only", imu_cols, 0, int(corpus.n_imu), "THIGH-ONLY"),
        ("emg_only", emg_cols, int(corpus.n_emg), 0, "EMG-ONLY"),
    ]


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

    corpus = load_recording_corpus()
    n_files = int(len(corpus.file_names))
    print(
        f"Recordings: files={n_files} sample_hz={corpus.sample_hz:.1f} "
        f"features={corpus.n_features} (emg={corpus.n_emg}, imu={corpus.n_imu})"
    )
    print(
        f"Training windows: source={int(SOURCE_WINDOW)} context={int(CONTEXT_WINDOW)} "
        f"train_stride={int(TRAIN_STRIDE)} eval_stride={int(EVAL_STRIDE)}"
    )
    for i, name in enumerate(corpus.file_names):
        n_steps = int(corpus.y_list[i].shape[0])
        n_train_windows = max(0, n_steps - int(SOURCE_WINDOW) + 1)
        print(f"  file {int(i):02d}: {name}  T={n_steps}  stride1_windows={n_train_windows}")

    if n_files < 2:
        raise SystemExit("Need at least two recording files for LOFO training.")

    results: dict[str, float] = {}
    fold_summaries: dict[str, list[dict]] = {}

    print("\n" + "=" * 60)
    print("ABLATION STUDY: ALL vs THIGH-ONLY vs EMG-ONLY")
    print("=" * 60)

    for key, feature_cols, n_emg_selected, n_imu_selected, label in _ablation_configs(corpus):
        print(f"\n>>> {label}")
        run_dir = SAVE_DIR / (RUN_NAME + f"_{key}")
        run_dir.mkdir(parents=True, exist_ok=True)
        fold_metrics = _train_cv(
            corpus=corpus,
            run_dir=run_dir,
            label=str(label),
            feature_cols=feature_cols,
            n_emg_selected=int(n_emg_selected),
            n_imu_selected=int(n_imu_selected),
            device=device,
        )
        mean_rmse = float(np.mean([m["best_rmse"] for m in fold_metrics]))
        results[str(key)] = mean_rmse
        fold_summaries[str(key)] = fold_metrics

    rmse_all = float(results["all"])
    rmse_thigh = float(results["thigh_only"])
    rmse_emg = float(results["emg_only"])
    emg_improvement = float(rmse_thigh - rmse_all)

    print("\n" + "=" * 60)
    print("ABLATION RESULTS (mean best RMSE across folds)")
    print("=" * 60)
    print(f"  EMG + thigh: {rmse_all:8.3f} deg")
    print(f"  Thigh only:  {rmse_thigh:8.3f} deg")
    print(f"  EMG only:    {rmse_emg:8.3f} deg")
    print("-" * 60)
    if emg_improvement > 0.5:
        print(f"  EMG helps: -{emg_improvement:.2f} deg RMSE")
    elif emg_improvement > 0.0:
        print(f"  EMG helps marginally: -{emg_improvement:.2f} deg RMSE")
    else:
        print(f"  WARNING: EMG not helping (thigh-only is {-emg_improvement:.2f} deg better)")

    summary = {
        "ablation": results,
        "emg_improvement_degrees": float(emg_improvement),
        "folds_all": fold_summaries["all"],
        "folds_thigh": fold_summaries["thigh_only"],
        "folds_emg": fold_summaries["emg_only"],
        "config": {
            "recording_glob": str(DATA_GLOB),
            "seed": int(SEED),
            "window": int(SOURCE_WINDOW),
            "context_window": int(CONTEXT_WINDOW),
            "train_stride": int(TRAIN_STRIDE),
            "eval_stride": int(EVAL_STRIDE),
            "label_shift": int(LABEL_SHIFT),
            "model": {
                "architecture": "sensor_fusion_last_step_transformer",
                "d_model": int(D_MODEL),
                "n_heads": int(N_HEADS),
                "d_ff": int(D_FF),
                "n_layers": int(N_LAYERS),
                "dropout": float(DROPOUT),
                "causal": bool(CAUSAL_ATTENTION),
            },
            "train": {
                "batch_size": int(BATCH_SIZE),
                "lr": float(LR),
                "weight_decay": float(WEIGHT_DECAY),
                "epochs": int(EPOCHS),
                "early_stop_patience": int(EARLY_STOP_PATIENCE),
                "train_noise_std": float(TRAIN_NOISE_STD),
                "train_drop_imu_prob": float(TRAIN_DROP_IMU_PROB),
            },
        },
    }
    summary_path = SAVE_DIR / RUN_NAME / "ablation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved ablation summary to: {summary_path}")


if __name__ == "__main__":
    main()
