"""Quick simulation demo: sample data matched against CMU DB, evaluated with MoCapAct.

The query is built by concatenating consecutive 1-second windows from the sample
source (rigtest recording or external OpenSim data).  DTW matching against the CMU
database returns a segment of the same length; consecutive frames from that position
are then used to fill TOTAL_SECONDS of simulation (resampled to 30 Hz).

Usage
-----
# Use rigtest recording (default):
    python run_quick_sim.py [--data-file data0.npy] [--n-windows N]

# Use external OpenSim sample data:
    python run_quick_sim.py --test-sample [--n-windows N]
"""
import argparse
import glob
import sys

import numpy as np

from mocap_evaluation.external_sample_data import extract_external_sample_curves
from mocap_evaluation.mocap_loader import load_aggregated_database, TARGET_FPS
from mocap_evaluation.motion_matching import find_best_match
from mocap_evaluation.mocapact_sim import (
    SIM_FPS,
    simulate_three_scenarios_mocapact,
    resolve_clip_from_match,
)

# ── Tunable parameters ────────────────────────────────────────────────────────
GOOD_NOISE_STD = 5.0    # degrees — realistic model error
BAD_NOISE_STD  = 25.0   # degrees — poor model, should cause instability
TOTAL_SECONDS  = 30.0   # total simulation length

WINDOW_FRAMES  = TARGET_FPS   # 1-second window at 200 Hz


def resample(arr: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    """Linear resample *arr* from *src_fps* to *dst_fps*."""
    if src_fps == dst_fps:
        return arr
    n_src = len(arr)
    n_dst = max(2, int(round(n_src * dst_fps / src_fps)))
    x_src = np.linspace(0.0, 1.0, n_src)
    x_dst = np.linspace(0.0, 1.0, n_dst)
    return np.interp(x_dst, x_src, arr).astype(np.float32)


# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument(
    "--test-sample", action="store_true",
    help="Use external OpenSim data instead of a rigtest recording",
)
parser.add_argument(
    "--data-file", default=None,
    help="Rigtest .npy file to use (default: first data*.npy found)",
)
parser.add_argument(
    "--n-windows", type=int, default=None,
    help="How many 1-second windows to aggregate (default: all available, up to TOTAL_SECONDS)",
)
args = parser.parse_args()

max_windows = int(TOTAL_SECONDS)   # cap to avoid extremely long DTW queries

# ── Resolve data source: auto-fall-back to external if no recordings found ────
if not args.test_sample and not args.data_file:
    candidates = sorted(p for p in glob.glob("data*.npy") if "samples" not in p)
    if not candidates:
        print("No data*.npy recording found — using external OpenSim data.")
        print("Run rigtest.py to capture your own data, or pass --data-file <path>.\n")
        args.test_sample = True
    else:
        args.data_file = candidates[0]

# ── Build the aggregated query from consecutive 1-second windows ──────────────
if args.test_sample:
    # ── External OpenSim path ─────────────────────────────────────────────────
    print("Downloading external OpenSim gait data...")
    curves = extract_external_sample_curves(seconds=TOTAL_SECONDS, pred_noise_std=GOOD_NOISE_STD)
    full_knee  = curves.knee_label_included_deg   # at TARGET_FPS
    full_thigh = curves.thigh_angle_deg

    n_complete = len(full_knee) // WINDOW_FRAMES
    if n_complete == 0:
        # Less than one full window — use whatever is available
        knee_query_200  = full_knee
        thigh_query_200 = full_thigh
        n_windows_used  = 1
    else:
        n_windows_used = (min(n_complete, max_windows)
                          if args.n_windows is None else min(args.n_windows, n_complete))
        keep = n_windows_used * WINDOW_FRAMES
        knee_query_200  = full_knee[:keep]
        thigh_query_200 = full_thigh[:keep]

    source_label = f"OpenSim {curves.source_file.split('/')[-1]}"
    print(f"Source: {curves.source_file}")

else:
    # ── Rigtest recording path — all data*.npy files ──────────────────────────
    from emg_tst.data import load_recording

    if args.data_file:
        all_data_files = [args.data_file]
    else:
        all_data_files = sorted(p for p in glob.glob("data*.npy") if "samples" not in p)

    knee_chunks: list[np.ndarray] = []
    thigh_chunks: list[np.ndarray] = []
    total_windows = 0

    for data_path in all_data_files:
        X, y, meta = load_recording(data_path)
        fk = y.astype(np.float32)
        ft = X[:, -1].astype(np.float32)
        n_complete = len(fk) // WINDOW_FRAMES
        if n_complete == 0:
            print(f"  {data_path}: too short ({len(fk)} frames), skipping")
            continue
        keep = n_complete * WINDOW_FRAMES
        knee_chunks.append(fk[:keep])
        thigh_chunks.append(ft[:keep])
        total_windows += n_complete
        print(f"  {data_path}: {n_complete} window(s), ~{meta['effective_hz']:.0f} Hz")

    if not knee_chunks:
        sys.exit("No usable windows found in any recording file.")

    # Concatenate windows across all files into one long query
    full_knee_all  = np.concatenate(knee_chunks)
    full_thigh_all = np.concatenate(thigh_chunks)

    n_windows_used = (min(total_windows, max_windows)
                      if args.n_windows is None else min(args.n_windows, total_windows))
    keep = n_windows_used * WINDOW_FRAMES
    knee_query_200  = full_knee_all[:keep]
    thigh_query_200 = full_thigh_all[:keep]

    source_label = f"rigtest ({len(all_data_files)} file(s))"

query_seconds = len(knee_query_200) / TARGET_FPS
print(f"Query: {n_windows_used} × 1s window(s) = {len(knee_query_200)} frames "
      f"({query_seconds:.1f}s) at {TARGET_FPS} Hz")

# ── Load CMU DB ────────────────────────────────────────────────────────────────
db = load_aggregated_database(mocap_root='mocap_data', try_download=False, use_cache=True)
print(f"CMU DB: {len(db['knee_right'])/db['fps']:.0f}s, {len(db['file_boundaries'])} files")

# ── Motion matching: DTW on the full aggregated query ─────────────────────────
print(f"\nRunning motion matching on {query_seconds:.1f}s query (L2 pre-filter + DTW)...")
best_start, dist, segment = find_best_match(knee_query_200, thigh_query_200, db)

matched_file = "?"
for s, e, f, c in db["file_boundaries"]:
    if s <= best_start < e:
        matched_file = f
        break

knee_matched = segment["knee_right"]
T_cmp = min(len(knee_matched), len(knee_query_200))
match_rmse = float(np.sqrt(np.mean((knee_matched[:T_cmp] - knee_query_200[:T_cmp]) ** 2)))
print(f"Best match: DTW={dist:.4f}, file={matched_file}")
print(f"Matched knee vs query RMSE: {match_rmse:.2f} deg")

# ── Resolve DTW match → specific CMU clip ─────────────────────────────────────
match_info = resolve_clip_from_match(best_start, len(knee_query_200), db)

# ── Extend matched position with consecutive CMU frames ───────────────────────
# The query already spans query_seconds; take consecutive CMU frames from
# best_start to cover TOTAL_SECONDS of simulation without repeating.
db_fps  = float(db['fps'])
T_db    = int(round(TOTAL_SECONDS * db_fps))
end_idx = min(best_start + T_db, len(db['knee_right']))

cmu_knee  = db['knee_right'][best_start:end_idx]
cmu_thigh = db['hip_right'][best_start:end_idx]

knee_gt     = resample(cmu_knee,  db_fps, SIM_FPS)
thigh_sim   = resample(cmu_thigh, db_fps, SIM_FPS)
T_sim       = len(knee_gt)
actual_seconds = T_sim / SIM_FPS

print(f"Simulation motion: {end_idx - best_start} frames @ {db_fps:.0f} Hz "
      f"→ {T_sim} frames @ {SIM_FPS:.0f} Hz ({actual_seconds:.1f}s)")

# ── Predictions: add noise to the GT motion signal ───────────────────────────
rng = np.random.default_rng(42)
knee_good = np.clip(
    knee_gt + rng.normal(0.0, GOOD_NOISE_STD, T_sim).astype(np.float32),
    0.0, 180.0,
)
knee_bad = np.clip(
    knee_gt + rng.normal(0.0, BAD_NOISE_STD, T_sim).astype(np.float32),
    0.0, 180.0,
)

print(f"Knee range: [{knee_gt.min():.1f}, {knee_gt.max():.1f}] deg (included)")

# ── Simulate all three scenarios simultaneously ───────────────────────────────
print(f"\n--- Simulating {actual_seconds:.1f}s with GT / good / bad knee "
      f"(noise={GOOD_NOISE_STD}/{BAD_NOISE_STD} deg) ---")

gt, pred_good, pred_bad = simulate_three_scenarios_mocapact(
    gt_knee=knee_gt,
    nominal_knee=knee_good,
    bad_knee=knee_bad,
    sample_thigh_right=thigh_sim,
    reference_knee=knee_gt,
    eval_seconds=actual_seconds,
    use_gui=True,
    match_info=match_info,
)

# ── Results ───────────────────────────────────────────────────────────────────
def show(label, m):
    print(f'{label}:')
    print(f'  fall: {m["fall_detected"]} (frame {m["fall_frame"]})')
    print(f'  steps: {m["step_count"]}, gait_symmetry: {m["gait_symmetry"]:.3f}')
    print(f'  stability: {m["stability_score"]:.3f}')
    print(f'  CoM: {m["com_height_mean"]:.3f} +/- {m["com_height_std"]:.3f} m')
    print(f'  knee RMSE: {m["knee_rmse_deg"]:.2f}, MAE: {m["knee_mae_deg"]:.2f}')

T_plot    = T_sim
good_rmse = float(np.sqrt(np.mean((knee_good[:T_plot] - knee_gt[:T_plot]) ** 2)))
bad_rmse  = float(np.sqrt(np.mean((knee_bad[:T_plot]  - knee_gt[:T_plot]) ** 2)))

print("\n" + "=" * 60)
print(f"Source: {source_label}  |  Matched: {matched_file}")
print(f"Query: {n_windows_used} × 1s window(s) ({query_seconds:.1f}s)  |  DTW={dist:.4f}  |  MatchRMSE={match_rmse:.2f} deg")
print()
show("Ground Truth (CMU knee)", gt)
print()
show(f"Good Prediction (noise std={GOOD_NOISE_STD}, RMSE={good_rmse:.1f})", pred_good)
print()
show(f"Bad Prediction (noise std={BAD_NOISE_STD}, RMSE={bad_rmse:.1f})", pred_bad)
print("=" * 60)

# ── Comparison plot ────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fps  = SIM_FPS
t_ax = np.arange(T_plot) / fps

fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

axes[0].plot(t_ax, knee_gt[:T_plot],   label='GT knee (CMU)', lw=2, color='blue')
axes[0].plot(t_ax, knee_good[:T_plot], label=f'Good pred (RMSE={good_rmse:.1f})', lw=2, color='orange', alpha=0.8)
axes[0].plot(t_ax, knee_bad[:T_plot],  label=f'Bad pred (RMSE={bad_rmse:.1f})',  lw=2, color='red',    alpha=0.7)
axes[0].set_ylabel('Knee flexion (deg)')
axes[0].set_title(
    f'{source_label} ({n_windows_used}×1s windows) → CMU: {matched_file}  '
    f'|  DTW={dist:.4f}  |  MatchRMSE={match_rmse:.1f}'
)
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

axes[1].plot(t_ax, np.asarray(gt['com_height_series'])[:T_plot],
             label='GT CoM', lw=2, color='blue')
axes[1].plot(t_ax, np.asarray(pred_good['com_height_series'])[:T_plot],
             label=f'Good pred CoM (stab={pred_good["stability_score"]:.2f})', lw=2, color='orange', alpha=0.8)
axes[1].plot(t_ax, np.asarray(pred_bad['com_height_series'])[:T_plot],
             label=f'Bad pred CoM (stab={pred_bad["stability_score"]:.2f})',  lw=2, color='red',    alpha=0.7)
axes[1].axhline(0.55, color='r', ls='--', lw=1, label='Fall threshold')
axes[1].set_ylabel('CoM height (m)')
axes[1].set_title(f'CoM  |  GT stab={gt["stability_score"]:.3f}')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

for i, (label, m, color) in enumerate([('GT', gt, 'blue'), ('Good', pred_good, 'orange'), ('Bad', pred_bad, 'red')]):
    rc = np.asarray(m['right_contact_frames']) / fps
    lc = np.asarray(m['left_contact_frames'])  / fps
    y  = 1 - i * 0.25
    axes[2].scatter(rc, np.full_like(rc, y),        marker='|', s=100, color=color,            label=f'{label} R')
    axes[2].scatter(lc, np.full_like(lc, y + 0.08), marker='|', s=100, color=color, alpha=0.5, label=f'{label} L')

axes[2].set_xlabel('Time (s)')
axes[2].set_title(
    f'Foot contacts  |  GT steps={gt["step_count"]}, '
    f'Good steps={pred_good["step_count"]}, Bad steps={pred_bad["step_count"]}'
)
axes[2].legend(loc='upper right', fontsize=7)
axes[2].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig('sim_comparison.png', dpi=150)
plt.close(fig)
print(f'\nPlot saved: sim_comparison.png')
