"""Quick simulation demo: external OpenSim query matched against CMU DB, evaluated with MoCapAct.

All three scenarios (GT knee / good prediction / bad prediction) are shown
simultaneously in separate viewer windows.  After DTW-matching the OpenSim
query to the CMU database, consecutive frames from the matched position are
extracted to drive the simulation — no cyclic repetition.
"""
import numpy as np
from mocap_evaluation.external_sample_data import extract_external_sample_curves
from mocap_evaluation.mocap_loader import load_aggregated_database
from mocap_evaluation.motion_matching import find_best_match
from mocap_evaluation.mocapact_sim import (
    SIM_FPS,
    simulate_three_scenarios_mocapact,
    resolve_clip_from_match,
)

# ── Tunable parameters ────────────────────────────────────────────────────────
GOOD_NOISE_STD = 5.0    # degrees — realistic model error
BAD_NOISE_STD  = 25.0   # degrees — poor model, should cause instability
TOTAL_SECONDS  = 30.0   # how long to simulate (capped by available CMU data)
MATCH_SECONDS  = 4.0    # query window sent to DTW motion matcher


def resample(arr: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    """Linear resample *arr* from *src_fps* to *dst_fps*."""
    if src_fps == dst_fps:
        return arr
    n_src = len(arr)
    n_dst = max(2, int(round(n_src * dst_fps / src_fps)))
    x_src = np.linspace(0.0, 1.0, n_src)
    x_dst = np.linspace(0.0, 1.0, n_dst)
    return np.interp(x_dst, x_src, arr).astype(np.float32)


# ── External query (OpenSim IK — used only for motion matching) ───────────────
print("Downloading external OpenSim gait data...")
curves = extract_external_sample_curves(seconds=MATCH_SECONDS, pred_noise_std=GOOD_NOISE_STD)

src_fps        = float(curves.fps)           # 200 Hz
knee_query_200 = curves.knee_label_included_deg
thigh_query_200 = curves.thigh_angle_deg

print(f"Source: {curves.source_file}")
print(f"Query: {len(knee_query_200)} frames @ {src_fps:.0f} Hz "
      f"({len(knee_query_200)/src_fps:.2f}s)")

# ── Load CMU DB ────────────────────────────────────────────────────────────────
db = load_aggregated_database(mocap_root='mocap_data', try_download=False, use_cache=True)
db_fps = float(db['fps'])   # 200 Hz (TARGET_FPS)
print(f"CMU DB: {len(db['knee_right'])/db_fps:.0f}s, {len(db['file_boundaries'])} files")

# ── Motion matching: DTW against CMU DB ───────────────────────────────────────
print("\nRunning motion matching (L2 pre-filter + DTW)...")
best_start, dist, segment = find_best_match(knee_query_200, thigh_query_200, db)

matched_file = "?"
file_end = len(db['knee_right'])
for s, e, f, c in db['file_boundaries']:
    if s <= best_start < e:
        matched_file = f
        file_end = int(e)
        break

knee_matched = segment["knee_right"]
T_cmp = min(len(knee_matched), len(knee_query_200))
match_rmse = float(np.sqrt(np.mean((knee_matched[:T_cmp] - knee_query_200[:T_cmp]) ** 2)))
print(f"Best match: DTW={dist:.4f}, file={matched_file}, frame={best_start}")
print(f"Matched knee vs query RMSE: {match_rmse:.2f} deg")

# ── Extract consecutive CMU frames from best_start ────────────────────────────
# Use the actual motion data that follows the matched position in the database
# (different frames every step, not a cyclic repeat of the query window).
T_db_want  = int(round(TOTAL_SECONDS * db_fps))
avail_200  = file_end - best_start          # frames available in this file
T_db_take  = min(avail_200, T_db_want)

knee_gt_200  = db['knee_right'][best_start : best_start + T_db_take].copy()
# hip_right gives the thigh/hip sagittal angle — best proxy for thigh in CMU db
hip_gt_200   = db.get('hip_right', db['knee_right'])[best_start : best_start + T_db_take].copy()

# Resample from 200 Hz → simulation rate (30 Hz)
knee_gt  = resample(knee_gt_200,  db_fps, SIM_FPS)
thigh_gt = resample(hip_gt_200,   db_fps, SIM_FPS)
T_sim    = len(knee_gt)
actual_seconds = T_sim / SIM_FPS

print(f"\nExtracted {T_db_take} CMU frames ({T_db_take/db_fps:.1f}s) from {matched_file}")
print(f"Resampled to {SIM_FPS:.0f} Hz: {T_sim} frames ({actual_seconds:.1f}s)")
if T_db_take < T_db_want:
    print(f"  [note] File only had {avail_200/db_fps:.1f}s from match point; "
          f"using {actual_seconds:.1f}s (still loops in viewer)")

# ── Build noisy variants ──────────────────────────────────────────────────────
rng = np.random.default_rng(42)
knee_good = np.clip(
    knee_gt + rng.normal(0.0, GOOD_NOISE_STD, T_sim).astype(np.float32),
    0.0, 180.0,
)
knee_bad = np.clip(
    knee_gt + rng.normal(0.0, BAD_NOISE_STD, T_sim).astype(np.float32),
    0.0, 180.0,
)

good_rmse = float(np.sqrt(np.mean((knee_good - knee_gt) ** 2)))
bad_rmse  = float(np.sqrt(np.mean((knee_bad  - knee_gt) ** 2)))
print(f"Good pred RMSE={good_rmse:.1f} deg, Bad pred RMSE={bad_rmse:.1f} deg")

# ── Resolve DTW match → specific CMU clip for MoCapAct env init ──────────────
match_info = resolve_clip_from_match(best_start, len(knee_query_200), db)

# ── Simulate all three scenarios simultaneously ───────────────────────────────
print(f"\n--- Simulating {actual_seconds:.1f}s: GT / good / bad knee "
      f"(noise std={GOOD_NOISE_STD}/{BAD_NOISE_STD} deg) ---")
print("    (Viewer loops automatically — close all windows to finish)")

gt, pred_good, pred_bad = simulate_three_scenarios_mocapact(
    gt_knee=knee_gt,
    nominal_knee=knee_good,
    bad_knee=knee_bad,
    sample_thigh_right=thigh_gt,
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

print("\n" + "=" * 60)
print(f"Query: {curves.source_file}  |  Matched: {matched_file}")
print(f"DTW: {dist:.4f}  |  Match RMSE: {match_rmse:.2f} deg  |  Sim: {actual_seconds:.1f}s")
print()
show("Ground Truth (CMU knee)", gt)
print()
show(f"Good Prediction (noise std={GOOD_NOISE_STD}, RMSE={good_rmse:.1f})", pred_good)
print()
show(f"Bad Prediction  (noise std={BAD_NOISE_STD}, RMSE={bad_rmse:.1f})", pred_bad)
print("=" * 60)

# ── Comparison plot ────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

t_ax = np.arange(T_sim) / SIM_FPS

fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

axes[0].plot(t_ax, knee_gt,   label='GT knee (CMU)', lw=2, color='blue')
axes[0].plot(t_ax, knee_good, label=f'Good pred (RMSE={good_rmse:.1f})', lw=2, color='orange', alpha=0.8)
axes[0].plot(t_ax, knee_bad,  label=f'Bad pred (RMSE={bad_rmse:.1f})',  lw=2, color='red',    alpha=0.7)
axes[0].set_ylabel('Knee (deg, included)')
axes[0].set_title(f'OpenSim query -> CMU match: {matched_file}  |  DTW={dist:.4f}  |  MatchRMSE={match_rmse:.1f}')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

axes[1].plot(t_ax, np.asarray(gt['com_height_series'])[:T_sim],
             label='GT CoM', lw=2, color='blue')
axes[1].plot(t_ax, np.asarray(pred_good['com_height_series'])[:T_sim],
             label=f'Good CoM (stab={pred_good["stability_score"]:.2f})', lw=2, color='orange', alpha=0.8)
axes[1].plot(t_ax, np.asarray(pred_bad['com_height_series'])[:T_sim],
             label=f'Bad CoM (stab={pred_bad["stability_score"]:.2f})',  lw=2, color='red',    alpha=0.7)
axes[1].axhline(0.55, color='r', ls='--', lw=1, label='Fall threshold')
axes[1].set_ylabel('CoM height (m)')
axes[1].set_title(f'CoM  |  GT stab={gt["stability_score"]:.3f}')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

for i, (label, m, color) in enumerate([('GT', gt, 'blue'), ('Good', pred_good, 'orange'), ('Bad', pred_bad, 'red')]):
    rc = np.asarray(m['right_contact_frames']) / SIM_FPS
    lc = np.asarray(m['left_contact_frames'])  / SIM_FPS
    y  = 1 - i * 0.25
    axes[2].scatter(rc, np.full_like(rc, y),        marker='|', s=100, color=color,            label=f'{label} R')
    axes[2].scatter(lc, np.full_like(lc, y + 0.08), marker='|', s=100, color=color, alpha=0.5, label=f'{label} L')
axes[2].set_xlabel('Time (s)')
axes[2].set_title(f'Foot contacts  |  steps: GT={gt["step_count"]}, '
                  f'Good={pred_good["step_count"]}, Bad={pred_bad["step_count"]}')
axes[2].legend(loc='upper right', fontsize=7)
axes[2].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig('sim_comparison.png', dpi=150)
plt.close(fig)
print(f'\nPlot saved: sim_comparison.png')
