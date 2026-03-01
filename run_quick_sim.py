"""Quick simulation demo: external OpenSim query matched against CMU DB, evaluated with MoCapAct.

All three scenarios (GT knee / good prediction / bad prediction) are shown
simultaneously in separate viewer windows.  The DTW motion match locates the
best-matching position in the CMU DB; consecutive frames from that position are
then aggregated to TOTAL_SECONDS of non-repeating motion (resampled from the
DB's 200 Hz to the simulation rate of 30 Hz).
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
TOTAL_SECONDS  = 30.0   # how long to simulate (data tiles automatically)
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


# ── External query (OpenSim IK from GitHub, NOT from CMU DB) ─────────────────
print("Downloading external OpenSim gait data...")
curves = extract_external_sample_curves(seconds=MATCH_SECONDS, pred_noise_std=GOOD_NOISE_STD)

src_fps = float(curves.fps)           # 200 Hz from external_sample_data

# Resample to simulation rate so walking speed is physiologically correct
knee_query_sim  = resample(curves.knee_label_included_deg, src_fps, SIM_FPS)
thigh_query_sim = resample(curves.thigh_angle_deg,         src_fps, SIM_FPS)

# Keep 200-Hz versions for motion matching (DTW uses fine-grained timing)
knee_query_200  = curves.knee_label_included_deg
thigh_query_200 = curves.thigh_angle_deg

print(f"Source: {curves.source_file}")
print(f"At {src_fps:.0f} Hz: {len(knee_query_200)} frames "
      f"({len(knee_query_200)/src_fps:.2f}s)")
print(f"Resampled to {SIM_FPS:.0f} Hz: {len(knee_query_sim)} frames "
      f"({len(knee_query_sim)/SIM_FPS:.2f}s)")

# ── Load CMU DB (cached, no download) ─────────────────────────────────────────
db = load_aggregated_database(mocap_root='mocap_data', try_download=False, use_cache=True)
print(f"CMU DB: {len(db['knee_right'])/db['fps']:.0f}s, {len(db['file_boundaries'])} files")

# ── Motion matching: use the 200-Hz query for best DTW accuracy ───────────────
print("\nRunning motion matching (L2 pre-filter + DTW)...")
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

# ── Resolve DTW match → specific CMU clip ────────────────────────────────────
match_info = resolve_clip_from_match(best_start, len(knee_query_200), db)

# ── Aggregate consecutive CMU DB frames to build the simulation motion ────────
# Instead of tiling the short OpenSim query, take consecutive frames from the
# CMU DB starting at the matched position for a realistic, non-repeating motion.
db_fps = float(db['fps'])                        # 200 Hz
T_db   = int(round(TOTAL_SECONDS * db_fps))      # frames needed at 200 Hz
end_idx = min(best_start + T_db, len(db['knee_right']))
cmu_knee  = db['knee_right'][best_start:end_idx]
cmu_thigh = db['hip_right'][best_start:end_idx]

knee_gt     = resample(cmu_knee,  db_fps, SIM_FPS)
thigh_tiled = resample(cmu_thigh, db_fps, SIM_FPS)
T_sim = len(knee_gt)
actual_seconds = T_sim / SIM_FPS

print(f"CMU motion window: {end_idx - best_start} frames @ {db_fps:.0f} Hz "
      f"→ {T_sim} frames @ {SIM_FPS:.0f} Hz ({actual_seconds:.1f}s)")

# Good prediction: low noise on the motion signal
rng = np.random.default_rng(42)
knee_good = np.clip(
    knee_gt + rng.normal(0.0, GOOD_NOISE_STD, T_sim).astype(np.float32),
    0.0, 180.0,
)

# Bad prediction: high noise
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
    sample_thigh_right=thigh_tiled,
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

T_plot = T_sim
good_rmse = float(np.sqrt(np.mean((knee_good[:T_plot] - knee_gt[:T_plot]) ** 2)))
bad_rmse  = float(np.sqrt(np.mean((knee_bad[:T_plot]  - knee_gt[:T_plot]) ** 2)))

print("\n" + "=" * 60)
print(f"Query: OpenSim subject01_walk1  |  Matched: {matched_file}")
print(f"DTW: {dist:.4f}  |  Match RMSE: {match_rmse:.2f} deg")
print()
show("Ground Truth (OpenSim knee)", gt)
print()
show(f"Good Prediction (noise std={GOOD_NOISE_STD}, RMSE={good_rmse:.1f})", pred_good)
print()
show(f"Bad Prediction (noise std={BAD_NOISE_STD}, RMSE={bad_rmse:.1f})", pred_bad)
print("=" * 60)

# ── Comparison plot ────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fps = SIM_FPS
t_ax = np.arange(T_plot) / fps

fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

# Panel 1: Knee angles
axes[0].plot(t_ax, knee_gt[:T_plot],   label='GT knee (OpenSim)', lw=2, color='blue')
axes[0].plot(t_ax, knee_good[:T_plot], label=f'Good pred (RMSE={good_rmse:.1f})', lw=2, color='orange', alpha=0.8)
axes[0].plot(t_ax, knee_bad[:T_plot],  label=f'Bad pred (RMSE={bad_rmse:.1f})',  lw=2, color='red',    alpha=0.7)
axes[0].set_ylabel('Knee flexion (deg)')
axes[0].set_title(f'OpenSim query -> CMU match: {matched_file}  |  DTW={dist:.4f}  |  MatchRMSE={match_rmse:.1f}')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# Panel 2: CoM height
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

# Panel 3: Foot contacts
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
