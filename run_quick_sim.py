"""Proper evaluation: external OpenSim query matched against CMU DB."""
import numpy as np
from mocap_evaluation.external_sample_data import extract_external_sample_curves
from mocap_evaluation.mocap_loader import load_aggregated_database
from mocap_evaluation.motion_matching import find_best_match
from mocap_evaluation.prosthetic_sim import simulate_prosthetic_walking

# ── External query (OpenSim IK from GitHub, NOT from CMU DB) ──────────────
print("Downloading external OpenSim gait data...")
curves = extract_external_sample_curves(seconds=5.0, pred_noise_std=5.0)
knee_query = curves.knee_label_included_deg
thigh_query = curves.thigh_angle_deg
pred_knee = curves.predicted_knee_included_deg
print(f"Source: {curves.source_file}")
print(f"Query: {len(knee_query)} frames @ {curves.fps} Hz")
print(f"Knee range: [{knee_query.min():.1f}, {knee_query.max():.1f}] deg (included)")

# ── Load CMU DB (cached, no download) ─────────────────────────────────────
db = load_aggregated_database(mocap_root='mocap_data', try_download=False, use_cache=True)
print(f"CMU DB: {len(db['knee_right'])/db['fps']:.0f}s, {len(db['file_boundaries'])} files")

# ── Motion matching: OpenSim query vs CMU DB ──────────────────────────────
print("\nRunning motion matching (L2 pre-filter + DTW)...")
best_start, dist, segment = find_best_match(
    knee_query, thigh_query, db, categories=["walk"],
)

# Find matched file
matched_file = "?"
for s, e, f, c in db["file_boundaries"]:
    if s <= best_start < e:
        matched_file = f
        break

knee_matched = segment["knee_right"]
T = min(len(knee_matched), len(knee_query))
match_rmse = float(np.sqrt(np.mean((knee_matched[:T] - knee_query[:T]) ** 2)))
print(f"Best match: DTW={dist:.4f}, file={matched_file}")
print(f"Matched knee vs query RMSE: {match_rmse:.2f} deg")

# ── Simulate: GT knee ────────────────────────────────────────────────────
print("\n--- GT knee simulation (OpenSim query) ---")
gt = simulate_prosthetic_walking(
    segment, knee_query, use_gui=False,
    sample_thigh_right=thigh_query,
    save_trajectory='gt_traj.npz',
)

# ── Simulate: predicted knee ──────────────────────────────────────────────
print("\n--- Prediction knee simulation (OpenSim + 5 deg noise) ---")
pred = simulate_prosthetic_walking(
    segment, pred_knee, use_gui=False,
    sample_thigh_right=thigh_query,
    save_trajectory='pred_traj.npz',
)

# ── Results ───────────────────────────────────────────────────────────────
def show(label, m):
    print(f'{label}:')
    print(f'  fall: {m["fall_detected"]} (frame {m["fall_frame"]})')
    print(f'  steps: {m["step_count"]}, gait_symmetry: {m["gait_symmetry"]:.3f}')
    print(f'  stability: {m["stability_score"]:.3f}')
    print(f'  CoM: {m["com_height_mean"]:.3f} +/- {m["com_height_std"]:.3f} m')
    print(f'  knee RMSE: {m["knee_rmse_deg"]:.2f}, MAE: {m["knee_mae_deg"]:.2f}')

print("\n" + "=" * 60)
print(f"Query: OpenSim subject01_walk1  |  Matched: {matched_file}")
print(f"DTW: {dist:.4f}  |  Match RMSE: {match_rmse:.2f} deg")
print()
show("Ground Truth (OpenSim knee)", gt)
print()
show("Prediction (OpenSim + 5 deg noise)", pred)
pred_rmse = float(np.sqrt(np.mean((pred_knee[:T] - knee_query[:T]) ** 2)))
print(f"\nPred vs label RMSE: {pred_rmse:.2f} deg")
print("=" * 60)

# ── Comparison plot ───────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

seg_len = T
fps = curves.fps
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
t = np.arange(seg_len) / fps

axes[0].plot(t, np.asarray(gt['ref_knee_series'])[:seg_len], label='Matched CMU knee (ref)', lw=1.5, color='green', alpha=0.7)
axes[0].plot(t, np.asarray(gt['pred_knee_series'])[:seg_len], label='GT knee (OpenSim)', lw=2, color='blue')
axes[0].plot(t, np.asarray(pred['pred_knee_series'])[:seg_len], label='Pred knee (+5 noise)', lw=2, color='orange', alpha=0.8)
axes[0].set_ylabel('Knee flexion (deg)')
axes[0].set_title(f'OpenSim query -> CMU match: {matched_file}  |  DTW={dist:.4f}  |  MatchRMSE={match_rmse:.1f}')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(t, np.asarray(gt['com_height_series'])[:seg_len], label='GT CoM', lw=2, color='blue')
axes[1].plot(t, np.asarray(pred['com_height_series'])[:seg_len], label='Pred CoM', lw=2, color='orange', alpha=0.8)
axes[1].axhline(0.55, color='r', ls='--', label='Fall threshold')
axes[1].set_ylabel('CoM height (m)')
axes[1].set_title(f'CoM  |  GT stab={gt["stability_score"]:.3f}, Pred stab={pred["stability_score"]:.3f}')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

for i, (label, m, color) in enumerate([('GT', gt, 'blue'), ('Pred', pred, 'orange')]):
    rc = np.asarray(m['right_contact_frames']) / fps
    lc = np.asarray(m['left_contact_frames']) / fps
    y = 1 - i * 0.3
    axes[2].scatter(rc, np.full_like(rc, y), marker='|', s=100, color=color, label=f'{label} R')
    axes[2].scatter(lc, np.full_like(lc, y + 0.1), marker='|', s=100, color=color, alpha=0.5, label=f'{label} L')

axes[2].set_xlabel('Time (s)')
axes[2].set_title(f'Foot contacts  |  GT steps={gt["step_count"]}, Pred steps={pred["step_count"]}')
axes[2].legend(loc='upper right', fontsize=8)
axes[2].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig('sim_comparison.png', dpi=150)
plt.close(fig)
print(f'\nPlot saved: sim_comparison.png')
