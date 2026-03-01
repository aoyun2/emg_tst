"""Quick simulation demo: external OpenSim query matched against CMU DB, evaluated with MoCapAct."""
import numpy as np
from mocap_evaluation.external_sample_data import extract_external_sample_curves
from mocap_evaluation.mocap_loader import load_aggregated_database
from mocap_evaluation.motion_matching import find_best_match
from mocap_evaluation.mocapact_sim import simulate_prosthetic_walking_mocapact

# ── Noise levels ──────────────────────────────────────────────────────────
GOOD_NOISE_STD = 5.0    # degrees — realistic model error
BAD_NOISE_STD = 25.0    # degrees — poor model, should cause instability

# ── External query (OpenSim IK from GitHub, NOT from CMU DB) ──────────────
print("Downloading external OpenSim gait data...")
curves = extract_external_sample_curves(seconds=10.0, pred_noise_std=GOOD_NOISE_STD)
knee_query = curves.knee_label_included_deg
thigh_query = curves.thigh_angle_deg
pred_knee_good = curves.predicted_knee_included_deg

# Generate a bad prediction with much higher noise
rng = np.random.default_rng(42)
pred_knee_bad = np.clip(
    knee_query + rng.normal(0.0, BAD_NOISE_STD, len(knee_query)).astype(np.float32),
    0.0, 180.0,
)

print(f"Source: {curves.source_file}")
print(f"Query: {len(knee_query)} frames @ {curves.fps} Hz")
print(f"Knee range: [{knee_query.min():.1f}, {knee_query.max():.1f}] deg (included)")

# ── Load CMU DB (cached, no download) ─────────────────────────────────────
db = load_aggregated_database(mocap_root='mocap_data', try_download=False, use_cache=True)
print(f"CMU DB: {len(db['knee_right'])/db['fps']:.0f}s, {len(db['file_boundaries'])} files")

# ── Motion matching: OpenSim query vs CMU DB ──────────────────────────────
print("\nRunning motion matching (L2 pre-filter + DTW)...")
best_start, dist, segment = find_best_match(
    knee_query, thigh_query, db,
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

# Shared kwargs for all three MoCapAct simulation calls
_sim_kwargs = dict(
    mocap_db=db,
    best_start=best_start,
)

# ── Simulate: GT knee ────────────────────────────────────────────────────
print("\n--- GT knee simulation (OpenSim query) ---")
gt = simulate_prosthetic_walking_mocapact(
    knee_query, reference_knee=knee_query, **_sim_kwargs,
)

# ── Simulate: good prediction ────────────────────────────────────────────
print(f"\n--- Good prediction simulation (noise std={GOOD_NOISE_STD} deg) ---")
pred_good = simulate_prosthetic_walking_mocapact(
    pred_knee_good, reference_knee=knee_query, **_sim_kwargs,
)

# ── Simulate: bad prediction ─────────────────────────────────────────────
print(f"\n--- Bad prediction simulation (noise std={BAD_NOISE_STD} deg) ---")
pred_bad = simulate_prosthetic_walking_mocapact(
    pred_knee_bad, reference_knee=knee_query, **_sim_kwargs,
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
good_rmse = float(np.sqrt(np.mean((pred_knee_good[:T] - knee_query[:T]) ** 2)))
show(f"Good Prediction (noise std={GOOD_NOISE_STD}, RMSE={good_rmse:.1f})", pred_good)
print()
bad_rmse = float(np.sqrt(np.mean((pred_knee_bad[:T] - knee_query[:T]) ** 2)))
show(f"Bad Prediction (noise std={BAD_NOISE_STD}, RMSE={bad_rmse:.1f})", pred_bad)
print("=" * 60)

# ── Comparison plot ───────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

seg_len = T
fps = curves.fps
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
t = np.arange(seg_len) / fps

# Panel 1: Knee angles
axes[0].plot(t, np.asarray(gt['ref_knee_series'])[:seg_len], label='GT knee (OpenSim)', lw=2, color='blue')
axes[0].plot(t, np.asarray(pred_good['pred_knee_series'])[:seg_len], label=f'Good pred (RMSE={good_rmse:.1f})', lw=2, color='orange', alpha=0.8)
axes[0].plot(t, np.asarray(pred_bad['pred_knee_series'])[:seg_len], label=f'Bad pred (RMSE={bad_rmse:.1f})', lw=2, color='red', alpha=0.7)
axes[0].set_ylabel('Knee flexion (deg)')
axes[0].set_title(f'OpenSim query -> CMU match: {matched_file}  |  DTW={dist:.4f}  |  MatchRMSE={match_rmse:.1f}')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# Panel 2: CoM height
axes[1].plot(t, np.asarray(gt['com_height_series'])[:seg_len], label='GT CoM', lw=2, color='blue')
axes[1].plot(t, np.asarray(pred_good['com_height_series'])[:seg_len], label=f'Good pred CoM (stab={pred_good["stability_score"]:.2f})', lw=2, color='orange', alpha=0.8)
axes[1].plot(t, np.asarray(pred_bad['com_height_series'])[:seg_len], label=f'Bad pred CoM (stab={pred_bad["stability_score"]:.2f})', lw=2, color='red', alpha=0.7)
axes[1].axhline(0.55, color='r', ls='--', lw=1, label='Fall threshold')
axes[1].set_ylabel('CoM height (m)')
axes[1].set_title(f'CoM  |  GT stab={gt["stability_score"]:.3f}')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

# Panel 3: Foot contacts
for i, (label, m, color) in enumerate([('GT', gt, 'blue'), ('Good', pred_good, 'orange'), ('Bad', pred_bad, 'red')]):
    rc = np.asarray(m['right_contact_frames']) / fps
    lc = np.asarray(m['left_contact_frames']) / fps
    y = 1 - i * 0.25
    axes[2].scatter(rc, np.full_like(rc, y), marker='|', s=100, color=color, label=f'{label} R')
    axes[2].scatter(lc, np.full_like(lc, y + 0.08), marker='|', s=100, color=color, alpha=0.5, label=f'{label} L')

axes[2].set_xlabel('Time (s)')
axes[2].set_title(f'Foot contacts  |  GT steps={gt["step_count"]}, Good steps={pred_good["step_count"]}, Bad steps={pred_bad["step_count"]}')
axes[2].legend(loc='upper right', fontsize=7)
axes[2].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig('sim_comparison.png', dpi=150)
plt.close(fig)
print(f'\nPlot saved: sim_comparison.png')
