"""Proper evaluation: query from one walk file, match against rest of DB."""
import numpy as np
from mocap_evaluation.mocap_loader import load_aggregated_database, TARGET_FPS
from mocap_evaluation.motion_matching import find_best_match
from mocap_evaluation.prosthetic_sim import simulate_prosthetic_walking

# ── Load cached database ──────────────────────────────────────────────────
db = load_aggregated_database(mocap_root='mocap_data', try_download=False, use_cache=True)
bounds = db["file_boundaries"]
fps = db["fps"]

# ── Pick a query walk file ────────────────────────────────────────────────
walk_files = [(s, e, f, c) for s, e, f, c in bounds if c == "walk" and (e - s) >= 5 * fps]
print(f"Walk files with >=5s: {len(walk_files)}")

query_entry = walk_files[1]
qs, qe, qf, qc = query_entry
print(f"Query file: {qf} ({(qe-qs)/fps:.1f}s)")

# Take a 5s segment from the middle
seg_len = int(5.0 * fps)
mid = (qs + qe) // 2
q_start = mid - seg_len // 2
q_end = q_start + seg_len

knee_query = db["knee_right"][q_start:q_end].astype(np.float32)
thigh_query = db["hip_right"][q_start:q_end].astype(np.float32)
print(f"Query: {seg_len} frames, knee [{knee_query.min():.1f}, {knee_query.max():.1f}]")

# ── Build a DB copy EXCLUDING the query file ──────────────────────────────
print(f"Building search DB excluding {qf}...")
keys_1d = ["knee_right", "knee_left", "hip_right", "hip_left",
           "ankle_right", "ankle_left", "pelvis_tilt", "trunk_lean"]
keys_3d = ["root_pos"]

# Collect frame ranges to keep (everything except the query file)
keep_slices = []
for s, e, f, c in bounds:
    if f != qf:
        keep_slices.append((s, e, f, c))

# Build new arrays
search_db = {"fps": fps, "source": db["source"]}
for k in keys_1d:
    search_db[k] = np.concatenate([db[k][s:e] for s, e, _, _ in keep_slices])
for k in keys_3d:
    search_db[k] = np.concatenate([db[k][s:e] for s, e, _, _ in keep_slices])

# Rebuild file_boundaries and categories
new_bounds = []
new_cats = []
offset = 0
for s, e, f, c in keep_slices:
    length = e - s
    new_bounds.append((offset, offset + length, f, c))
    new_cats.extend([c] * length)
    offset += length
search_db["file_boundaries"] = new_bounds
search_db["categories"] = np.array(new_cats)

excluded_frames = qe - qs
total_frames = len(search_db["knee_right"])
print(f"Search DB: {total_frames/fps:.1f}s ({total_frames} frames, excluded {excluded_frames} from {qf})")

# Simulated prediction: GT + 5 deg noise
rng = np.random.default_rng(42)
pred_knee = np.clip(
    knee_query + rng.normal(0, 5.0, size=knee_query.shape).astype(np.float32),
    0.0, 180.0,
)

# ── Motion matching against the EXCLUDED DB ───────────────────────────────
print("\nRunning motion matching (L2 pre-filter + DTW)...")
best_start, dist, segment = find_best_match(
    knee_query, thigh_query, search_db, categories=["walk"],
)

# Find which file it matched
matched_file = "?"
for s, e, f, c in new_bounds:
    if s <= best_start < e:
        matched_file = f
        break

print(f"Best match: DTW={dist:.4f}, file={matched_file}")

knee_matched = segment["knee_right"]
match_rmse = float(np.sqrt(np.mean((knee_matched[:seg_len] - knee_query[:seg_len]) ** 2)))
print(f"Matched knee vs query RMSE: {match_rmse:.2f}° (non-zero = different motion)")

# ── Simulate: GT knee ────────────────────────────────────────────────────
print("\n--- GT knee simulation ---")
gt = simulate_prosthetic_walking(
    segment, knee_query, use_gui=False,
    sample_thigh_right=thigh_query,
    save_trajectory='gt_traj.npz',
)

# ── Simulate: predicted knee ──────────────────────────────────────────────
print("\n--- Prediction knee simulation ---")
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
print(f"Query: {qf}  |  Matched: {matched_file}  |  DTW: {dist:.4f}")
print(f"Match knee RMSE: {match_rmse:.2f} deg")
print()
show("Ground Truth (query knee drives right leg)", gt)
print()
show("Prediction (query knee + 5 deg noise)", pred)
print()
pred_rmse = float(np.sqrt(np.mean((pred_knee - knee_query) ** 2)))
print(f"Pred vs label RMSE: {pred_rmse:.2f} deg")
print("=" * 60)

# ── Comparison plot ───────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
t = np.arange(seg_len) / fps

axes[0].plot(t, np.asarray(gt['ref_knee_series']), label='Matched mocap knee (ref)', lw=1.5, color='green', alpha=0.7)
axes[0].plot(t, np.asarray(gt['pred_knee_series']), label='GT knee (query)', lw=2, color='blue')
axes[0].plot(t, np.asarray(pred['pred_knee_series']), label='Pred knee (+5 noise)', lw=2, color='orange', alpha=0.8)
axes[0].set_ylabel('Knee flexion (deg)')
axes[0].set_title(f'Query: {qf} -> Matched: {matched_file}  |  DTW={dist:.4f}  |  MatchRMSE={match_rmse:.1f}')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(t, np.asarray(gt['com_height_series']), label='GT CoM', lw=2, color='blue')
axes[1].plot(t, np.asarray(pred['com_height_series']), label='Pred CoM', lw=2, color='orange', alpha=0.8)
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
