"""Quick simulation using already-cached mocap data (no download)."""
import numpy as np
from mocap_evaluation.mocap_loader import load_aggregated_database
from mocap_evaluation.prosthetic_sim import simulate_prosthetic_walking

# Load CACHED database only (no download)
db = load_aggregated_database(mocap_root='mocap_data', try_download=False, use_cache=True)
bounds = db.get("file_boundaries", [])
fps = db["fps"]
seg_len = int(5.0 * fps)

# Find a walk segment
for entry in bounds:
    start, end, fname, cat = entry
    if cat == "walk" and (end - start) >= seg_len:
        walk_start = start
        print(f'Walk file: {fname} ({(end-start)/fps:.1f}s)')
        break

# Build segment dict
keys = ['knee_right','knee_left','hip_right','hip_left',
        'ankle_right','ankle_left','pelvis_tilt','trunk_lean','root_pos']
segment = {k: db[k][walk_start:walk_start+seg_len].astype(np.float32) for k in keys}
segment['category'] = 'walk'

knee_gt = segment['knee_right'].copy()
thigh_gt = segment['hip_right'].copy()

# Noisy prediction (~5 deg RMSE)
rng = np.random.default_rng(42)
pred_knee = knee_gt + rng.normal(0, 5.0, size=knee_gt.shape).astype(np.float32)

print(f'{seg_len} frames @ {fps} Hz = {seg_len/fps:.1f}s')
print(f'Knee range: {knee_gt.min():.1f} - {knee_gt.max():.1f} deg')

def show(label, m):
    print(f'\n{label}:')
    print(f'  fall_detected: {m["fall_detected"]}, fall_frame: {m["fall_frame"]}')
    print(f'  steps: {m["step_count"]}, gait_symmetry: {m["gait_symmetry"]:.3f}')
    print(f'  stability_score: {m["stability_score"]:.3f}')
    print(f'  com_height: {m["com_height_mean"]:.3f} +/- {m["com_height_std"]:.3f} m')
    print(f'  knee RMSE: {m["knee_rmse_deg"]:.2f} deg, MAE: {m["knee_mae_deg"]:.2f} deg')

# GT simulation
print('\n--- Ground-truth knee ---')
gt = simulate_prosthetic_walking(segment, knee_gt, use_gui=False,
                                  sample_thigh_right=thigh_gt,
                                  save_trajectory='gt_traj.npz')
show('GT result', gt)

# Prediction simulation
print('\n--- Prediction knee (5 deg noise) ---')
pred = simulate_prosthetic_walking(segment, pred_knee, use_gui=False,
                                    sample_thigh_right=thigh_gt,
                                    save_trajectory='pred_traj.npz')
show('Pred result', pred)

rmse = float(np.sqrt(np.mean((pred_knee - knee_gt)**2)))
print(f'\n--- Summary ---')
print(f'Pred vs GT RMSE: {rmse:.2f} deg')
print(f'GT stability:   {gt["stability_score"]:.3f}  |  Pred stability: {pred["stability_score"]:.3f}')
print(f'GT fell: {gt["fall_detected"]}  |  Pred fell: {pred["fall_detected"]}')
print(f'GT steps: {gt["step_count"]}  |  Pred steps: {pred["step_count"]}')

# Generate comparison plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
t = np.arange(seg_len) / fps

# Knee angles
axes[0].plot(t, np.asarray(gt['pred_knee_series']), label='GT knee (flexion)', lw=2)
axes[0].plot(t, np.asarray(pred['pred_knee_series']), label='Pred knee (flexion)', lw=2, alpha=0.8)
axes[0].set_ylabel('Knee flexion (deg)')
axes[0].set_title('Knee angle: Ground Truth vs Prediction (5 deg noise)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# CoM height
axes[1].plot(t, np.asarray(gt['com_height_series']), label='GT CoM', lw=2)
axes[1].plot(t, np.asarray(pred['com_height_series']), label='Pred CoM', lw=2, alpha=0.8)
axes[1].axhline(0.55, color='r', ls='--', label='Fall threshold')
axes[1].set_ylabel('CoM height (m)')
axes[1].set_title(f'Center of Mass Height  |  GT stability={gt["stability_score"]:.3f}, Pred={pred["stability_score"]:.3f}')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Foot contacts
for i, (label, m) in enumerate([('GT', gt), ('Pred', pred)]):
    rc = np.asarray(m['right_contact_frames']) / fps
    lc = np.asarray(m['left_contact_frames']) / fps
    y = 1 - i*0.3
    axes[2].scatter(rc, np.full_like(rc, y), marker='|', s=100, label=f'{label} R foot')
    axes[2].scatter(lc, np.full_like(lc, y+0.1), marker='|', s=100, label=f'{label} L foot')

axes[2].set_ylabel('Contacts')
axes[2].set_xlabel('Time (s)')
axes[2].set_title(f'Foot contacts  |  GT steps={gt["step_count"]}, Pred steps={pred["step_count"]}')
axes[2].legend(loc='upper right', fontsize=8)
axes[2].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig('sim_comparison.png', dpi=150)
plt.close(fig)
print(f'\nPlot saved: sim_comparison.png')
