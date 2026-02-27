"""
mocap_evaluation — empirical prosthetic evaluation via motion matching + physics simulation.

Quick start
-----------
Evaluate a trained model checkpoint:
    python -m mocap_evaluation.run_evaluation \
        --checkpoint checkpoints/<run>/fold_01/reg_best.pt \
        --data       samples_dataset.npy

Test sample run (no checkpoint required):
    python -m mocap_evaluation.run_evaluation --test-sample

Download CMU mocap data:
    python -m mocap_evaluation.cmu_downloader

Visualise in MuJoCo (requires display + mujoco):
    python -m mocap_evaluation.prosthetic_sim --demo

Pipeline
--------
1. cmu_catalog / cmu_downloader : curated index + batch download of CMU mocap BVH files
2. bvh_parser        : parse BVH motion capture files
3. mocap_loader      : load CMU BVH source with per-file category metadata
4. motion_matching   : category-aware DTW sliding-window search to align IMU signals with mocap
5. prosthetic_sim    : MuJoCo-first simulation; right knee = model prediction, GUI visualisation
6. run_evaluation    : end-to-end pipeline + JSON metrics output

The simulation replaces the RIGHT KNEE joint angle with the model's prediction
while all other joints (left knee, both hips, both ankles, trunk) follow the
matched mocap trajectory.  A fall is declared when the centre-of-mass drops
below 0.55 m; gait symmetry, step count and stability scores are also reported.

Data organisation
-----------------
CMU mocap BVH files are stored under ``mocap_data/``:

    mocap_data/
        cmu/      — CMU Graphics Lab  (~300 files, free academic)
"""
