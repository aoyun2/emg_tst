"""
mocap_evaluation — empirical prosthetic evaluation via motion matching + physics simulation.

Quick start
-----------
Smoke test (no data / checkpoint required):
    python -m mocap_evaluation.run_evaluation --smoke-test

Full evaluation with the complete CMU mocap database:
    python -m mocap_evaluation.run_evaluation \\
        --checkpoint checkpoints/<run>/fold_01/reg_best.pt \\
        --data       samples_dataset.npy \\
        --full-db --categories walk run jump \\
        --n-samples  20

Download CMU mocap BVH files by category:
    python -m mocap_evaluation.cmu_downloader --list
    python -m mocap_evaluation.cmu_downloader -c walk run jump dance
    python -m mocap_evaluation.cmu_downloader            # all categories

Visualise in PyBullet (requires display + pybullet):
    python -m mocap_evaluation.prosthetic_sim --demo
    python -m mocap_evaluation.prosthetic_sim --demo --full-db

Pipeline
--------
1. cmu_catalog      : curated index of CMU mocap subjects, trials, and motion categories
2. cmu_downloader   : batch download BVH files from the CMU database with retry logic
3. bvh_parser       : parse CMU Graphics Lab BVH motion capture files
4. mocap_loader     : load BVH files with per-file category metadata (or synthetic fallback)
5. motion_matching  : category-aware DTW sliding-window search to align IMU signals with mocap
6. prosthetic_sim   : PyBullet simulation; right knee = model prediction, GUI visualisation
7. run_evaluation   : end-to-end pipeline + JSON metrics output

The simulation replaces the RIGHT KNEE joint angle with the model's prediction
while all other joints (left knee, both hips, both ankles, trunk) follow the
matched mocap trajectory.  A fall is declared when the centre-of-mass drops
below 0.55 m; gait symmetry, step count and stability scores are also reported.

Motion categories
-----------------
The CMU catalog covers: walk, run, jump, dance, climb, sport, exercise,
sit_stand, reach_lift, balance, misc — approximately 300 curated trials
across 40+ subjects.
"""
