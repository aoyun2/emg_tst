"""
mocap_evaluation — empirical prosthetic evaluation via motion matching + physics simulation.

Quick start
-----------
Smoke test (no data / checkpoint required):
    python -m mocap_evaluation.run_evaluation --smoke-test

Full evaluation (needs a trained model + samples_dataset.npy):
    python -m mocap_evaluation.run_evaluation \\
        --checkpoint checkpoints/<run>/fold_01/reg_best.pt \\
        --data       samples_dataset.npy \\
        --n-samples  20

Pipeline
--------
1. bvh_parser       : parse CMU Graphics Lab BVH motion capture files
2. mocap_loader     : download CMU walking data (or use synthetic norms from Winter 2009)
3. motion_matching  : DTW sliding-window search to align IMU signals with mocap
4. prosthetic_sim   : PyBullet PD-control simulation; right knee = model prediction
5. run_evaluation   : end-to-end pipeline + JSON metrics output

The simulation replaces the RIGHT KNEE joint angle with the model's prediction
while all other joints (left knee, both hips, both ankles, trunk) follow the
matched mocap trajectory.  A fall is declared when the centre-of-mass drops
below 0.55 m; gait symmetry, step count and stability scores are also reported.
"""
