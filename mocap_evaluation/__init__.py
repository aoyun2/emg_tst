"""
mocap_evaluation — empirical prosthetic evaluation via motion matching + physics simulation.

Quick start
-----------
Test sample run (external online gait curves, no checkpoint required):
    python -m mocap_evaluation.run_evaluation --test-sample

Full evaluation with the aggregated mocap database:
    python -m mocap_evaluation.run_evaluation \\
        --checkpoint checkpoints/<run>/fold_01/reg_best.pt \\
        --data       samples_dataset.npy \\
        --n-samples  20

Download all mocap datasets (Bandai Namco + CMU + LAFAN1 + SFU):
    python -m mocap_evaluation.download_all

Or individually:
    python -m mocap_evaluation.bandai_namco_downloader
    python -m mocap_evaluation.cmu_downloader
    python -m mocap_evaluation.lafan1_downloader
    python -m mocap_evaluation.sfu_downloader

Visualise in MuJoCo (requires display + mujoco):
    python -m mocap_evaluation.prosthetic_sim --demo

Pipeline
--------
1. bandai_namco_downloader : download Bandai Namco Motiondataset-2 BVH files
2. cmu_catalog / cmu_downloader : curated index + batch download of CMU mocap BVH files
3. lafan1_downloader : download Ubisoft LAFAN1 BVH files
4. sfu_downloader    : download SFU Motion Capture Database BVH files
5. download_all      : download all datasets in one go
6. bvh_parser        : parse BVH motion capture files
7. mocap_loader      : load + aggregate all BVH sources with per-file category metadata
8. motion_matching   : category-aware DTW sliding-window search to align IMU signals with mocap
9. prosthetic_sim    : MuJoCo-first simulation; right knee = model prediction, GUI visualisation
10. run_evaluation   : end-to-end pipeline + JSON metrics output

The simulation replaces the RIGHT KNEE joint angle with the model's prediction
while all other joints (left knee, both hips, both ankles, trunk) follow the
matched mocap trajectory.  A fall is declared when the centre-of-mass drops
below 0.55 m; gait symmetry, step count and stability scores are also reported.

Data organisation
-----------------
All mocap BVH files are stored in per-dataset sub-folders under ``mocap_data/``:

    mocap_data/
        bandai/   — Bandai Namco Motiondataset-2  (~2,900 files, CC BY-NC 4.0)
        cmu/      — CMU Graphics Lab              (~300 files, free academic)
        lafan1/   — Ubisoft LAFAN1                 (~135 files, CC BY-NC-ND 4.0)
        sfu/      — SFU Motion Capture             (~38 files, free academic)
"""
