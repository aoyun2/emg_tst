# EMG-TST (samples-only, no flags)

This repo trains an encoder-only Time Series Transformer (TST) with stateful (Markov) masked-reconstruction pretraining, then fine-tunes for knee-angle regression.

## Workflow (no CLI flags)

1) Record new data with `uMyo_python_tools/rigtest.py` (it writes `data0.npy`, `data1.npy`, ...).

2) Build mutually-exclusive fixed-length samples (1 second windows @ 100 Hz):
```bash
python split_to_samples.py
```
This writes `samples_dataset.npy` (now includes `y_seq` for full-sequence supervision).

3) Train with 5-fold sample-level cross-validation:
```bash
python -m emg_tst.run_experiment
```
Checkpoints are saved under `checkpoints/tst_YYYYMMDD_HHMMSS/fold_XX/`.

4) Visualize a few sample predictions:
Edit `emg_tst/visualize.py` to point `CKPT_PATH` to a `reg_best.pt`, then run:
```bash
python -m emg_tst.visualize
```

## Notes

- There is **no sliding-window training path** anymore; training uses only `samples_dataset.npy`.
- "Hold out one file" used to mean testing on an entire recording you never trained on. We removed that; this code does **sample-level KFold** (samples are disjoint, non-overlapping 1s windows).
- **Full-sequence supervision**: the model is trained to predict the knee angle at every timestep in the window, not just the last timestep. Eval reports both last-step RMSE and full-sequence RMSE.
- **Pretraining**: 40 epochs of masked reconstruction (Markov-chain masking).
- **LR scheduling**: cosine annealing during fine-tuning (decays to 1% of initial LR).

## Mocap motion-matching evaluator (with mock curves)

If you do not yet have recordings from `rigtest.py`, run the evaluator with generated mock
`knee_label` and `thigh_angle` curves:

Before running evaluation, download the recommended mocap source (Bandai Namco Motiondataset):

```bash
python -m mocap_evaluation.bandai_namco_downloader --dest mocap_data
```

```bash
python -m mocap_evaluation.run_evaluation \
  --mock-data \
  --mock-seconds 6 \
  --top-k 5 \
  --full-db \
  --sim-backend mujoco \
  --out eval_mock_results.json \
  --save-mock mock_curves.npz
```

Notes:
- Matching uses your thigh angle + knee label curves, then runs **K simulations** on the top-K mocap matches.
- A "correct/reference" simulation is always run with the right knee replaced by the label curve.
- A second simulation replaces only the right knee with the model prediction curve.
- Angle convention is standardized across recorded data, BVH parsing, matching, and evaluation: included-angle `[0, 180]` with `180 = straight`.


### Aggregate Bandai + CMU datasets and visualise a match

To combine both sources into one searchable mocap database, keep files in:
- `mocap_data/bandai/`
- `mocap_data/cmu/`

Then run:

```bash
python -m mocap_evaluation.visualize_match \
  --aggregate-datasets \
  --mocap-dir mocap_data \
  --seconds 6 \
  --out artifacts/mock_vs_match.png
```

This generates a mock knee/thigh segment, runs motion matching, and saves a plot of
query-vs-match curves.
