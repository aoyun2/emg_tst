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
