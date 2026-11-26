# Prompt - rebuild this FYP as your own model

Paste everything below the line into a new project chat. Do not paste the old `AGENT_*.md` files.

---

You are helping me build my own final-year project in acoustic-emission partial-discharge denoising. I inherited a working system called MR-TAE from Oo Jian Hong (Universiti Malaya, 2026). I must not submit his thesis text, his figures, or his checkpoints as my own results. I may use his frozen test sets, his raw `.mat` files, and his metric definitions so my numbers are comparable.

## Goal

Design a new 1D denoiser. Working name: pick one and keep it. It must:

1. Map a noisy AE window `(batch, 1, 2048)` to a denoised window of the same shape.
2. Optionally emit a per-sample PD mask. Segmentation is a regulariser, not the headline.
3. Beat or justify against these locked reference numbers from `paper\metrics_master.json` (30 May 2026), model `MR-TAE-FULL (NoMix)`:
   - Mean synthetic NCC vs clean ground truth, 7 SNRs (-15 to +15 dB step 5), 2000 windows each: **0.8282**
   - Mean SNR improvement on that set: **9.64 dB**
   - Dice of the PD mask: **0.9192**
   - Grounded real NCC: **0.4535**
4. Stay trainable on one 12 GB NVIDIA GPU.
5. Prefer under 1.5M parameters if accuracy is within 0.02 synthetic NCC of the reference. The old full model is about 3.0M and 10-27 ms per window. Size was not the win. The wavelet front-end and the loss were.

The NoMix weights are not on disk (`results_nomix\` is missing). Do not pretend to load them. The phased checkpoints are in `E:\FYP_MASTER\results\<model>\checkpoints\selected_final.pt`. Use those only as a baseline to re-score, not as my model.

## Physics you must not "simplify" away

From `Try Model\mr_tae_fusion\config.py`. An early version of this project used 1 GHz and nanosecond pulses. That was wrong. Keep:

- Sample rate 1 MHz
- Window 2048 samples = 2.048 ms
- PD carrier 20-500 kHz
- Decay tens to hundreds of microseconds, not nanoseconds
- Composite noise: white Gaussian, Bernoulli-Gaussian impulses, powerline and harmonics, narrowband interference, magnetostrictive hum around 100/200 Hz
- Wavelet in the old model is Daubechies-4 (`wavelet_layers.py` default `db4`). The architecture markdown that says Haar is wrong.

Raw lab files, gap in the filename:

- `E:\FYP_MASTER\Datset from Q.lin\data1.0mm.mat`
- `data1.8mm.mat`, `data2.0mm.mat`, `data2.5mm.mat`

Ignore any note that says 12/15/18/21 mm or 2 MHz. Those notes do not match these files or this config.

## Data contract

Frozen tensors, hashes in `E:\FYP_MASTER\docs\TEST_SET_HASHES.md`:

- `data\test_synthetic_ae.pt` sha256 `9dd9fb8c913b8b2f0aebac63c700eb832a107a0e00626defb505f2820ddaef5c`
- `data\test_real_qlin.pt` sha256 `de22e1f513c56dce125ce58e54ed9ba86fc8f6804a07a9bceb5005ed4f3d8896`
- `data\real_noise_pool.pt` sha256 `7525cb7a992f6f93c41286bd2c6e66a4aa32b1cde2530d8426ffca39d7acc36a`
- `data\test_real_qlin_clean.pt` sha256 `f04878e4ad33c55838c8b49ccc611b522a0a73ad19a54e263f8224d5f897e482`
- `data\test_real_qlin_noisy.pt` sha256 `6734dd17cac9b0944e9faca3ce233c2e5659994c067663c67e6d913455c9187f`
- `data\test_real_qlin_snr_labels.pt` sha256 `70d0fbc7d6295d0c5b626794813d28fd2e4ded58a774177452fee9a103e6efd6`

Verify the synthetic hash before training. If it does not match, stop.

There is no clean ground truth inside the raw Q.Lin recordings. The honest real test injects known noise onto a clean extracted pulse, then scores NCC against that pulse. Call this grounded real NCC.

## Metric rules

Report only these as headlines:

1. Synthetic NCC vs clean, mean over the 7 SNRs, plus the -15 dB slice.
2. SNR improvement in dB on the same synthetic set.
3. Grounded real NCC.

Do not report NCC of the model output against the noisy input. That number sits near 0.96 even for weak denoisers, because the noisy input already correlates with itself. `docs\RESULTS_COMPARISON.md` uses that definition. Do not cite it.

Do not cite demo checkpoints (`best_demo.pt`). They were built with a leakage flag.

Do not mix these three published numbers in one sentence:

- Thesis PDF abstract: NCC 0.5145, SNR +16.18 dB (submitted claim, metric protocol not recoverable from the scripts on disk)
- 3-SNR May figure pass, LossB, `best_final.pt`: mean NCC 0.6531
- 7-SNR 30 May lock, NoMix: synthetic NCC 0.8282, grounded real 0.4535

My report uses definition 3 only, and says so in the caption.

## What to reuse from the old code

Read, then copy the minimum:

- `Try Model\mr_tae_fusion\data\pulse_generators.py`
- `Try Model\mr_tae_fusion\data\noise_generators.py`
- `Try Model\mr_tae_fusion\data\transfer_function.py`
- `Try Model\mr_tae_fusion\evaluation\metrics.py`
- `data\build_test_sets.py`
- `config\training_config.yaml` for seed, split size, and the 3-phase SNR schedule

You may read `Try Model\mr_tae_fusion\models\mr_tae_fusion.py` to see the old graph. Do not start by subclassing it.

## New architecture rules

Build one model, not 13, until that one model beats the baselines below on grounded real NCC.

Start from this shape, then change one thing at a time:

1. db4 DWT downsampling for 3 levels. Keep both sub-bands. This was the load-bearing idea. Max-pool deletes short PD spikes.
2. A small Conv1d residual block after each DWT. GELU. GroupNorm or BatchNorm, pick one and stay with it.
3. A bottleneck that is either a BiGRU or a short Swin/attention block, not both, until an ablation on the frozen test set shows the second block adds at least 0.02 synthetic NCC.
4. Inverse DWT decoder. One skip path per level. A sigmoid gate on the skip is allowed. It is not required in version 1.
5. Linear output. No tanh, no sigmoid on the waveform. AE pulses are bipolar.
6. Loss version 1: Charbonnier on the waveform plus a light Dice on the mask if the mask head exists. Add focal and uncertainty weights only after version 1 is trained and scored.
7. Same 3-phase SNR curriculum and seed 42 so the comparison is fair. A no-mix curriculum (all SNRs in every batch) is the variant that won the locked table. Try that as version 2, not as a rewrite of the network.

Baselines I must put in the same table, same test files:

- Wavelet BayesShrink
- Wiener
- Savitzky-Golay (expect it to look poor on pulse shape; that result is real, not a bug)
- My new model
- Optional: re-scored `results\MR-TAE-FULL\checkpoints\selected_final.pt` labelled as the inherited baseline, with the citation

## Training rules

- Seed 42
- 50000 train windows, 5000 val, generated from the same generators, or loaded if a frozen train set is added later
- AdamW, lr 3e-4, weight decay 1e-4, grad clip 1.0, batch 128, AMP on
- Save `best.pt` by validation synthetic NCC and `selected_final.pt` by mean NCC across the 7 SNRs
- Write one JSON next to the checkpoint with the six hashes and the three headline metrics
- If the GPU goes above 82 C, pause. Resume below 68 C.

## Failure modes already found - do not rediscover them

- Scoring against the noisy input and calling it real-data NCC.
- Changing sample rate or pulse duration without rewriting the test set and the hashes.
- Training on the test `.pt` files.
- Reporting `best_demo.pt`.
- Claiming the thesis abstract number 0.5145 / +16.18 dB as the output of `metrics_master.json`.
- Adding WGAN augmentation, SHAP, and a web dashboard before the single model beats BayesShrink on grounded real NCC (0.3844 in the long report).

## Deliverables in the new repo

1. `README.md` with the physics constants, the hash check command, the train command, and the score command.
2. `src/` with model, data, loss, train, evaluate. No second copy of the model under another folder.
3. `configs\base.yaml`
4. `results\<run>\metrics.json` matching the three headline metrics.
5. One results page: table vs BayesShrink, Wiener, Savitzky-Golay, and the inherited MR-TAE-FULL number 0.7991 synthetic / 0.4366 grounded, plus my number.
6. A short note on what I removed from MR-TAE and the measured delta.

## First tasks, in order

1. Create the new repo. Copy only the data generators, metrics, and the `data\*.pt` files. Verify the synthetic hash.
2. Implement the small db4 U-Net above with Charbonnier loss and no attention.
3. Train it. Score it on the frozen sets. Write `metrics.json`.
4. Add either BiGRU or Swin, not both. Score again.
5. Only then decide whether a mask head and Dice loss earn their parameters.
6. Stop when the table is filled. Do not build a dashboard.

Ask me before you change the test set, the sample rate, or the metric definition.
