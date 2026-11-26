# Junior handoff - Intelligent PD denoising (MR-TAE)

Written 26 Nov 2026 after a full pass of `E:\FYP_MASTER` and `E:\FYP`.
Read this file first. Then read `JUNIOR_REBUILD_PROMPT.md` if you are starting your own model.

Time to read: about 15 minutes. Time to re-score one checkpoint on a GPU: about 30-60 minutes. Full retrain of one model: many hours on an RTX 4070.

---

## 1. What this work is

Acoustic-emission partial discharge (AE-PD) denoising for high-voltage insulation.
The model is MR-TAE: a 1D wavelet autoencoder that reconstructs a clean pulse and also marks where the pulse is.

- Student: Oo Jian Hong, 22004678, Universiti Malaya, Electrical Engineering, 2026
- Supervisor named in the long report: Ir. Dr. Wong Jee Keen Raymond
- Submitted thesis: `E:\FYP\OoJianHong_22004678_FYP.pdf` and `.docx` (22 Jun 2026)
- Searchable text of that thesis: `E:\FYP\OoJianHong-22004678-Report-FYP.txt`
- Slides: `E:\FYP\pd.pptx`
- Code and weights: `E:\FYP_MASTER`

---

## 2. Use these facts. Skip the rest.

The disk has three different scoreboards. They measure different things. Do not average them. Do not put them in one table.

### A. Submitted thesis abstract (what was handed in)

Source: the PDF above, and the matching txt extract.

- Claimed SNR improvement: +16.18 dB
- Claimed NCC: 0.5145
- Claimed traditional NCC: 0.220
- BayesShrink SNR improvement in that abstract: +9.68 dB
- This 0.5145 is also the number in `docs\METRIC_RECONCILIATION.md` ("thesis reported NCC")
- I could not find a script on this disk that reprints 0.5145 and +16.18 dB from the current checkpoints
- Treat the PDF as the submitted claim. Treat the JSON below as the number you can recompute

### B. Locked re-benchmark (the number to rebuild from)

Source: `paper\metrics_master.json` generated 30 May 2026.
Index: `paper\README.md` and `paper\research_report.md`.

Protocol:

- Synthetic NCC vs a clean ground-truth window
- 7 SNR levels: -15, -10, -5, 0, +5, +10, +15 dB
- 2000 synthetic windows per SNR (14k total)
- Headline real score is grounded NCC: take a clean Q.Lin window, add composite noise, denoise, score against the clean window
- Checkpoint pick: highest mean synthetic NCC across those 7 levels, tie-break -15 dB, file name `selected_final.pt`

Headline row in that JSON, key `MR-TAE-FULL (NoMix)`:

| Metric | Value |
| --- | --- |
| Mean synthetic NCC | 0.8282 |
| SNR improvement | 9.64 dB |
| Segmentation Dice | 0.9192 |
| Grounded real NCC | 0.4535 |
| Synthetic NCC at -15 dB | 0.378 |

Ablation reference in the same file, `MR-TAE-FULL` (phased curriculum): synthetic NCC 0.7991, grounded real NCC 0.4366.

BayesShrink grounded real NCC used in `paper\research_report.md`: 0.3844. Wiener 0.3641. Savitzky-Golay 0.2915.

Gap you must tell her: `results_nomix\` is not on this disk. The NoMix weights that produced 0.8282 are gone. The JSON remains. `results\` does contain `selected_final.pt` for the phased ablation models (16 files). She can re-score those. She must retrain if she wants NoMix again.

`scripts\run_phaseA_rebench.py` and `scripts\make_paper_figures.py` are named in `paper\README.md` and are not in `scripts\`. Do not hunt for them. The locked numbers are already in the JSON.

### C. Earlier 3-SNR figure pass (May, superseded as the paper table)

Source: `NEW FOCUS\results\method_comparison_figures\method_comparison_summary.json`
Uses `best_final.pt`. Only -15, -5, +5 dB. Mean of those three for MR-TAE-FULL-LossB is 0.6531.

| SNR | LossB NCC |
| --- | --- |
| -15 dB | 0.2465 |
| -5 dB | 0.7596 |
| +5 dB | 0.9532 |

Useful as a picture of low-SNR behaviour. It is not the 7-level locked table.

---

## 3. Scores that look great and are the wrong metric

`docs\RESULTS_COMPARISON.md` shows Real NCC 0.9602 for MR-TAE-FULL and 0.9713 for BayesShrink.

`results\MR-TAE-FULL-LossB\benchmark_results.json` records why:

- `real_qlin.reference` = `input_window`
- Note in the file: no clean ground truth; NCC is versus the observed noisy window
- A model that changes nothing scores high on that number
- The same file's `real_qlin_grounded` block is the honest one (LossB grounded mean NCC about 0.463 on that older run)

`other_results\FINAL_VERDICT_V2.md` ranks models by the input-referenced real NCC (FULL 0.9602). Skip that ranking.

`docs\FINAL_VERDICT.md` is an empty template. Skip it.

`PROJECT_STATUS.md` is an April snapshot that says training is on epoch 1. Training finished after that. Skip it.

`FILES_TO_CHECK.md` and `SESSION_HANDOFF.md` and `RESUME_PROMPT.md` are May agent notes. Skip them.

`README.md` still describes a package at `mr_tae_fusion\` in the repo root and a synthetic hash `ff49e064...`. Both are stale.

- Real package: `Try Model\mr_tae_fusion\`
- Hash that matches `docs\TEST_SET_HASHES.md`: `data\test_synthetic_ae.pt` = `9dd9fb8c913b8b2f0aebac63c700eb832a107a0e00626defb505f2820ddaef5c`
- A second `test_synthetic_ae.pt` used to sit in the repo root. Its hash was `ac9dc039...` (different file). It was removed in this cleanup. Use only `data\`.

`docs\ARCHITECTURE_SUMMARY.md` draws a Haar wavelet. The code default is Daubechies-4. Trust the code.

```text
Try Model\mr_tae_fusion\models\wavelet_layers.py
get_wavelet_filters(wavelet='db4')
```

`README.md` says the Q.Lin set is XLPE cable gaps at 12/15/18/21 mm and 2 MHz. The files on disk are:

```text
Datset from Q.lin\data1.0mm.mat
Datset from Q.lin\data1.8mm.mat
Datset from Q.lin\data2.0mm.mat
Datset from Q.lin\data2.5mm.mat
```

`Try Model\mr_tae_fusion\config.py` sets `sample_rate = 1e6` (1 MHz), `signal_length = 2048` (2.048 ms), carrier 20-500 kHz, decay 200-1200 us. An early bug used 1 GHz and nanosecond decays. That bug is fixed in this config. Keep these numbers.

---

## 4. What to copy into her project

Copy these. Leave the rest.

| Keep | Why |
| --- | --- |
| `Try Model\mr_tae_fusion\` | Model, losses, synthetic PD generator, metrics |
| `config\model_registry.yaml` | The 13 ablation flags, so she can see what was switched off |
| `config\training_config.yaml` | Seed 42, 50k train / 5k val, phases 40/40/30, lr 3e-4, batch 128 |
| `training\` | Curriculum trainer, heat pause, resume |
| `pipeline\mlflow_orchestrator.py` | One training entry point |
| `data\build_test_sets.py` | Rebuilds the frozen `.pt` files |
| `data\*.pt` | Frozen tests. Verify hashes in `docs\TEST_SET_HASHES.md` before any new training |
| `Datset from Q.lin\*.mat` | Raw lab recordings |
| `results\<model>\checkpoints\best.pt` | Training-best weights |
| `results\<model>\checkpoints\best_final.pt` | Fine-tune weights used by the May figure script |
| `results\<model>\checkpoints\selected_final.pt` | Weights chosen for the 30 May table |
| `paper\metrics_master.json` | Locked numbers |
| `paper\research_report.md` | Long method write-up aligned to that JSON |
| `docs\TEST_SET_HASHES.md` | Fingerprints |
| `docs\EXAMINER_QA.md` | Viva answers. Check any number in it against section 2 before she repeats it |
| `E:\FYP\OoJianHong_22004678_FYP.pdf` | Submitted thesis |

`best_demo.pt` is a demo-leakage checkpoint from `NEW FOCUS\finetune_final.py --demo-leakage`. Do not report metrics from it.

---

## 5. How to stand the project back up

Machine that trained this: Windows, Python 3.11, NVIDIA GPU, tested on RTX 4070 12 GB. `.venv` is already in the repo (PyTorch with CUDA). `Try Model\myenv` was a second environment and was deleted.

```powershell
cd E:\FYP_MASTER
.\.venv\Scripts\Activate.ps1
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Confirm the frozen synthetic file before trusting any new number:

```powershell
python -c "import hashlib,pathlib; print(hashlib.sha256(pathlib.Path(r'data\test_synthetic_ae.pt').read_bytes()).hexdigest())"
```

Expected: `9dd9fb8c913b8b2f0aebac63c700eb832a107a0e00626defb505f2820ddaef5c`

Train one model (this resumes from `session_state.json` if a run was interrupted):

```powershell
python pipeline\mlflow_orchestrator.py --model MR-TAE-FULL
```

Train the whole ablation queue:

```powershell
python pipeline\mlflow_orchestrator.py --run-all
```

Dashboard, optional:

```powershell
python app.py
```

Then open `http://localhost:1001`.

Thesis-style pictures from `best_final.pt` (3 SNR levels, not the 7-level table):

```powershell
python "NEW FOCUS\evaluate_and_visualise.py"
```

Smoke test of the package:

```powershell
python tests\smoke_test_all.py
```

If imports fail, the package lives under `Try Model\mr_tae_fusion` and some scripts still contain `D:\New folder (2)\Data\FYP_MASTER`. Point those paths at `E:\FYP_MASTER` before running them. Do not "fix" the metric definition to make Real NCC go back above 0.9.

---

## 6. Architecture that actually runs

Input `(batch, 1, 2048)`.

1. 3-level wavelet encoder. Downsample with db4 DWT, keep both approximation and detail, then Conv1d blocks. Skip tensors saved.
2. Bottleneck: 2-layer BiGRU (hidden 128 each direction) then a 1D Swin block (window 8, 4 heads).
3. Decoder: inverse DWT, attention-gated skips, Conv1d blocks.
4. Two heads: denoised signal `(batch, 1, 2048)` and a per-sample PD mask.

Loss names in `config\model_registry.yaml`:

- A: Charbonnier + focal + Dice, uncertainty weights. This is MR-TAE-FULL.
- B: Charbonnier + Dice. This is MR-TAE-FULL-LossB. `training_config.yaml` is set to B.
- C: Charbonnier + SSIM, no segmentation. This is what `noMTL` uses.

Curriculum in `training_config.yaml`: phase 1 easy SNR, phase 2 mid, phase 3 hard, seed 42.

What the ablations showed, in plain language:

- Wavelet downsampling matters. Removing it collapses the real-domain score.
- Stacking BiGRU + Swin + attention + multi-task loss makes a 3.0M model that is slow (about 10-27 ms per window depending on the benchmark) and hard to debug.
- The smaller MWCNN-BiGRU (about 1.31M params) is the edge candidate in the early tables.
- Loss choice moves the low-SNR number as much as removing a whole block.
- The later NoMix curriculum beat the phased FULL model on the locked synthetic mean (0.828 vs 0.799) and on grounded real (0.453 vs 0.437). Those weights are not on this disk.

---

## 7. Folders after this cleanup

Removed on 26 Nov 2026:

- `results\**\checkpoint_epoch*.pt` and the same pattern under `other_results` (intermediate epochs, about 7.9 GB). `best.pt`, `best_final.pt`, `best_demo.pt`, `selected_final.pt` stay.
- `Try Model\myenv` (about 4.9 GB duplicate environment)
- Project `__pycache__` folders outside `.venv`
- `NEW FOCUS\repo_inventory_full.txt` and `repo_inventory_dirs.txt`
- Repo-root `test_real_qlin.pt` and `test_synthetic_ae.pt` (not the hashed copies; those stay in `data\`)
- `E:\FYP\~$pd.pptx` (Word lock file)
- `E:\FYP\pptx_to_md\.venv` if it was present

Moved, not deleted, under `E:\FYP\_drafts`:

- Older report docx copies, the 150 MB `Slide 1.docx`, the Turnitin-style `66643_*.docx`, and the watermark jpg

Left in `E:\FYP` root on purpose:

- `OoJianHong_22004678_FYP.pdf` and `.docx` (final)
- `pd.pptx`
- `Slide_Deck_Spec.md`
- `OoJianHong-22004678-Report-FYP.txt`
- `MR-TAE_Chapter4_Results.docx`
- `Thesis template 2025 (1).docx`
- `ST-AI-0183_Paper.docm`
- citation `*.txt`
- `pptx_to_md\` source (converter only, not part of the science)

`noisy_minus10dB_18mm.mat` (about 1.4 GB) stays at the repo root. It is not inside `Datset from Q.lin`.

`.venv` stays. Recreate it from `requirements.txt` only if activation fails.

---

## 8. Her job

Two legal paths. Pick one and write it on the first page of her report.

1. Reproduce. Re-score `selected_final.pt` with grounded NCC and synthetic NCC vs clean. Quote `paper\metrics_master.json` only when her script prints the same protocol. Quote the PDF only as the submitted thesis, with the metric definition written next to the number.
2. New model. Use `JUNIOR_REBUILD_PROMPT.md` as the brief. Same data, same hashes, same two headline metrics. New architecture. She must beat grounded real NCC 0.4535 and synthetic NCC 0.8282, or explain why a simpler model is close enough at under 1.5M parameters.
