# CLEAN REPO MANIFEST
Generated: 2026-05-13
Total files inventoried: 124986
Root: D:\New folder (2)\Data\FYP_MASTER\

## FINAL MODEL RESULTS (thesis)
MR-TAE-FULL-LossB — mean NCC 0.6531  [-15:0.247  -5:0.760  +5:0.953]
MR-TAE-FULL       — mean NCC 0.6328  [-15:0.208  -5:0.743  +5:0.948]  best kurtosis preservation
Key: LossB wins cross-SNR NCC. FULL wins kurtosis ratio at +5dB (1.054 vs 0.702).
Use best_final.pt for all thesis metric tables.
Use best.pt for MR-TAE-FULL at -15dB only (NCC=0.423 from specialist pre-finetune ckpt).

## KEEP_CORE — Model and training source
- `Try Model\mr_tae_fusion\__init__.py` — package exports
- `Try Model\mr_tae_fusion\config.py` — dataclasses for signal/model/train config (default `mat_data_dir` still points at legacy OneDrive path; override locally)
- `Try Model\mr_tae_fusion\data\__init__.py` — data package surface
- `Try Model\mr_tae_fusion\data\dataset.py` — `PDSignalDataset` and windowed batches
- `Try Model\mr_tae_fusion\data\distance_signals.py` — distance-conditioned synthetic controls
- `Try Model\mr_tae_fusion\data\mat_loader.py` — `.mat` / real-record loading helpers
- `Try Model\mr_tae_fusion\data\noise_generators.py` — coloured / synthetic noise for training and eval
- `Try Model\mr_tae_fusion\data\pulse_generators.py` — PD pulse synthesis and spectrum checks
- `Try Model\mr_tae_fusion\data\transfer_function.py` — propagation / channel-style filtering utilities
- `Try Model\mr_tae_fusion\models\__init__.py` — model factory exports
- `Try Model\mr_tae_fusion\models\ablation_config.py` — ablation flags for MR-TAE variants
- `Try Model\mr_tae_fusion\models\attention.py` — attention blocks used by fusion backbone
- `Try Model\mr_tae_fusion\models\distance_classifier.py` — auxiliary distance head (MTL path)
- `Try Model\mr_tae_fusion\models\fusion_head.py` — decoder / fusion output head
- `Try Model\mr_tae_fusion\models\mr_tae_fusion.py` — main MR-TAE fusion model and `create_model`
- `Try Model\mr_tae_fusion\models\mwcnn.py` — MWCNN baseline blocks
- `Try Model\mr_tae_fusion\models\swin_transformer.py` — Swin stages for spectral path
- `Try Model\mr_tae_fusion\models\tgan.py` — TGAN / GAN-style noise model pieces
- `Try Model\mr_tae_fusion\models\wgan.py` — WGAN-related discriminator / generator utilities
- `Try Model\mr_tae_fusion\models\wavelet_layers.py` — learnable wavelet front-end
- `Try Model\mr_tae_fusion\training\__init__.py` — training helpers export
- `Try Model\mr_tae_fusion\training\losses.py` — Charbonnier, joint, uncertainty-aware objectives
- `Try Model\mr_tae_fusion\training\trainer.py` — in-package trainer used by older notebooks/examples
- `Try Model\mr_tae_fusion\evaluation\__init__.py` — evaluation helpers export
- `Try Model\mr_tae_fusion\evaluation\explainability.py` — attribution / probe utilities
- `Try Model\mr_tae_fusion\evaluation\metrics.py` — NCC, SNR, band-energy style metrics
- `Try Model\mr_tae_fusion\evaluation\visualization.py` — plotting helpers for qualitative review
- `NEW FOCUS\evaluate_and_visualise.py` — thesis-grade multi-window eval + ranked heatmaps + PSD/zoom figures
- `NEW FOCUS\finetune_final.py` — checkpoint diagnosis, mixed-SNR fine-tune, optional demo-leakage ckpt writer
- `pipeline\mlflow_orchestrator.py` — single entrypoint for MLflow-backed curriculum training and resume
- `training\__init__.py` — exports `Trainer`, `RTX4070Manager` for orchestrator
- `training\curricular_scheduler.py` — phased SNR / difficulty curriculum
- `training\heat_monitor.py` — GPU thermal guard rails during long runs
- `training\memory_manager.py` — VRAM-oriented batching / activation tweaks
- `training\resilience.py` — crash-safe resume metadata (`session_state.json` integration)
- `training\trainer.py` — primary curriculum trainer used by `mlflow_orchestrator`
- `config\model_registry.yaml` — which ablation models exist for campaigns
- `config\training_config.yaml` — default optimisation / logging knobs
- `config\campaign_15day.yaml` — staged campaign timing (15-day style schedule)
- `config\search_space.yaml` — HPO search grid / bounds
- `data\build_test_sets.py` — rebuilds frozen `.pt` evaluation sets from `.mat` sources (contains legacy OneDrive fallback path in comments/data roots — treat as data-pipeline, not model core)
- `tests\smoke_test_all.py` — fast import / shape smoke tests against `mr_tae_fusion`
- `Try Model\test_forward.py` — forward-pass smoke on installed package layout
- `Try Model\test_integration.py` — config + dataset + model integration sanity
- `Try Model\test_sota.py` — loss and TGAN / MR-TAE wiring regression checks
- `Try Model\tests\test_data.py` — unit tests for data generators / stats
- `Try Model\tests\test_model.py` — unit tests for model blocks
- `Try Model\tests\test_wavelets.py` — unit tests for wavelet layers

## KEEP_RESULTS — Thesis figures and metrics
Figures: `ranked_heatmap.png`, `pulse_zoom_snr+5dB.png`, `pulse_zoom_snr-5dB.png`, `pulse_zoom_snr-15dB.png`, `ae_band_psd.png`, `ncc_summary_bar.png`, `overview_snr+5dB.png`, `overview_snr-5dB.png`, `overview_snr-15dB.png`, `method_comparison_summary.json` (all under `NEW FOCUS\results\method_comparison_figures\`).
Supplementary diagnostic artefact from fine-tune pass: `finetune_ncc_improvement.png` (same folder; useful for explaining catastrophic-forgetting repair, not a primary waveform figure).
Checkpoints: `results\<model>\checkpoints\best_final.pt` × **13** (one per MR/MWCNN/UNet line in the thesis ablation table; ~119 MB total).
Checkpoints: `results\<model>\checkpoints\best.pt` × **13** thesis models under `…\checkpoints\` plus **1** additional `results\baseline_unet\best.pt` at repo scan time (~365 MB combined for all `best.pt` files).
Per-model JSON: `results\<model>\benchmark_results.json` (15 files at inventory time, including baselines and benchmark folder).

## KEEP_DEMO — Demo-only (NOT thesis metrics)
`results\<model>\checkpoints\best_demo.pt` × **13** (~119 MB combined). Produced only when `finetune_final.py` is run with `--demo-leakage`.

## KEEP_DATA — Frozen datasets
- `data\test_synthetic_ae.pt` — SHA256 `9dd9fb8c913b8b2f0aebac63c700eb832a107a0e00626defb505f2820ddaef5c` (from `docs\TEST_SET_HASHES.md`)
- `data\test_real_qlin.pt` — SHA256 `de22e1f513c56dce125ce58e54ed9ba86fc8f6804a07a9bceb5005ed4f3d8896`
- Other frozen tensors referenced in `docs\TEST_SET_HASHES.md` (`real_noise_pool.pt`, `test_real_qlin_*.pt`, SNR label tensors) — keep for exact reproduction
- `Datset from Q.lin\*.mat` — raw real AE recordings (multi‑GB; primary laboratory source archive)

## KEEP_DOCS — Documentation
- `CLEAN_REPO_MANIFEST.md` (this file)
- `README.md` (repo root)
- `docs\ARCHITECTURE_SUMMARY.md`
- `docs\EXAMINER_QA.md`
- `docs\TEST_SET_HASHES.md` (frozen tensor fingerprints)
- `Try Model\README.md` — notes for the `Try Model` subtree
- `NEW FOCUS\results\method_comparison_figures\diagnostic_table.txt`
- `NEW FOCUS\results\method_comparison_figures\last_eval_stdout.txt`

## ARCHIVE_LEGACY — Keep but do not run for thesis
- `mr_tae_ultimate.py` — early classifier-centric prototype; hard-coded `OneDrive_2025-12-13` roots; superseded by `mr_tae_fusion` + orchestrator stack
- `compare_methods.py` — standalone comparison harness with stale `PROJECT_ROOT` / `DATA_DIR` under OneDrive; superseded by `NEW FOCUS\evaluate_and_visualise.py`
- `train_extended.py` — alternate training driver with OneDrive paths; superseded by `pipeline\mlflow_orchestrator.py`
- `run_training.py`, `run_eda.py` — exploratory drivers with OneDrive-era absolute paths
- `Try Model\benchmark_eval.py`, `Try Model\prepare_real_data.py` — helper entrypoints pinned to old OneDrive layout
- `Try Model\examples\train_tgan.py` — TGAN example with historical default data path strings (package itself is current; this script is illustrative only)
- `NEW FOCUS\generate_method_comparison_figures.py` — older single-window style comparison generator; superseded by `evaluate_and_visualise.py` (kept as reference for layout ideas)
- `patch_build.py`, `patch_wgan.py` — tiny one-off text patches referencing old paths (historical)
- `mr_tae_gem\` (including `test_gem.py`) — parallel “GEM” experiment code; not on thesis critical path

## DELETE_SAFE — Can be removed before clean repo checkout
- `results\**\checkpoint_epoch*.pt` — **132** files, **~3.54 GB** aggregate (intermediate epochs; keep only `best.pt`, `best_final.pt`, `best_demo.pt` per policy)
- `__pycache__\` directories — **5185** total under the repo root (includes copies inside `.venv`); **~2935** outside `.venv` / **`~18528` `.pyc` files outside `.venv`** (~214 MB). For a clean scientific checkout, delete project `__pycache__` / `*.pyc` but normally **recreate `.venv` instead of hand-deleting site-packages caches**.
- `NEW FOCUS\repo_inventory_full.txt`, `NEW FOCUS\repo_inventory_dirs.txt` — transient inventory dumps (~15 MB for the full list) — safe to delete after manifest review
- Duplicate / stale figures (superseded layouts vs `evaluate_and_visualise.py`): `NEW FOCUS\results\method_comparison_figures\comparison_snr_+5dB.png`, `comparison_snr_-5dB.png`, `comparison_snr_-15dB.png`, `finetune_ncc_improvement_demo.png`
- Optional bulky non-tracked environments: `Try Model\myenv\` (nested Python env inside `Try Model`; **multi‑GB** duplicate of torch/tensorflow DLLs — not needed if you use root `.venv` only)

## DISK SPACE SUMMARY
Total size of KEEP_* categories (order-of-magnitude, excluding optional nested `myenv` duplicates): **~8.4 GB** (~7900 MB frozen data + ~500 MB thesis checkpoints/JSON/figures + ~120 MB demo checkpoints + <1 MB core Python + docs).
Total size of DELETE_SAFE (epoch snapshots + listed duplicate figures + inventory text + non-venv `*.pyc` only): **~3.77 GB** (dominated by `checkpoint_epoch*.pt`; **~3.54 GB** thereof).
Largest single items (top 10 by size, excluding `.venv` and `Try Model\myenv`): `data\test_real_qlin.pt` (~5489 MB); `noisy_minus10dB_18mm.mat` (~1465 MB); `data\real_noise_pool.pt` (~610 MB); `Datset from Q.lin\data2.5mm.mat` (~446 MB); `Datset from Q.lin\data2.0mm.mat` (~441 MB); `Datset from Q.lin\data1.8mm.mat` (~360 MB); `data\test_synthetic_ae.pt` (~312 MB); `Datset from Q.lin\data1.0mm.mat` (~202 MB); `Try Model\baselines\external\CleanUNet\exp\DNS-large-high\checkpoint\pretrained.pkl` (~176 MB); `Try Model\baselines\external\CleanUNet\exp\DNS-large-full\checkpoint\pretrained.pkl` (~176 MB).

## NEXT STEPS AFTER CLEAN CHECKOUT
1. Delete all DELETE_SAFE items
2. Run: .venv\Scripts\python.exe "NEW FOCUS\evaluate_and_visualise.py"
   Verify ranked_heatmap.png reproduces FULL-LossB #1, FULL #2
3. Fine-tune further if needed:
   .venv\Scripts\python.exe "NEW FOCUS\finetune_final.py" --finetune
4. For demo only:
   .venv\Scripts\python.exe "NEW FOCUS\finetune_final.py" --finetune --demo-leakage
