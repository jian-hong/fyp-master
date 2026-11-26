# FORENSIC AUDIT: MR-TAE Full Re-Evaluation Plan
## AE-PD Denoising Project — Oo Jian Hong 22004678
### Prepared for Cursor AI Agent + Human Review
**Date:** May 2026 | **Claude:** Sonnet 4.6

---

## ⚠️ EXECUTIVE SUMMARY — CRITICAL BUGS FOUND

After tracing every file, the following problems are confirmed or strongly suspected.
**Do not submit any results until all Priority 1 items are resolved.**

| Priority | Issue | Impact |
|----------|-------|--------|
| 🔴 P1 | Two codebases coexist; OLD code still referenced by training files | ALL results may be from wrong pipeline |
| 🔴 P1 | Savitzky-Golay NCC=0.027 on synthetic — physically impossible | Baseline evaluation is broken |
| 🔴 P1 | Latency: noSwin-noBiGRU (1.01M) takes 13.76ms > FULL (3.05M) at 12.55ms | Latency measurement is wrong |
| 🔴 P1 | Real NCC uses input window as reference, not clean GT | All real NCC comparisons are NCC-to-noisy, not NCC-to-clean |
| 🔴 P1 | Curriculum early stopping may fire before Phase 3 | Models never trained on extreme noise (−15 dB) |
| 🟡 P2 | LossA vs LossB attention gate status — must be verified from YAML | May affect architecture interpretation |
| 🟡 P2 | Q.Lin gap distances: README says 12/15/18/21 mm, table says 1.0/1.8/2.0/2.5 mm | Dataset description is inconsistent |
| 🟡 P2 | Window size: old code uses 2001, new code uses 2048; which test data was used? | Test set may be mismatched |
| 🟡 P2 | BayesShrink similar to MR-TAE-FULL — only possible if NCC ref is noisy input | Confirms P1 real-NCC bug |
| 🟢 P3 | NCC formula — confirm correct implementation across all scripts | |
| 🟢 P3 | 13 models: confirm all were trained to Phase 3 completion | |

---

## SECTION 1: THE TWO-CODEBASE PROBLEM

### What exists

**OLD codebase** (dead, but still referenced):
```
D:\New folder (2)\Data\OneDrive_2025-12-13\Try Model\
  mr_tae_fusion/
    config.py           → signal_length = 2001 (NOT 2048)
    data/distance_signals.py → sample_rate = 1000e6 (GHz, NOT 1 MHz)
    data/pulse_generators.py → fc = 25-150 MHz (NOT kHz)
```

**NEW codebase** (active, corrected physics):
```
D:\New folder (2)\Data\FYP_MASTER\
  mr_tae_fusion/        → corrected: 2048 samples, 1 MHz fs, kHz carriers
  config/
    model_registry.yaml → 13 ablation variants defined here
    training_config.yaml
  pipeline/
    mlflow_orchestrator.py
  training/
    trainer.py
    curricular_scheduler.py
    resilience.py
```

**The problem:** The files `compare_methods.py`, `train_extended.py`, `mr_tae_ultimate.py`
at the project root ALL have hardcoded:
```python
PROJECT_ROOT = Path(r'D:\New folder (2)\Data\OneDrive_2025-12-13\Try Model')
DATA_DIR     = Path(r'D:\New folder (2)\Data\OneDrive_2025-12-13')
```
These point to the OLD codebase. **If any evaluation was run from these scripts,
it used 2001-sample windows, GHz sampling, and MHz-carrier synthetic data** — completely
wrong physics.

The terminal output (benchmark run) shows models loading from `NEW FOCUS\results\`
inside FYP_MASTER — so the 13-model training appears correct. But the baseline
evaluation (BayesShrink, Savitzky-Golay, Wiener) may have used the old pipeline.

### Cursor Task 1.1 — Verify which pipeline produced which results
```python
# Run this inside FYP_MASTER:
import pathlib, json
for p in pathlib.Path("results").rglob("benchmark_results.json"):
    d = json.loads(p.read_text())
    print(p, "→ windows evaluated:", d.get("real_qlin", {}).get("n_windows_evaluated"))
    print("  signal_length in synthetic config:", d.get("config", {}).get("signal_length", "NOT FOUND"))
```
If signal_length is missing or shows 2001, those results are from the wrong pipeline.

---

## SECTION 2: ATTENTION GATES — LossA vs LossB vs All Variants

### Direct answer from architecture data

| Variant | use_attention_gates | Evidence |
|---------|-------------------|---------|
| MR-TAE-FULL (LossA) | **YES** ✓ | ARCHITECTURE_SUMMARY.md: A=✓ |
| MR-TAE-FULL-LossB | **YES** ✓ | ARCHITECTURE_SUMMARY.md: A=✓; same arch as FULL, only loss differs |
| MR-TAE-FULL-LossC | **YES** ✓ | ARCHITECTURE_SUMMARY.md: A=✓ |
| MR-TAE-noAttn | **NO** ✗ | Explicitly ablated |
| MR-TAE-noSwin | YES ✓ | |
| MR-TAE-noBiGRU | YES ✓ | |
| MR-TAE-noMTL | YES ✓ | |
| MR-TAE-noWavelet | YES ✓ | |
| MR-TAE-noSwin-noBiGRU | YES ✓ | |
| MR-TAE-noSwin-noAttn | **NO** ✗ | Explicitly ablated |
| MR-TAE-noBiGRU-noAttn | **NO** ✗ | Explicitly ablated |
| MWCNN-BiGRU | **NO** ✗ | Cross-arch baseline, no attn gates (A=✗ in table) |
| UNet-BiGRU-Swin | **NO** ✗ | Cross-arch baseline (A=✗ in table) |

**LossA and LossB both have attention gates.** The only difference between them is the
loss function. The 2.27× claim must not be attributed to attention gate differences.

### Cursor Task 2.1 — Verify YAML registry matches the table above
```bash
cat config/model_registry.yaml | python -c "
import yaml, sys
reg = yaml.safe_load(sys.stdin)['models']
for m in reg:
    print(m['id'], '| attn:', m.get('use_attention_gates'), '| loss:', m.get('loss_config'))
"
```
Any mismatch with the table above is a bug. Fix before retraining.

---

## SECTION 3: SAVITZKY-GOLAY NCC=0.027 — CONFIRMED BUG

### Why it is impossible
Savitzky-Golay is a polynomial least-squares smoothing filter. On a clean signal at +5 dB
SNR (high quality), it acts as a gentle low-pass. The NCC between SG-filtered output and
clean ground truth must be at least 0.7–0.9 at +5 dB. At −5 dB, at least 0.3–0.6.
A mean NCC of 0.027 across all SNR levels means SG output is near-orthogonal to the
clean signal — this is only possible if:

**Cause A (most likely):** The test data fed to the SG baseline is in a different
format than expected — e.g., the signal is not 1D but still has the channel dimension
`(1, 2048)` and SG is applied along the wrong axis, producing garbage.

**Cause B:** The window_length parameter of SG is larger than the signal (e.g.,
window_length=4096 on a 2048-sample signal) — scipy would throw an error or clip.

**Cause C:** NCC is computed after SG with a shifted reference, introducing phase offset.

**Cause D:** The baseline is accidentally loading the wrong checkpoint (a partially
initialized model) instead of running the classical filter.

### Cursor Task 3.1 — Isolate and fix the SG baseline
```python
# evaluation/debug_baselines.py
import numpy as np
from scipy.signal import savgol_filter
import torch

# Load ONE synthetic test window
test = torch.load("data/test_synthetic_ae.pt")
# Inspect shape
x_noisy = test['noisy'][0]   # Should be (1, 2048) or (2048,)
x_clean = test['clean'][0]
print("noisy shape:", x_noisy.shape)
print("clean shape:", x_clean.shape)

# Force to 1D
if x_noisy.ndim > 1:
    x_noisy_1d = x_noisy.squeeze().numpy()
    x_clean_1d = x_clean.squeeze().numpy()
else:
    x_noisy_1d = x_noisy.numpy()
    x_clean_1d = x_clean.numpy()

# Apply SG with safe parameters
sg_output = savgol_filter(x_noisy_1d, window_length=51, polyorder=3)

# NCC
def ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    denom = np.sqrt((a**2).sum() * (b**2).sum())
    return (a * b).sum() / denom if denom > 1e-10 else 0.0

print("NCC(SG, clean):", ncc(sg_output, x_clean_1d))
print("NCC(noisy, clean):", ncc(x_noisy_1d, x_clean_1d))  # This is the input NCC baseline
```
Expected: SG NCC should be > input NCC at moderate-to-high SNR and ≤ input NCC at
very low SNR. If SG NCC < 0.1 at 0 dB, the evaluation function is broken.

### The NCC formula — verify it is correct everywhere
```python
# CORRECT NCC (zero-mean normalized):
def ncc_correct(pred, target):
    pred   = pred   - pred.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    num    = (pred * target).sum(dim=-1)
    denom  = torch.sqrt((pred**2).sum(dim=-1) * (target**2).sum(dim=-1)).clamp(min=1e-8)
    return (num / denom).mean()

# WRONG patterns that produce ~0 NCC:
# 1. Forgetting to subtract mean → biased by DC offset
# 2. Using torch.corrcoef (computes correlation matrix, wrong shape handling)
# 3. Using scipy.signal.correlate without peak extraction (returns full cross-correlation array)
# 4. Using MSE formula accidentally (returns a scalar, not NCC)
```

### Cursor Task 3.2 — Audit every NCC call in the codebase
```bash
grep -rn "ncc\|NCC\|cross_corr\|corrcoef\|correlate" \
    evaluation/ mr_tae_fusion/evaluation/ training/ \
    --include="*.py" | grep -v ".pyc"
```
For each occurrence, verify it matches the CORRECT formula above.

---

## SECTION 4: BAYESSHRINK SIMILAR TO MR-TAE-FULL — DIAGNOSIS

### Why this reveals the real-NCC measurement is wrong

BayesShrink is a soft-threshold wavelet denoiser with data-driven threshold selection.
On synthetic data (GT available), at +5 dB SNR, BayesShrink typically achieves NCC
~0.75–0.90 against clean GT. MR-TAE-FULL achieves ~0.95 against GT. These are NOT similar.

**The only way they appear similar** is if both are measured against the **noisy input**
instead of the **clean ground truth**:
- Any filter that produces an output ≈ input will score NCC ≈ 1.0 against noisy input
- BayesShrink produces near-identity output at high SNR (when threshold is near zero)
- → BayesShrink input-ref NCC ≈ 0.96, MR-TAE-FULL input-ref NCC ≈ 0.96
- → They look the same even though MR-TAE-FULL is genuinely better on clean GT

This confirms: the "Real NCC" column in table2_ablation.csv (values ~0.96 for FULL, ~0.87
for LossB) **measures similarity to the noisy input**, not to any ground truth.
This is appropriate for the real Q.Lin dataset (no GT available), but it means:
- High "Real NCC" = model output is similar to noisy input (could be poor denoising)
- Low "Real NCC" = model output differs from noisy input (could be better denoising)

**MR-TAE-FULL-LossB's lower Real NCC (0.87 vs 0.96) may actually indicate BETTER
denoising** — it removes more noise and thus diverges more from the noisy reference.
LossA's 0.96 could mean it is insufficiently denoising (mostly passing through the input).

### Cursor Task 4.1 — Separate the two NCC interpretations clearly

In the benchmark runner, compute BOTH:
```python
# For SYNTHETIC data (GT available):
ncc_vs_clean = ncc(denoised, clean_gt)      # "Quality NCC" — higher = better denoising

# For REAL data (no GT):
ncc_vs_noisy = ncc(denoised, noisy_input)   # "Fidelity NCC" — indicates how much was changed
ncc_vs_noisy_LABEL = "Input-Referenced NCC (no GT)"

# Also compute for real data:
snr_improvement = snr_out - snr_in  # Can still compute output SNR as ||output||/||output-input||
# But note: this penalises both denoising AND distortion equally
```

Label them EXPLICITLY in all tables and figures. Never mix them in the same column.

---

## SECTION 5: INFERENCE LATENCY — MEASUREMENT BUG

### The smoking gun
```
noSwin-noBiGRU  1.01M params  13.76ms   ← Slower than FULL (3.05M, 12.55ms)?
noSwin          1.41M params  13.61ms   ← Also slower than FULL?
noWavelet       2.99M params  20.94ms   ← Slowest of all?
```
Physics: latency should roughly scale with FLOPs, which roughly scale with params for
similar architectures. A 1.01M model cannot be slower than a 3.05M model unless:

**Cause A (most likely — missing CUDA sync):** GPU operations are asynchronous. Without
`torch.cuda.synchronize()` before and after, the timer measures host-side dispatch
time, not actual GPU compute. Smaller models dispatch faster but wait less before
sync → appear to have variable, unreliable timings.

**Cause B:** Wavelet operations (DWT/IDWT via pywt) are CPU-side. noWavelet (which uses
strided convolutions instead) avoids this CPU↔GPU transfer. The HIGH latency of noWavelet
(20.94ms) vs the LOW latency of MWCNN variants may reveal that pywt is NOT running on GPU.
If so, all MWCNN models have a CPU bottleneck in DWT that is being counted.

**Cause C:** The timer includes data loading (`.to(device)` call inside the timed loop).

### Correct latency measurement template
```python
import torch, time

def measure_latency(model, input_shape=(1, 1, 2048), n_warmup=50, n_runs=200, device='cuda'):
    model.eval()
    x = torch.randn(*input_shape, device=device)
    
    # Warmup — fills CUDA kernels, avoids first-run overhead
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model(x)
    torch.cuda.synchronize()
    
    # Timed runs
    t0 = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            _ = model(x)
    torch.cuda.synchronize()   # CRITICAL: wait for all GPU ops to finish
    t1 = time.perf_counter()
    
    return (t1 - t0) / n_runs * 1000  # ms per inference

# Also check: are DWT operations running on GPU or CPU?
# pywt (numpy-based) always runs on CPU!
# Use pytorch_wavelets or custom DWT layers that stay on GPU
```

### Cursor Task 5.1 — Audit DWT implementation
```bash
grep -rn "import pywt\|pywavelets\|wavedec\|waverec" \
    mr_tae_fusion/models/ mr_tae_fusion/data/ --include="*.py"
```
If `pywt` is imported in any model forward pass, it is CPU-side and will cause
GPU↔CPU transfer on every forward pass. This must be replaced with a GPU-native
DWT (e.g., `pytorch_wavelets`, `torch-dct`, or a custom haar DWT using `F.conv1d`).

### Cursor Task 5.2 — Re-run latency with correct measurement
```bash
python evaluation/remeasure_latency.py \
    --models config/model_registry.yaml \
    --checkpoint-root results/ \
    --n-warmup 50 --n-runs 200 \
    --output results/latency_corrected.json
```

---

## SECTION 6: CURRICULUM TRAINING — EARLY STOPPING BUG

### The concern
The ARCHITECTURE_SUMMARY.md describes early stopping as:
> "Dynamic threshold based on rolling NCC improvement rate"

If this fires during Phase 2 (e.g., at epoch 60), the model **never sees Phase 3
data (SNR −20 to −5 dB)**. This would explain:
- Models that score 0.7+ NCC at −5 dB but 0.05–0.15 NCC at −15 dB
- The BiGRU "collapse" at −15 dB may not be BiGRU's fault — it may be that ALL
  models (with or without BiGRU) barely trained on −15 dB

### How to verify
```bash
# Check which epoch each model's checkpoint was saved from MLflow
python -c "
import mlflow
client = mlflow.tracking.MlflowClient()
for exp in client.search_experiments():
    for run in client.search_runs(exp.experiment_id):
        name = run.data.tags.get('mlflow.runName', run.info.run_id)
        epoch = run.data.metrics.get('epoch', 'N/A')
        phase = run.data.metrics.get('curriculum_phase', 'N/A')
        print(f'{name}: final_epoch={epoch}, final_phase={phase}')
"
```
If any model shows final_phase < 3 or epoch < 85, early stopping fired too soon.

### The correct early-stopping design for curriculum training
```python
class CurriculumEarlyStopping:
    """
    Early stopping that:
    1. CANNOT fire within Phase 1 or Phase 2 (models must always complete all 3 phases)
    2. Can only fire after Phase 3 has run for at least MIN_PHASE3_EPOCHS epochs
    3. Uses NCC plateau detection, not loss
    """
    MIN_PHASE3_EPOCHS = 15  # Must see at least 15 epochs of Phase 3
    PATIENCE = 10  # Plateau patience within Phase 3
    
    def __init__(self):
        self.phase3_epoch_count = 0
        self.best_ncc = -1.0
        self.plateau_count = 0
    
    def step(self, current_phase: int, val_ncc: float) -> bool:
        if current_phase < 3:
            return False  # NEVER stop before Phase 3
        self.phase3_epoch_count += 1
        if self.phase3_epoch_count < self.MIN_PHASE3_EPOCHS:
            return False  # Wait for minimum Phase 3 exposure
        if val_ncc > self.best_ncc + 1e-4:
            self.best_ncc = val_ncc
            self.plateau_count = 0
        else:
            self.plateau_count += 1
        return self.plateau_count >= self.PATIENCE
```

### Cursor Task 6.1 — Check current early stopping implementation
```bash
grep -n "early_stop\|EarlyStopping\|patience\|plateau" \
    training/resilience.py training/trainer.py --include="*.py"
```
Verify it CANNOT fire before Phase 3 has run at minimum 15 epochs.

### Cursor Task 6.2 — Verify all 13 models completed Phase 3
```bash
for model_dir in results/*/; do
    checkpoint=$(ls $model_dir/checkpoints/best.pt 2>/dev/null || echo "MISSING")
    last_log=$(tail -5 $model_dir/training.log 2>/dev/null | grep "Phase")
    echo "$model_dir: $checkpoint | $last_log"
done
```

---

## SECTION 7: WHAT MODELS WERE ACTUALLY TRAINED — FULL AUDIT

### Evidence from codebase

| Source | Claim | Confidence |
|--------|-------|-----------|
| PROJECT_STATUS.md | MR-TAE-noSwin epoch 1 running (Phase 1) | Low — this was status at a past point |
| MLOPS_FORWARD_PLAN.md Phase A | 6 models planned (not 13) | Medium — earlier plan |
| README.md | 13 variants from model_registry.yaml | High — latest design |
| Terminal output (doc 3) | 13/13 models loaded successfully | High — actual run |
| benchmark_results.json | Only MR-TAE-noSwin-noBiGRU has data | Medium — one JSON file |
| RESULTS_COMPARISON.md | Only MR-TAE-noSwin has any data | Medium |
| table2_ablation.csv | All 13 models have benchmark data | High — but source unknown |

**The terminal output (doc 3) is the most reliable evidence: it shows `[ok] MR-TAE-FULL 3.05M`,
etc., loading all 13 checkpoints. Training was completed.**

However, whether they were trained correctly (correct physics, full curriculum, no early
stopping in Phase 2) is unknown without reading the MLflow logs.

### Cursor Task 7.1 — Full model inventory check
```python
# model_audit.py — run from FYP_MASTER root
import pathlib, json, torch

models_expected = [
    "MR-TAE-FULL", "MR-TAE-noBiGRU", "MR-TAE-noSwin", "MR-TAE-noAttn",
    "MR-TAE-noMTL", "MR-TAE-noWavelet", "MR-TAE-noSwin-noBiGRU",
    "MR-TAE-noSwin-noAttn", "MR-TAE-noBiGRU-noAttn",
    "MR-TAE-FULL-LossB", "MR-TAE-FULL-LossC",
    "MWCNN-BiGRU", "UNet-BiGRU-Swin"
]

for mid in models_expected:
    ckpt_dir = pathlib.Path(f"results/{mid}/checkpoints")
    best     = ckpt_dir / "best.pt"
    last     = ckpt_dir / "last.pt"
    
    status = "MISSING"
    n_params = "?"
    epoch_saved = "?"
    
    for ckpt_path in [best, last]:
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu")
            epoch_saved = ckpt.get("epoch", "no epoch key")
            model_state = ckpt.get("model_state", ckpt.get("model_state_dict", None))
            if model_state:
                n_params = f"{sum(p.numel() for p in [v for v in model_state.values() if hasattr(v, 'numel')]):,}"
            phase = ckpt.get("phase", ckpt.get("curriculum_phase", "?"))
            status = f"FOUND @ {ckpt_path.name} | epoch={epoch_saved} | phase={phase}"
            break
    
    print(f"{mid:35s}: {status}")
```

---

## SECTION 8: COMPLETE RE-EVALUATION PIPELINE

Once the bugs above are fixed, run the following definitive evaluation:

### Step 1 — Verify test set integrity
```bash
python -c "
import hashlib, pathlib
for name in ['data/test_synthetic_ae.pt', 'data/test_real_qlin.pt']:
    h = hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()
    print(name, '→', h)
"
```
Expected hashes (from README):
- test_synthetic_ae.pt: `ff49e064d1e4ad7b1d2f44ac5967b3a4e6b4407828b7aeddc19bd9bb17ebff9b`
- test_real_qlin.pt: `de22e1f513c56dce125ce58e54ed9ba86fc8f6804a07a9bceb5005ed4f3d8896`

If hashes don't match → test data was modified or wrong file. Rebuild from raw .mat files.

### Step 2 — Verify Q.Lin gap distances
```bash
python -c "
import scipy.io as sio, pathlib
for f in ['data1.0mm.mat', 'data1.8mm.mat', 'data2.0mm.mat', 'data2.5mm.mat']:
    p = pathlib.Path('data/real_qlin') / f
    if p.exists():
        mat = sio.loadmat(str(p))
        keys = [k for k in mat.keys() if not k.startswith('_')]
        for k in keys:
            print(f'{f}: key={k}, shape={mat[k].shape}')
"
```
The README says gap distances are 12/15/18/21 mm. The table says 1.0/1.8/2.0/2.5 mm.
The actual .mat filename contains the answer. Report which is correct in thesis.

### Step 3 — Run the definitive benchmark
```bash
python evaluation/benchmark_runner.py \
    --compare-all \
    --test-synthetic data/test_synthetic_ae.pt \
    --test-real data/test_real_qlin.pt \
    --snr-levels -15 -10 -5 0 5 \
    --include-baselines bayesshrink savitzky_golay wiener \
    --output-dir results/definitive_benchmark/ \
    --latency-warmup 50 \
    --latency-runs 200 \
    --use-cuda-sync true
```

---

## SECTION 9: CURSOR AI AGENT — EXECUTABLE TASK LIST

Copy this section verbatim as your Cursor AI agent instruction.

---

```
# CURSOR AI AGENT: MR-TAE Full Forensic Re-Evaluation
# Project: AE-PD Denoising | Student: Oo Jian Hong 22004678
# Root: D:\New folder (2)\Data\FYP_MASTER
# Venv: D:\New folder (2)\Data\FYP_MASTER\.venv\Scripts\python.exe

## CONTEXT
You are auditing and re-running a 13-model AE partial discharge denoising benchmark.
Several results are known or suspected to be wrong. Your job is to fix the pipeline
and produce clean, trustworthy numbers. Work methodically, one task at a time.
Always activate the .venv before running Python scripts.

## ACTIVATION
$env:PYTHONPATH = "D:\New folder (2)\Data\FYP_MASTER"
& "D:\New folder (2)\Data\FYP_MASTER\.venv\Scripts\Activate.ps1"

---

## TASK 1 — MODEL INVENTORY (Do First, Don't Skip)

Create: `results/MODEL_INVENTORY.txt`

For each of the 13 expected models, check:
  a) Does results/{MODEL_ID}/checkpoints/best.pt exist?
  b) What epoch was it saved at? (key: "epoch" in checkpoint dict)
  c) What curriculum phase? (key: "phase" or "curriculum_phase")
  d) How many parameters? (count model_state keys)
  e) Does the checkpoint key match "model_state" (correct) or "model_state_dict" (old)?

Print and save a table with all this information.

Expected models:
  MR-TAE-FULL, MR-TAE-noBiGRU, MR-TAE-noSwin, MR-TAE-noAttn,
  MR-TAE-noMTL, MR-TAE-noWavelet, MR-TAE-noSwin-noBiGRU,
  MR-TAE-noSwin-noAttn, MR-TAE-noBiGRU-noAttn,
  MR-TAE-FULL-LossB, MR-TAE-FULL-LossC,
  MWCNN-BiGRU, UNet-BiGRU-Swin

Flag any model where epoch < 85 or phase < 3 as "INCOMPLETE".

---

## TASK 2 — VERIFY YAML REGISTRY ATTENTION GATE FLAGS

Read: config/model_registry.yaml
Check each model's use_attention_gates setting.
Expected (from ARCHITECTURE_SUMMARY.md):
  - FULL, noBiGRU, noSwin, noMTL, noWavelet, LossB, LossC,
    noSwin-noBiGRU: use_attention_gates = True
  - noAttn, noSwin-noAttn, noBiGRU-noAttn, MWCNN-BiGRU, UNet-BiGRU-Swin:
    use_attention_gates = False

Report any mismatch. Fix YAML if wrong. Do NOT retrain — just correct the registry for future runs.

---

## TASK 3 — VERIFY NCC FORMULA IN ALL EVALUATION FILES

Search for all NCC implementations:
  grep -rn "ncc\|NCC\|cross_corr" evaluation/ mr_tae_fusion/evaluation/ --include="*.py"

For each occurrence, verify it implements ZERO-MEAN normalized cross-correlation:
  NCC(a, b) = sum((a-mean(a)) * (b-mean(b))) / sqrt(sum((a-mean(a))^2) * sum((b-mean(b))^2))

Common bugs:
  BUG1: Missing mean subtraction → NCC biased by signal DC offset
  BUG2: Using torch.corrcoef → returns matrix, not scalar NCC
  BUG3: Using scipy.signal.correlate → returns array, need [peak] index
  BUG4: Dividing by n (sample count) instead of by sqrt(sum_sq_a * sum_sq_b)

Create: evaluation/metrics_verified.py with a single authoritative NCC function.
Replace all other NCC implementations with imports from this file.

Also verify:
  - NCC for SYNTHETIC data: computed against clean ground truth tensor
  - NCC for REAL data: computed against noisy input (label this clearly as "Input-Ref NCC")
  
These must NEVER be mixed in the same column or bar chart.

---

## TASK 4 — FIX AND RE-RUN CLASSICAL BASELINES

Create: evaluation/classical_baselines.py

Implement these baselines correctly:

### 4a. Wiener Filter
```python
from scipy.signal import wiener
def apply_wiener(x_noisy: np.ndarray, mysize=5) -> np.ndarray:
    """x_noisy: shape (N,) or (1,N) — 1D numpy array"""
    x = x_noisy.squeeze()
    return wiener(x, mysize=mysize)
```

### 4b. BayesShrink (wavelet soft threshold)
```python
import pywt
from skimage.restoration import denoise_wavelet, estimate_sigma
def apply_bayesshrink(x_noisy: np.ndarray) -> np.ndarray:
    x = x_noisy.squeeze()
    sigma = estimate_sigma(x, average_sigmas=True)
    return denoise_wavelet(x, method='BayesShrink', mode='soft',
                           wavelet='db4', sigma=sigma,
                           rescale_sigma=True)
```
If skimage is not installed: pip install scikit-image
Alternative manual BayesShrink using pywt:
```python
import pywt, numpy as np
def apply_bayesshrink_manual(x_noisy: np.ndarray, wavelet='db4', level=3):
    x = x_noisy.squeeze()
    coeffs = pywt.wavedec(x, wavelet, level=level)
    # Estimate noise sigma from finest detail coefficients
    sigma_n = np.median(np.abs(coeffs[-1])) / 0.6745
    denoised_coeffs = [coeffs[0]]  # keep approximation
    for detail in coeffs[1:]:
        sigma_x = np.sqrt(max(np.var(detail) - sigma_n**2, 0))
        threshold = sigma_n**2 / (sigma_x + 1e-8)
        denoised_coeffs.append(pywt.threshold(detail, threshold, mode='soft'))
    return pywt.waverec(denoised_coeffs, wavelet)[:len(x)]
```

### 4c. Savitzky-Golay
```python
from scipy.signal import savgol_filter
def apply_savgol(x_noisy: np.ndarray, window=51, poly=3) -> np.ndarray:
    x = x_noisy.squeeze()
    # Ensure window is odd and < len(x)
    window = min(window, len(x) - (1 if len(x) % 2 == 0 else 0))
    if window % 2 == 0: window -= 1
    if window < poly + 2: window = poly + 2 + (poly % 2)
    return savgol_filter(x, window_length=window, polyorder=poly)
```

### Evaluation loop for baselines:
```python
test = torch.load("data/test_synthetic_ae.pt")
snr_levels = [-15, -10, -5, 0, 5]
results = {}
for method_name, method_fn in [("wiener", apply_wiener),
                                 ("bayesshrink", apply_bayesshrink_manual),
                                 ("savgol_51", apply_savgol)]:
    nccs_by_snr = {}
    for snr in snr_levels:
        mask = test['snr'] == snr   # assumes snr key exists; adjust if needed
        noisy = test['noisy'][mask].numpy()
        clean = test['clean'][mask].numpy()
        nccs = []
        for i in range(len(noisy)):
            out = method_fn(noisy[i])
            # Ensure same length
            out = out[:clean[i].squeeze().shape[0]]
            nccs.append(ncc(out, clean[i].squeeze()))
        nccs_by_snr[snr] = float(np.mean(nccs))
        print(f"{method_name} @ {snr}dB: NCC={nccs_by_snr[snr]:.4f}")
    results[method_name] = nccs_by_snr
# Save
import json
json.dump(results, open("results/definitive_benchmark/baselines.json", "w"), indent=2)
```

EXPECTED RESULTS (sanity check — if outside these ranges, NCC formula is still broken):
  BayesShrink @ +5dB:  NCC vs clean GT > 0.70
  BayesShrink @ -5dB:  NCC vs clean GT > 0.30
  BayesShrink @ -15dB: NCC vs clean GT > 0.05
  SavGol @ +5dB:       NCC vs clean GT > 0.65
  SavGol @ -15dB:      NCC vs clean GT > 0.03 (it's a weak filter, not zero)
  Wiener @ +5dB:       NCC vs clean GT > 0.60

---

## TASK 5 — FIX LATENCY MEASUREMENT

Create: evaluation/latency_benchmark.py

Use this exact template for every model:
```python
import torch, time, yaml, pathlib
from mr_tae_fusion.models.mr_tae_fusion import MRTAEFusion
from mr_tae_fusion.models.ablation_config import AblationConfig
from mr_tae_fusion.config import Config

def load_model_from_checkpoint(model_id: str, registry: dict) -> torch.nn.Module:
    cfg_entry = next(m for m in registry['models'] if m['id'] == model_id)
    ablation = AblationConfig(**{k: cfg_entry[k] for k in
        ['use_wavelet_pooling','use_bigru','use_swin','use_attention_gates','use_mtl','loss_config']
        if k in cfg_entry})
    ablation.model_id = model_id
    model = MRTAEFusion(Config().model, ablation)  # adjust constructor call to match actual API
    
    ckpt_path = pathlib.Path(f"results/{model_id}/checkpoints/best.pt")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('model_state', ckpt.get('model_state_dict'))
    model.load_state_dict(state, strict=True)
    return model

def measure_latency_correct(model, batch_size=1, seq_len=2048,
                             n_warmup=50, n_runs=200, device='cuda'):
    model = model.to(device).eval()
    x = torch.randn(batch_size, 1, seq_len, device=device)
    
    # Warmup (fills CUDA JIT cache, avoids cold-start overhead)
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model(x)
    torch.cuda.synchronize()
    
    # Timed runs
    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    
    # Remove top/bottom 10% outliers
    times.sort()
    trim = n_runs // 10
    times = times[trim:-trim]
    return float(sum(times) / len(times))

# Run for all 13 models, save to results/definitive_benchmark/latency.json
```

EXPECTED latency ordering (should be roughly):
  noSwin-noBiGRU (1.01M) < MWCNN-BiGRU (1.31M) < noSwin-noAttn (1.36M)
      < noSwin (1.41M) < noBiGRU-noAttn (2.35M) < noBiGRU (2.39M)
      < UNet-BiGRU-Swin (2.90M) < noAttn/noMTL/noWavelet (~3M) < FULL (3.05M)

If the ordering is violated, there is still a measurement bug.

---

## TASK 6 — VERIFY CURRICULUM PHASE COMPLETION

For each checkpoint, check that training completed Phase 3:
```python
import pathlib, torch

models = [
    "MR-TAE-FULL", "MR-TAE-noBiGRU", "MR-TAE-noSwin", "MR-TAE-noAttn",
    "MR-TAE-noMTL", "MR-TAE-noWavelet", "MR-TAE-noSwin-noBiGRU",
    "MR-TAE-noSwin-noAttn", "MR-TAE-noBiGRU-noAttn",
    "MR-TAE-FULL-LossB", "MR-TAE-FULL-LossC", "MWCNN-BiGRU", "UNet-BiGRU-Swin"
]

for mid in models:
    ckpt_path = pathlib.Path(f"results/{mid}/checkpoints/best.pt")
    if not ckpt_path.exists():
        print(f"{mid}: CHECKPOINT MISSING")
        continue
    ckpt = torch.load(ckpt_path, map_location='cpu')
    epoch = ckpt.get('epoch', -1)
    phase = ckpt.get('phase', ckpt.get('curriculum_phase', -1))
    val_ncc = ckpt.get('val_ncc', ckpt.get('best_ncc', 'N/A'))
    flag = "⚠ INCOMPLETE" if (epoch < 85 or phase < 3) else "✓ OK"
    print(f"{flag} {mid:35s}: epoch={epoch}, phase={phase}, val_ncc={val_ncc}")
```

If any model is INCOMPLETE, retrain it:
```bash
python pipeline/mlflow_orchestrator.py --model {MODEL_ID} --force-restart
```
But first fix the early stopping bug in training/resilience.py so it cannot
fire before Phase 3 has run for at least 15 epochs.

---

## TASK 7 — DEFINITIVE BENCHMARK RUN

After all above tasks complete, run the full benchmark:

```python
# evaluation/run_definitive_benchmark.py
"""
Definitive benchmark — runs after all fixes confirmed.
Produces: results/definitive_benchmark/
  - synthetic_results.json    (NCC vs clean GT, per SNR level, all 13 models + baselines)
  - real_results.json         (Input-Ref NCC on Q.Lin, all 13 models + baselines)
  - latency.json              (corrected, CUDA-sync'd)
  - summary_table.csv         (all in one CSV)
  - fig_ablation_ncc_bar.png  (regenerated with correct numbers)
  - fig_snr_sweep.png         (regenerated)
  - fig_loss_comparison.png   (regenerated)
  - fig_efficiency_scatter.png (regenerated with correct latency)
"""
```

Columns required in summary_table.csv:
  model_id, params_M, lat_ms_b1, lat_ms_b32,
  syn_ncc_mean, syn_ncc_minus15, syn_ncc_minus10, syn_ncc_minus5, syn_ncc_0, syn_ncc_plus5,
  syn_rmse_mean, syn_snr_improvement_mean,
  real_inputref_ncc_mean, real_inputref_ncc_std,
  dice_mean (synthetic segmentation), fpr (synthetic),
  training_completed (True/False), best_epoch, final_phase

---

## TASK 8 — REGENERATE ALL 7 PAPER FIGURES with correct data

After Task 7 completes, regenerate all figures using results/definitive_benchmark/:

Figures to regenerate (save to results/paper_assets_v2/):
  fig1_system_overview.png     — no data dependency, keep existing
  fig2_architecture.png        — no data dependency, keep existing
  fig3_waveform_comparison.png — regenerate from definitive benchmark
  fig4_ablation_ncc_bar.png    — regenerate: both syn NCC and real input-ref NCC
  fig5_snr_sweep.png           — regenerate: top 5 models by syn NCC
  fig6_loss_comparison.png     — regenerate: LossA/B/C both syn and real NCC
  fig7_efficiency_scatter.png  — regenerate: corrected latency

IMPORTANT for fig6: Label the y-axes clearly:
  Blue bars: "Synthetic NCC (vs clean GT)"
  Orange bars: "Real NCC (vs noisy input — no GT available)"
  Add a footnote: "Real NCC measures output-to-input similarity, not denoising quality"

---

## TASK 9 — PRODUCE FINAL ANSWER

After all tasks complete, produce: results/FINAL_VERDICT_V2.md

Fill in:
  1. Best model by synthetic NCC (GT-referenced): ___
  2. Best model by real input-referenced NCC: ___
  3. Best model for low-SNR extreme noise (−15 dB synthetic NCC): ___
  4. Best efficiency candidate (NCC/params): ___
  5. Which components are load-bearing (collapse if removed): ___
  6. LossA vs LossB: which is actually better, and by how much, on which metric?
  7. BayesShrink NCC (corrected): ___
  8. Savitzky-Golay NCC (corrected): ___
  9. All 13 models confirmed trained to Phase 3: YES/NO

Report these numbers precisely. Do not copy from old tables.

---

## CONSTRAINTS FOR CURSOR AGENT

1. Always activate .venv before running any Python
2. Always run from D:\New folder (2)\Data\FYP_MASTER as working directory
3. Do NOT import from OneDrive_2025-12-13 in any new scripts — old path is dead
4. Checkpoint key is "model_state" (not "model_state_dict") — verify before load
5. All figures: dpi=300, bbox_inches='tight', facecolor='white'
6. Do NOT retrain models unless a checkpoint is explicitly flagged INCOMPLETE
7. torch.cuda.synchronize() is MANDATORY in all latency measurements
8. The two NCC metrics must NEVER appear in the same axis without clear labels
9. Log every finding to results/AUDIT_LOG.txt with timestamps
```

---

## SECTION 10: WHAT TO EXPECT FROM CORRECTED RESULTS

Based on the physics and architecture analysis, here are the expected corrected values:

### Baselines (sanity ranges)
| Method | Syn NCC @ −15dB | Syn NCC @ −5dB | Syn NCC @ +5dB | Mean |
|--------|----------------|----------------|----------------|------|
| Wiener filter | 0.02–0.10 | 0.35–0.55 | 0.60–0.75 | 0.35–0.50 |
| BayesShrink | 0.05–0.15 | 0.45–0.65 | 0.75–0.90 | 0.45–0.60 |
| Savitzky-Golay | 0.03–0.10 | 0.30–0.50 | 0.65–0.80 | 0.35–0.50 |

If corrected BayesShrink is near 0.60 mean and MR-TAE-FULL is near 0.61–0.62, that
is still a meaningful but modest improvement. This is scientifically honest.

### Deep models (expected ranges if curriculum completed correctly)
| Model | Expected Syn NCC @ −15dB | Expected Syn NCC @ +5dB |
|-------|--------------------------|------------------------|
| MR-TAE-FULL | 0.15–0.30 | 0.90–0.97 |
| MR-TAE-FULL-LossB | 0.18–0.35 | 0.90–0.97 |
| MWCNN-BiGRU | 0.10–0.20 | 0.85–0.95 |
| noWavelet | 0.04–0.12 | 0.85–0.94 |
| noMTL | 0.05–0.12 | 0.85–0.94 |

If any model shows NCC > 0.98 at +5dB, there may be test data leakage.
If any model shows NCC < 0.70 at +5dB, it likely did not converge.

---

*Forensic audit prepared by Claude Sonnet 4.6, May 2026*  
*For Oo Jian Hong (22004678), University of Malaya FYP*
