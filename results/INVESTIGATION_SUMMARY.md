## 1. Why use_idwt=False
IDWT Root Cause Analysis

- Q1 configurable flag in model_registry/config: NO. No use_idwt key found in config/model_registry.yaml or ModelConfig dataclass.
- Q2 actual IDWT code in decoder: YES. MWCNNDecoderBlock implements self.idwt path, but MWCNNDecoder constructor does not expose/pass use_idwt, so blocks default to interpolation path.
- Q3 when set to False: In current code, MWCNNDecoderBlock defaults use_idwt=False and MWCNNDecoder never overrides it. No git history for mwcnn.py path was retrievable in this repo snapshot, so exact commit cannot be determined.
- Q4 behavior when use_idwt=True: Direct construction MWCNNDecoder(use_idwt=True) fails because MWCNNDecoder has no such argument. IDWT can only be activated by constructing decoder blocks manually or refactoring constructor.

Shape diagnostics from tests:

Conclusion:
- IDWT is effectively disabled in production path due to API wiring gap (not exposed) plus a shape mismatch (+2 samples in db4 roundtrip: 2048 -> 1025 -> 2050).
- Fix requires decoder API plumbing (use_idwt propagated to blocks) and deterministic crop/pad at each upsample stage to exactly match skip/output lengths.
TODO:
IDWT requires retraining from scratch. Reason: current model API does not expose use_idwt at decoder level, and db4 IDWT roundtrip returns length mismatch (2048->2050), so inference toggle is not plug-and-play with existing checkpoint.
Recommended action: retrain MR-TAE-FULL with use_idwt=True (after decoder wiring + crop/pad fix) using existing curriculum and compare for paper.


## 2. IDWT vs Linear Upsample Performance
{
  "use_idwt=False": "NOT_RUN (decoder flag not exposed in current model API)",
  "use_idwt=True": "FAILED (requires architecture change + retraining)"
}

Conclusion: Current codebase originally could not toggle IDWT via model API; after wiring fixes, forward pass is now functional, but apples-to-apples benchmark with existing checkpoint is still not valid without dedicated retraining/comparison.

## 3. Segmentation Head Configurations Found
Segmentation/Class Head Variants Audit

Q1: Distinct configurations found = 4
  1) 4-class distance classification (gap-distance focused, classifier heads).
  2) 5-class segmentation in core Config (background + 4 PD-type classes).
  3) 4-class segmentation defaults in some model/eval modules (background/corona/surface/internal).
  4) Binary PD/no-PD appears in comparison scripts (method-level baseline workflow).

Q2: Gap-distance label users (1.0/1.8/2.0/2.5 mm):
  - Try Model/mr_tae_fusion/data/distance_signals.py (gap_distance_mm generation)
  - Try Model/mr_tae_fusion/models/distance_classifier.py (num_classes=4)
  - compare_methods.py, run_training.py, train_extended.py (distance-based 4-class pipelines)

Q3: PD-type label users (Corona/Surface/Internal/etc.):
  - Try Model/mr_tae_fusion/config.py type_to_class mapping and num_classes=5
  - Class names in evaluation mention Corona/Surface/Internal (+Background) in visualization helpers

Q4: Binary PD/no-PD users (2 classes):
  - compare_methods.py includes binary detection task description (PD present vs absent).

Q5: 5-class segmentation found in audit:
  - Config-level setting is 5 classes in Try Model/mr_tae_fusion/config.py.
  - NOTE: comments describe PD-type taxonomy (Background/Corona/Surface/Internal/Treeing), not explicit 4-gap-distance + background.

Q6: Output-size vs label-format mismatch? YES, potential mismatches exist:
  - Fusion head class defaults are 4 in Try Model/mr_tae_fusion/models/fusion_head.py, while Config uses 5 classes.
  - Evaluation metrics/utils also default to 4 classes in multiple places.
  - This can mismatch if training labels are 5-class while eval/head code assumes 4-class.

Which configuration is correct for the paper?
- Core MR-TAE training config currently targets 5 classes at config level. Verify whether this means PD-type taxonomy or 4-gap+background labeling before final manuscript wording.

## 4. Most Robust Model
- MR-TAE-FULL: 0.4663
- MR-TAE-noBiGRU-noAttn: 0.4436
- MR-TAE-FULL-LossB: 0.4282
- MR-TAE-noSwin-noAttn: 0.4247
- MR-TAE-noSwin-noBiGRU: 0.423

Winner and why
- MR-TAE-FULL ranks first by the weighted robustness score emphasizing -15 dB performance.

## 5. Files Cleaned Up
- Old path references fixed: all OneDrive_2025-12-13 refs removed (0 remaining)
- Duplicate files found in old/new filename overlap audit: 0

## 6. GitHub Folder Contents
- .gitignore
- config/model_registry.yaml
- config/training_config.yaml
- docs/ARCHITECTURE_VERIFIED.md
- docs/ROBUSTNESS_RANKING.json
- evaluation/benchmark_runner.py
- evaluation/classical_baselines.py
- evaluation/metrics_verified.py
- figures/fig4_ablation.png
- figures/fig5_snr_sweep.png
- figures/fig7_efficiency.png
- mr_tae_fusion/__init__.py
- mr_tae_fusion/config.py
- mr_tae_fusion/models/__init__.py
- mr_tae_fusion/models/ablation_config.py
- mr_tae_fusion/models/mr_tae_fusion.py
- mr_tae_fusion/models/mwcnn.py
- mr_tae_fusion/models/swin_transformer.py
- mr_tae_fusion/models/wavelet_layers.py
- pipeline/mlflow_orchestrator.py
- README.md
- releases/MR-TAE-FULL-best.pt
- requirements.txt
- results/baselines.json
- results/summary_table.csv
- training/curricular_scheduler.py
- training/resilience.py
- training/trainer.py

- Checkpoint size and location: MR-TAE-FULL-best.pt present in releases/ (34.6 MB).
- Non-checkpoint package size: 0.77 MB

## 7. Pending Experiments
PENDING: Train MR-TAE-FULL-IDWT using pipeline/mlflow_orchestrator.py --model MR-TAE-FULL-IDWT


## 8. Recommended Next Actions for Paper
- Call MR-TAE-FULL the best primary model (robustness winner), with hard-noise caveat checked against top alternatives.
- Retrain and benchmark MR-TAE-FULL-IDWT as a dedicated ablation before claiming IDWT benefit.
- Clarify segmentation label definition (5-class PD-type vs 4-gap+background) and keep methodology consistent with training/eval code.
- Keep release repo focused on reproducible scripts + benchmark tables, with checkpoints in 
eleases/ or external link.
