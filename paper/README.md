# MR-TAE Paper Reproducibility Index



## Source of truth



| Artifact | Path | Generator |

|----------|------|-----------|

| Master metrics | `paper/metrics_master.json` | `scripts/run_phaseA_rebench.py` |

| Table 1 (13 ablations) | `results/paper_table1_ablation.csv` | same |

| Table 2 (improved + classical) | `results/paper_table2_improved_models.csv` | same |

| Frozen synthetic test | `data/test_synthetic_ae.pt` (14k windows) | `data/build_test_sets.py` |

| Grounded real test | `data/test_real_qlin_clean.pt`, `noisy.pt`, `snr_labels.pt` | same |

| Test hashes | `docs/TEST_SET_HASHES.md` | same |



**SNR levels:** −15, −10, −5, 0, +5, +10, +15 dB (7 levels, composite noise).



**Checkpoint selection:** late-epoch candidates scored by **mean synthetic NCC across all 7 SNR levels** (tie-break −15 dB) → `checkpoints/selected_final.pt` + `checkpoint_selection.json`.



**Headline real metric:** grounded NCC — clean Q.Lin window + injected composite noise → denoise → NCC vs clean.



Do **not** use `results/FINAL_BENCHMARK_LOCKED.json` or `results/REAL_QLIN_BENCHMARK.json` (stale / inconsistent).



## Re-run Phase A (unified re-benchmark)



```powershell

cd "D:\New folder (2)\Data\FYP_MASTER"

& .venv\Scripts\Activate.ps1

python scripts/run_phaseA_rebench.py

python scripts/run_phaseA_rebench.py --no-sweep --resume   # resume only

```



Rebuild frozen test sets first (if needed):



```powershell

python data/build_test_sets.py

```



## Figures



```powershell

python scripts/make_paper_figures.py

```



| Figure | File(s) | Script | Data source |

|--------|---------|--------|-------------|

| System overview | `fig1_system_overview.png` | `make_paper_figures.py` | `results/paper_assets_v2/` (reused) |

| Architecture | `fig2_architecture.png` | same | reused |

| Loss comparison | `fig3_loss_comparison.png` | same | `paper_assets_v2/fig6_loss_comparison.png` |

| Ablation NCC bar | `fig4_ablation_ncc_bar.png`, alias `fig6_ablation_bar.png` | same | `metrics_master.json` |

| Ablation deltas | `fig6_ablation_deltas.png` | same | `metrics_master.json` |

| SNR sweep | `fig5_snr_sweep.png` | same | `metrics_master.json` |

| SNR-improvement + Dice panel | `fig7_snr_improvement_dice.png` | same | `metrics_master.json` |

| Grounded real Q.Lin | `fig7_grounded_real_qlin.png`, alias `fig9_real_qlin_grounded.png` | same | `metrics_master.json` |

| Efficiency scatter | `fig8_efficiency_scatter.png` | same | `metrics_master.json` |

| SNR improvement bar | `fig9_snr_improvement.png` | same | `metrics_master.json` |

| Dice bar | `fig10_dice_bar.png` | same | `metrics_master.json` |

| Qualitative waveforms (−15 dB) | `fig10_qualitative_waveforms.png` | same | `paper_assets_v2/fig_waveform_zoom_minus15dB.png` |

| LIME summary | `fig11_lime_summary.png` | same | `results/LIME_importances.json` |

| SHAP summary | `fig12_shap_summary.png` | same | `results/SHAP_attributions.json` |



**Note:** `fig3_loss_comparison.png` is copied from pre-rendered training-curve assets (number-independent). Qualitative panel is reused from the −15 dB waveform zoom asset.



## Explainability (LIME + SHAP at −15 dB)



```powershell

python scripts/run_explainability_all.py

```



Outputs: `results/LIME_importances.json`, `results/SHAP_attributions.json`, summary plots in `paper/figures/`.



## Fill and expand documents



```powershell

python scripts/fill_paper_from_metrics.py

python scripts/expand_research_report.py

python scripts/verify_paper_numbers.py

```



- `fill_paper_from_metrics.py` — resolves `{{TBD:...}}` and `<!-- FILL -->` blocks from metrics + config + LIME/SHAP.

- `expand_research_report.py` — regenerates Appendix A tables and adds appendices G–M (~5000-line target).

- `verify_paper_numbers.py` — spot-check + TBD token count.



## Documents



| File | Description |

|------|-------------|

| `paper/paper_arxiv.md` | ~15-page UG iFEST-style paper |

| `paper/research_report.md` | Extended research report (~5000 lines after expand) |



**Intentional placeholders:** NoMix curriculum variant only (`{{TBD: NoMix ...}}` if present). Not trained.



## Re-run order (full doc rebuild)



1. `data/build_test_sets.py`

2. `scripts/run_phaseA_rebench.py`

3. `scripts/run_explainability_all.py`

4. `scripts/make_paper_figures.py`

5. `scripts/fill_paper_from_metrics.py`

6. `scripts/expand_research_report.py`

7. `scripts/verify_paper_numbers.py`

