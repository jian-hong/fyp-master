# FINAL VERDICT V2 — MR-TAE Forensic Re-Evaluation
Generated from `results/definitive_benchmark/` (verified zero-mean NCC).

## Answers

1. **Best model by synthetic NCC (GT-referenced):** MR-TAE-FULL (0.6123)
2. **Best model by real input-referenced NCC:** MR-TAE-FULL (0.9602)
3. **Best model at −15 dB synthetic NCC:** MR-TAE-FULL (0.1951)
4. **Best efficiency (NCC/params):** MR-TAE-noSwin-noBiGRU
5. **Load-bearing components (Δ syn NCC > 0.02 when removed):** noBiGRU, noSwin, noAttn, noMTL, noWavelet
6. **LossA vs LossB:** LossB worse on synthetic mean NCC by 0.0496 (0.6123 vs 0.5627); LossC=0.5250
7. **BayesShrink NCC @ +5 dB (corrected):** 0.8678
8. **Savitzky-Golay (w=51) NCC @ +5 dB:** 0.0412 — note: heavy smoothing destroys pulse shape on AE windows (input NCC @+5 dB ≈ 0.87)
9. **All 13 models trained to Phase 3:** YES

## Latency ordering (batch=1, CUDA-sync, trimmed mean)
- MWCNN-BiGRU: 11.03 ms (1.312865 M params)
- MR-TAE-noSwin-noAttn: 12.41 ms (1.362854 M params)
- MR-TAE-noSwin: 15.40 ms (1.406636 M params)
- UNet-BiGRU-Swin: 15.57 ms (2.896209 M params)
- MR-TAE-noSwin-noBiGRU: 16.23 ms (1.011116 M params)
- MR-TAE-noBiGRU-noAttn: 17.70 ms (2.350998 M params)
- MR-TAE-noAttn: 18.84 ms (3.009686 M params)
- MR-TAE-noMTL: 19.81 ms (3.003479 M params)
- MR-TAE-noWavelet: 21.07 ms (2.98998 M params)
- MR-TAE-noBiGRU: 21.77 ms (2.39478 M params)
- MR-TAE-FULL-LossB: 22.04 ms (3.053468 M params)
- MR-TAE-FULL-LossC: 24.24 ms (3.053468 M params)
- MR-TAE-FULL: 26.88 ms (3.009686 M params)
