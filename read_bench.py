import json
from pathlib import Path
d = json.loads(Path('results/MR-TAE-FULL-LossB/benchmark_results.json').read_text())
print('=== SYNTHETIC ===')
print(f'NCC mean:        {d["synthetic"]["ncc_mean"]:.4f}')
print(f'SNR improvement: {d["synthetic"]["snr_improvement_mean"]:.4f} dB')
print(f'Dice:            {d["synthetic"]["dice_mean"]:.4f}')
for snr, v in d['synthetic']['per_snr_ncc'].items():
    print(f'  SNR {snr:>4} dB -> NCC {v["ncc_mean"]:.4f}')
print()
rg = d.get('real_qlin_grounded')
if rg:
    print('=== REAL GROUNDED (vs clean Q.Lin) ===')
    print(f'NCC mean: {rg["ncc_mean"]:.4f}')
    if 'per_snr_ncc' in rg:
        for snr, v in rg['per_snr_ncc'].items():
            print(f'  SNR {snr:>4} dB -> NCC {v["ncc_mean"]:.4f}')
else:
    print('real_qlin_grounded not found in results')
