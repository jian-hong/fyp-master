from pathlib import Path
import subprocess, sys

cmd = [
    sys.executable, "evaluation/benchmark_runner.py",
    "--model-id", "MR-TAE-FULL-LossB",
    "--checkpoint", "results/MR-TAE-FULL-LossB/checkpoints/best.pt",
    "--no-attention-gates"
]
subprocess.run(cmd)
