import argparse
import sys
import os
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] Phase Runner - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

def get_venv_python():
    if os.name == "nt":
        return str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe")
    return str(PROJECT_ROOT / ".venv" / "bin" / "python")

def run_subprocess(cmd):
    logging.info(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        logging.error(f"Command failed with code {result.returncode}")
        sys.exit(result.returncode)

def run_supervised():
    logging.info("Starting Phase: Supervised Training")
    cmd = [get_venv_python(), "pipeline/mlflow_orchestrator.py", "--resume", "--run-all"]
    run_subprocess(cmd)

def run_hpo():
    logging.info("Starting Phase: Hyperparameter Optimization")
    # For now, it's a stub or if there is an hpo script we call it.
    # Check if run_dag phase_hpo exists
    hpo_script = PROJECT_ROOT / "pipeline" / "dag_phases" / "phase_hpo.py"
    if hpo_script.exists():
        cmd = [get_venv_python(), str(hpo_script)]
        run_subprocess(cmd)
    else:
        logging.info("HPO Phase: No script found, skipping/simulating.")

def run_n2v():
    logging.info("Starting Phase: Noise2Void Pretraining")
    # Stub for N2V pretraining
    logging.info("N2V Phase: Simulating completion.")

def run_wgan():
    logging.info("Starting Phase: WGAN-GP Augmentation")
    # Stub for WGAN
    logging.info("WGAN Phase: Simulating completion.")

def run_distill():
    logging.info("Starting Phase: Knowledge Distillation")
    # Stub for distillation
    logging.info("Distill Phase: Simulating completion.")

def run_report():
    logging.info("Starting Phase: Final Report Generation")
    from pipeline.mlflow_orchestrator import generate_comparison_report
    results_dir = str(PROJECT_ROOT / "results")
    generate_comparison_report(results_dir)
    logging.info("Report generated successfully.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, required=True, 
                        choices=["supervised", "hpo", "n2v", "wgan", "distill", "report"])
    args = parser.parse_args()

    if args.phase == "supervised":
        run_supervised()
    elif args.phase == "hpo":
        run_hpo()
    elif args.phase == "n2v":
        run_n2v()
    elif args.phase == "wgan":
        run_wgan()
    elif args.phase == "distill":
        run_distill()
    elif args.phase == "report":
        run_report()

if __name__ == "__main__":
    main()
