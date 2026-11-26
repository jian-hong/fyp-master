import os
import sys

# venv and mlflow flush
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils_dag import init_dag_phase, verify_test_sets

def main():
    init_dag_phase()
    verify_test_sets() # Explicit test set verification as requested
    
    print("Executing supervised orchestration...")
    import subprocess
    cmd = [sys.executable, "pipeline/mlflow_orchestrator.py", "--resume"]
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(__file__)))
    if result.returncode != 0:
        sys.exit(result.returncode)

if __name__ == "__main__":
    main()
