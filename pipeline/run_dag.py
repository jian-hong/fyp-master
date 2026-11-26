import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Enforce .venv
from utils_dag import enforce_venv
enforce_venv()

PHASES = [
    "supervised",
    "hpo",
    "wgan",
    "n2v",
    "rl_finetune",
    "benchmark"
]

DAG_STATE_FILE = Path("pipeline/dag_state.json")
DAG_LOG_FILE = Path("logs/dag.log")

def log_dag(msg):
    os.makedirs(DAG_LOG_FILE.parent, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}\n"
    print(formatted.strip())
    with open(DAG_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(formatted)

def load_state():
    if not DAG_STATE_FILE.exists():
        return {
            "cycle": 1,
            "current_phase": "supervised",
            "completed_phases": []
        }
    with open(DAG_STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_state(state):
    with open(DAG_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=4)

def run_phase_script(phase_name):
    script_path = Path(f"pipeline/dag_phases/phase_{phase_name}.py")
    if not script_path.exists():
        log_dag(f"ERROR: Phase script {script_path} not found.")
        sys.exit(1)
        
    cmd = [sys.executable, str(script_path)]
    log_dag(f"STARTING PHASE: {phase_name} -> {' '.join(cmd)}")
    
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        log_dag(f"PHASE FAILED: {phase_name} with exit code {result.returncode}")
        log_dag(f"OUTPUT:\n{result.stdout}")
        return False
    else:
        log_dag(f"PHASE COMPLETED: {phase_name}")
        return True

def main():
    parser = argparse.ArgumentParser(description="Eternal DAG Orchestrator")
    parser.add_argument("--once", action="store_true", help="Run one full cycle and exit")
    parser.add_argument("--phase", type=str, choices=PHASES, help="Jump to specific phase")
    parser.add_argument("--cooldown", type=int, default=60, help="GPU cooldown seconds between phases")
    args = parser.parse_args()
    
    state = load_state()
    
    if args.phase:
        state["current_phase"] = args.phase
        log_dag(f"Parameter override: jumping to {args.phase}")
        save_state(state)
        
    while True:
        current_idx = PHASES.index(state["current_phase"])
        
        for idx in range(current_idx, len(PHASES)):
            phase = PHASES[idx]
            state["current_phase"] = phase
            save_state(state)
            
            success = run_phase_script(phase)
            if not success:
                log_dag("DAG Halted due to phase failure.")
                sys.exit(1)
                
            if phase not in state["completed_phases"]:
                state["completed_phases"].append(phase)
            save_state(state)
            
            # Cooldown
            if idx < len(PHASES) - 1:
                log_dag(f"Cooling down GPU for {args.cooldown} seconds...")
                time.sleep(args.cooldown)
                
        # Cycle complete
        log_dag(f"Cycle {state['cycle']} completely finished.")
        state["cycle"] += 1
        state["current_phase"] = "supervised"
        state["completed_phases"] = []
        save_state(state)
        
        if args.once:
            log_dag("Exiting because --once flag was specified.")
            break

if __name__ == "__main__":
    main()
