<#
.SYNOPSIS
Continuous 24/7 DAG Orchestration for AE-PD FYP

.DESCRIPTION
This script serves as a lightweight alternative to Apache Airflow. It runs continuously,
cycling through the standard MLflow orchestration queue, followed by Optuna HPO, 
WGAN-GP adversarial noise generation, and finally Noise2Void unsupervised finetuning.
It handles crashes automatically to ensure the GPU is utilized 24/7.
#>

$ErrorActionPreference = "Continue"

Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host "    AE-PD ULTIMATE 24/7 DAG CONTINUOUS RUNNER            " -ForegroundColor Cyan
Write-Host "=========================================================" -ForegroundColor Cyan

# Activate Environment
$venvPath = Join-Path $PSScriptRoot "..\.venv\Scripts\Activate.ps1"
if (Test-Path $venvPath) {
    . $venvPath
    Write-Host "Virtual environment activated." -ForegroundColor Green
} else {
    Write-Host "Virtual environment script not found at $venvPath" -ForegroundColor Yellow
}

$cycle = 1

while ($true) {
    Write-Host "`n>>> [DAG Cycle $cycle] Starting Full MLflow Orchestrator Suite" -ForegroundColor Magenta
    
    # 1. Standard Supervised Curriculum Training (The main orchestrator queue)
    Write-Host "--> Running Supervised Models Pipeline..." -ForegroundColor Cyan
    python pipeline/mlflow_orchestrator.py --resume
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "MLflow Orchestrator hit an error or stopped. Continuing DAG..." -ForegroundColor Yellow
    }

    # 2. WGAN-GP Generative Augmentation (Phase 6)
    Write-Host "`n--> Running WGAN-GP Real-Noise Generation Pipeline..." -ForegroundColor Cyan
    if (Test-Path "Try Model/examples/train_wgan.py") {
        python "Try Model/examples/train_wgan.py" --epochs 10 --samples 5000
    }

    # 3. Noise2Void Unsupervised Finetuning (Phase 7)
    Write-Host "`n--> Running N2V Self-Supervised Finetuning Pipeline..." -ForegroundColor Cyan
    if (Test-Path "Try Model/examples/n2v_finetune.py") {
        python "Try Model/examples/n2v_finetune.py" --epochs 5 --model_path "checkpoints/best_model.pt"
    }
    
    # 4. Wait & Reset
    Write-Host "`n>>> [DAG Cycle $cycle] Completed." -ForegroundColor Green
    Write-Host "Cooling down GPU for 60 seconds before next master cycle..." -ForegroundColor DarkGray
    Start-Sleep -Seconds 60
    
    # Optional logic to clear session state so it restarts the queue indefinitely could be added here
    # Remove-Item session_state.json -ErrorAction SilentlyContinue
    
    $cycle++
}
