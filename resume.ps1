Param()

$ErrorActionPreference = "Stop"

Set-Location -Path $PSScriptRoot

function Write-Section($text) {
    Write-Host ""
    Write-Host "[$text]"
}

# 1) Activate venv
Write-Section "ENV"
$activatePath = Join-Path $PSScriptRoot ".venv\Scripts\Activate.ps1"
if (-Not (Test-Path $activatePath)) {
    Write-Error ".venv not found. Create it first: python -m venv .venv"
}
. $activatePath
Write-Host "Activated: $((Get-Command python).Source)"

# 2) CUDA check
Write-Section "GPU"
python -c "import torch; print(f'Device: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'CUDA not available')"

# 3) Session summary
Write-Section "SESSION"
$sessionPath = Join-Path $PSScriptRoot "session_state.json"
if (Test-Path $sessionPath) {
    $s = Get-Content $sessionPath -Raw | ConvertFrom-Json
    if ($s.models) {
        foreach ($p in $s.models.PSObject.Properties) {
            $mid = $p.Name
            $d = $p.Value
            $status = if ($d.status) { $d.status } else { "?" }
            $ep = if ($null -ne $d.current_epoch) { [int]$d.current_epoch } else { 0 }
            $ncc = if ($null -ne $d.best_ncc) { [double]$d.best_ncc } else { 0.0 }
            Write-Host ("  {0,-30} {1,-10} ep={2,-5} best_ncc={3:N4}" -f $mid, $status, $ep, $ncc)
        }
    } else {
        Write-Host "No model state in session_state.json"
    }
    Write-Host "Last update: $($s.last_updated)"
} else {
    Write-Host "No session found - orchestrator will initialize a new one."
}

# 4) Ensure logs folder
$logsDir = Join-Path $PSScriptRoot "logs"
if (-Not (Test-Path $logsDir)) {
    New-Item -ItemType Directory -Path $logsDir | Out-Null
}

# 5) Start MLflow UI in background
Write-Section "MLFLOW"
$mlflowLog = Join-Path $logsDir "mlflow.log"
$mlflowErr = Join-Path $logsDir "mlflow.err"
Start-Process -FilePath "mlflow" -ArgumentList "ui --host 0.0.0.0 --port 5000" -RedirectStandardOutput $mlflowLog -RedirectStandardError $mlflowErr -WindowStyle Hidden
Write-Host "MLflow UI: http://localhost:5000"
Write-Host "MLflow log: $mlflowLog"

# 6) Launch resilient orchestrator
Write-Section "TRAIN"
Write-Host "Launching: python pipeline/mlflow_orchestrator.py --resume"
Write-Host "Live log: logs/training_live.log"
python pipeline/mlflow_orchestrator.py --resume

Write-Section "DONE"
Write-Host "Training run finished."
Write-Host "Run .\resume.ps1 again anytime to continue remaining models."

