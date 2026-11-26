# scripts/start_15day.ps1
# ─────────────────────────────────────────────────────────────────────────────
# One-click launcher for the 15-Day Autonomous Training Campaign.
# This replaces resume.ps1 as the primary entry point.
#
# MODES:
#   .\scripts\start_15day.ps1              → resume/start campaign
#   .\scripts\start_15day.ps1 -Start       → force fresh campaign (dangerous!)
#   .\scripts\start_15day.ps1 -Status      → print campaign state and exit
#   .\scripts\start_15day.ps1 -DryRun      → validate config, print plan, no execution
#   .\scripts\start_15day.ps1 -EvalOnly    → run evaluation + figures only
#   .\scripts\start_15day.ps1 -RetryPhase "phase_A_train" → retry a failed phase
#
# DROP INTO: FYP_MASTER\scripts\start_15day.ps1
# RUN FROM:  D:\New folder (2)\Data\FYP_MASTER\
# ─────────────────────────────────────────────────────────────────────────────

[CmdletBinding()]
param(
    [switch]$Start,
    [switch]$Status,
    [switch]$DryRun,
    [switch]$EvalOnly,
    [string]$RetryPhase = "",
    [switch]$InstallTask
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot | Split-Path -Parent

Set-Location -Path $ProjectRoot

function Write-Banner($text) {
    Write-Host ""
    Write-Host "┌─ $text " -ForegroundColor Cyan
}

function Write-Ok($text) { Write-Host "  ✅ $text" -ForegroundColor Green }
function Write-Warn($text) { Write-Host "  ⚠️  $text" -ForegroundColor Yellow }
function Write-Fail($text) { Write-Host "  ❌ $text" -ForegroundColor Red }

# ── Power settings (run once, idempotent) ─────────────────────────────────────
Write-Banner "POWER SETTINGS"
try {
    powercfg /change standby-timeout-ac 0 2>$null
    powercfg /change hibernate-timeout-ac 0 2>$null
    powercfg /change monitor-timeout-ac 30 2>$null
    powercfg -setactive SCHEME_MIN 2>$null
    Write-Ok "Sleep/standby disabled, high-performance plan active"
} catch {
    Write-Warn "Could not set power settings (run as admin for best results)"
}

# ── Activate venv ─────────────────────────────────────────────────────────────
Write-Banner "ENVIRONMENT"
$venvActivate = Join-Path $ProjectRoot ".venv\Scripts\Activate.ps1"
if (-Not (Test-Path $venvActivate)) {
    Write-Fail ".venv not found at $venvActivate"
    Write-Host "Create it: python -m venv .venv && .venv\Scripts\pip install -r requirements.txt"
    exit 1
}
. $venvActivate
Write-Ok "Python: $((Get-Command python).Source)"

# ── GPU check ─────────────────────────────────────────────────────────────────
Write-Banner "GPU"
$gpuInfo = python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')" 2>&1
if ($gpuInfo -match "NO CUDA") {
    Write-Warn "CUDA not available — training will be slow (CPU mode)"
} else {
    Write-Ok $gpuInfo
}

# ── Campaign state summary ────────────────────────────────────────────────────
Write-Banner "CAMPAIGN STATE"
$statePath = Join-Path $ProjectRoot "campaign_state.json"
if (Test-Path $statePath) {
    $s = Get-Content $statePath -Raw | ConvertFrom-Json
    Write-Host "  Version:       $($s.campaign_version)"
    Write-Host "  Started:       $($s.start_iso)"
    Write-Host "  Current phase: $($s.current_phase_id)"
    Write-Host "  Best real NCC: $($s.best_real_ncc)"
    Write-Host "  Best model:    $($s.best_model_id)"
    Write-Host "  Figures:       $($s.figure_manifest.Count)"
    if ($s.phases) {
        Write-Host "  Phases:"
        foreach ($phase in $s.phases.PSObject.Properties) {
            $pid = $phase.Name
            $pdata = $phase.Value
            $status = if ($pdata.status) { $pdata.status } else { "?" }
            $ncc = if ($pdata.metrics -and $pdata.metrics.best_real_ncc) { 
                " NCC=$([math]::Round([double]$pdata.metrics.best_real_ncc, 4))" 
            } else { "" }
            $emoji = switch ($status) {
                "done"    { "✅" }
                "running" { "🔄" }
                "failed"  { "❌" }
                "skipped" { "⏭️" }
                default   { "⏳" }
            }
            Write-Host ("    {0} {1,-35} {2}{3}" -f $emoji, $pid, $status, $ncc)
        }
    }
} else {
    Write-Warn "No campaign_state.json found — will start fresh."
}

# ── Install as Windows Task ────────────────────────────────────────────────────
if ($InstallTask) {
    Write-Banner "INSTALLING WINDOWS TASK"
    $taskScript = Join-Path $ProjectRoot "scripts\install_watchdog_task.ps1"
    if (Test-Path $taskScript) {
        & $taskScript -TaskName "MR-TAE-Campaign-15Day" -ProjectRoot $ProjectRoot
    } else {
        Write-Warn "install_watchdog_task.ps1 not found — install manually"
    }
}

# ── Status only ───────────────────────────────────────────────────────────────
if ($Status) {
    python pipeline/autonomous_loop.py --status
    exit 0
}

# ── Dry run ───────────────────────────────────────────────────────────────────
if ($DryRun) {
    Write-Banner "DRY RUN"
    python pipeline/autonomous_loop.py --dry-run
    exit 0
}

# ── Eval only ────────────────────────────────────────────────────────────────
if ($EvalOnly) {
    Write-Banner "EVALUATION ONLY"
    python pipeline/autonomous_loop.py --eval-only
    exit 0
}

# ── Ensure logs directory ─────────────────────────────────────────────────────
$logsDir = Join-Path $ProjectRoot "logs"
if (-Not (Test-Path $logsDir)) { New-Item -ItemType Directory -Path $logsDir | Out-Null }

# ── Start MLflow UI in background ────────────────────────────────────────────
Write-Banner "MLFLOW UI"
$mlflowLog = Join-Path $logsDir "mlflow.log"
$mlflowErr = Join-Path $logsDir "mlflow.err"
$mlflowProc = Get-Process -Name "python" -ErrorAction SilentlyContinue | 
    Where-Object { $_.CommandLine -match "mlflow" } | Select-Object -First 1
if ($mlflowProc) {
    Write-Ok "MLflow already running (PID $($mlflowProc.Id))"
} else {
    Start-Process -FilePath "mlflow" `
        -ArgumentList "ui --host 0.0.0.0 --port 5000" `
        -RedirectStandardOutput $mlflowLog `
        -RedirectStandardError $mlflowErr `
        -WindowStyle Hidden
    Write-Ok "MLflow UI started → http://localhost:5000"
}

# ── Launch the campaign ───────────────────────────────────────────────────────
Write-Banner "LAUNCHING 15-DAY CAMPAIGN"
$mainArgs = if ($Start) { "--start" } else { "--resume" }
if ($RetryPhase) { $mainArgs = "$mainArgs --retry-phase $RetryPhase" }

$liveLog = Join-Path $logsDir "campaign_live.log"
Write-Host "  Mode:     $mainArgs"
Write-Host "  Config:   config/campaign_15day.yaml"
Write-Host "  State:    campaign_state.json"
Write-Host "  Live log: $liveLog"
Write-Host "  MLflow:   http://localhost:5000"
Write-Host "  Dashboard: http://localhost:1001"
Write-Host ""

# Tee output to both console and log file
$campaignArgs = "pipeline/autonomous_loop.py $mainArgs"
python pipeline/autonomous_loop.py $mainArgs.Split(" ") 2>&1 | Tee-Object -FilePath $liveLog

Write-Banner "DONE"
Write-Host "Campaign run finished. Check campaign_state.json for phase statuses."
Write-Host "Re-run .\scripts\start_15day.ps1 to continue any remaining phases."
