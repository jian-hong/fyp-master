# install_watchdog_task.ps1
# -----------------------------------------------------------------------------
# Registers the MR-TAE training watchdog as a Windows Task Scheduler job
# so that the GPU runs the orchestrator 24/7 -- automatically resuming
# after reboots, crashes, Windows updates, and brief power outages
# (assuming BIOS "Restore on AC power loss" is set).
#
# Drop into:
#   FYP_MASTER\scripts\install_watchdog_task.ps1
#
# Usage (right-click PowerShell -> Run as Administrator):
#   .\scripts\install_watchdog_task.ps1                 # install the task
#   .\scripts\install_watchdog_task.ps1 -Uninstall      # remove it
#   .\scripts\install_watchdog_task.ps1 -ShowStatus     # query its state
# -----------------------------------------------------------------------------

[CmdletBinding()]
param(
    [switch]$Uninstall,
    [switch]$ShowStatus,
    [string]$TaskName = "MR-TAE-Watchdog",
    [string]$ProjectRoot = "D:\New folder (2)\Data\FYP_MASTER"
)

$ErrorActionPreference = "Stop"

# -----------------------------------------------------------------------------
# Self-elevate to admin if not already
# -----------------------------------------------------------------------------
$isAdmin = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(`
    [Security.Principal.WindowsBuiltInRole] "Administrator")
if (-not $isAdmin) {
    Write-Warning "Re-launching as Administrator..."
    $argList = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $MyInvocation.MyCommand.Path)
    if ($Uninstall)   { $argList += "-Uninstall" }
    if ($ShowStatus)  { $argList += "-ShowStatus" }
    Start-Process powershell -Verb RunAs -ArgumentList $argList
    exit
}

# -----------------------------------------------------------------------------
# Status / Uninstall paths
# -----------------------------------------------------------------------------
if ($ShowStatus) {
    Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue |
        Format-List TaskName, State, LastRunTime, NextRunTime, LastTaskResult
    exit 0
}

if ($Uninstall) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "Removed scheduled task: $TaskName" -ForegroundColor Green
    exit 0
}

# -----------------------------------------------------------------------------
# Install
# -----------------------------------------------------------------------------
$venvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$watchdog   = Join-Path $ProjectRoot "pipeline\watchdog.py"

foreach ($p in @($ProjectRoot, $venvPython, $watchdog)) {
    if (-not (Test-Path $p)) {
        Write-Error "Missing required path: $p"
        exit 1
    }
}

# Action: run watchdog.py with the project venv
$action = New-ScheduledTaskAction `
    -Execute $venvPython `
    -Argument "`"$watchdog`"" `
    -WorkingDirectory $ProjectRoot

# Triggers: at user logon AND at system boot, plus a 5-minute retry every hour
# if the task isn't running -- this is the belt-and-braces 24/7 guarantee.
$triggerLogon = New-ScheduledTaskTrigger -AtLogOn
$triggerBoot  = New-ScheduledTaskTrigger -AtStartup
$triggerHourly = New-ScheduledTaskTrigger `
    -Once -At (Get-Date) `
    -RepetitionInterval (New-TimeSpan -Hours 1) `
    -RepetitionDuration (New-TimeSpan -Days 365)

# Settings: highest priority for the GPU, NEVER stop, restart on failure,
# do NOT pause when on battery (this is a desktop), allow start on demand.
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 5 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Days 0) `
    -MultipleInstances IgnoreNew

# Principal: run with highest privileges as the current user
$principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive `
    -RunLevel Highest

# Register
Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger @($triggerLogon, $triggerBoot, $triggerHourly) `
    -Settings $settings `
    -Principal $principal `
    -Description "MR-TAE training watchdog -- keeps mlflow_orchestrator running 24/7" `
    -Force | Out-Null

Write-Host ""
Write-Host "Installed scheduled task: $TaskName" -ForegroundColor Green
Write-Host "  Triggers:    logon, system boot, hourly retry" -ForegroundColor Gray
Write-Host "  Action:      $venvPython `"$watchdog`"" -ForegroundColor Gray
Write-Host "  Working dir: $ProjectRoot" -ForegroundColor Gray
Write-Host ""
Write-Host "Recommended Windows power settings (run these once):" -ForegroundColor Cyan
Write-Host "  powercfg /change standby-timeout-ac 0"
Write-Host "  powercfg /change hibernate-timeout-ac 0"
Write-Host "  powercfg /change monitor-timeout-ac 30"
Write-Host "  powercfg -setactive SCHEME_MIN  # high-performance"
Write-Host ""
Write-Host "Trigger now (without rebooting):" -ForegroundColor Cyan
Write-Host "  Start-ScheduledTask -TaskName $TaskName"
Write-Host ""
Write-Host "Verify status:" -ForegroundColor Cyan
Write-Host "  .\scripts\install_watchdog_task.ps1 -ShowStatus"
