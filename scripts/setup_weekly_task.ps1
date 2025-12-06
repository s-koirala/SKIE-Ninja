# SKIE_Ninja Weekly Retraining Task Setup
# Run this script as Administrator to create the scheduled task

$TaskName = "SKIE_Ninja_Weekly_Retrain"
$TaskDescription = "Weekly ONNX model retraining for SKIE_Ninja trading strategy"
$ProjectDir = "C:\Users\skoir\Documents\SKIE Enterprises\SKIE-Ninja\SKIE-Ninja-Project\SKIE_Ninja"
$ScriptPath = "$ProjectDir\scripts\weekly_retrain.bat"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " SKIE_Ninja Weekly Retraining Task Setup" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "`nERROR: This script must be run as Administrator" -ForegroundColor Red
    Write-Host "Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    exit 1
}

# Remove existing task if it exists
$existingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existingTask) {
    Write-Host "`nRemoving existing task..." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

# Create the action
$Action = New-ScheduledTaskAction -Execute $ScriptPath -WorkingDirectory $ProjectDir

# Create the trigger (every Sunday at 6:00 PM)
$Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At "18:00"

# Create the principal (run whether user is logged on or not)
$Principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest

# Create settings
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries

# Register the task
Write-Host "`nCreating scheduled task..." -ForegroundColor Green
Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Principal $Principal -Settings $Settings -Description $TaskDescription

# Verify
$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($task) {
    Write-Host "`nSUCCESS: Scheduled task created!" -ForegroundColor Green
    Write-Host "`nTask Details:" -ForegroundColor Cyan
    Write-Host "  Name:      $TaskName"
    Write-Host "  Schedule:  Every Sunday at 6:00 PM"
    Write-Host "  Script:    $ScriptPath"
    Write-Host "`nTo manage this task:" -ForegroundColor Yellow
    Write-Host "  - Open Task Scheduler (taskschd.msc)"
    Write-Host "  - Look for '$TaskName' in Task Scheduler Library"
    Write-Host "`nTo run manually:" -ForegroundColor Yellow
    Write-Host "  Start-ScheduledTask -TaskName '$TaskName'"
} else {
    Write-Host "`nERROR: Failed to create scheduled task" -ForegroundColor Red
}

Write-Host "`n============================================================" -ForegroundColor Cyan
