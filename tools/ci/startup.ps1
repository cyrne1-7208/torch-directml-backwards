# Copyright (c) 2026 Cyrne1. MIT License; see LICENSE.
# Run once per disposable VM. The JIT runner accepts only one job.
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
if (Get-ScheduledTask -TaskPath '\' | Where-Object TaskName -eq 'tdml-runner') { return }

function Get-Metadata([string]$Key) {
    Invoke-RestMethod "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$Key" `
        -Headers @{ 'Metadata-Flavor' = 'Google' } -TimeoutSec 10
}

$Version = Get-Metadata runner-version
$ExpectedHash = Get-Metadata runner-sha256
$Jit = Get-Metadata jit-config
if ((Get-Metadata tdml-ci-revision) -notmatch '^[0-9a-f]{40}$') {
    throw 'This script requires a CI revision in the VM metadata.'
}

# These two settings are adapted from runner-images; see LICENSE.runner-images.
Set-MpPreference -DisableRealtimeMonitoring $true -DisableArchiveScanning $true
$Preference = Get-MpPreference
if (-not $Preference.DisableRealtimeMonitoring -or -not $Preference.DisableArchiveScanning) {
    throw 'Windows Defender CI preferences did not take effect.'
}

function Install-Component([string]$Path, [string[]]$Arguments) {
    $Process = Start-Process $Path -ArgumentList $Arguments -PassThru
    # Cache the handle before waiting so PowerShell 5.1 retains ExitCode.
    $null = $Process.Handle
    try {
        if (-not $Process.WaitForExit(300000)) {
            Stop-Process -Id $Process.Id -Force
            throw "Installer timed out: $Path"
        }
        $Process.WaitForExit()
        # Exit code 3010 means installation succeeded but requires a reboot.
        if ($Process.ExitCode -notin @(0, 3010)) { throw "Installer failed: $Path (exit $($Process.ExitCode))" }
        return $Process.ExitCode
    }
    finally { $Process.Dispose() }
}

# Install before the single-use runner starts: a requested reboot cannot interrupt a job.
$Prerequisites = 'C:\tdml-prerequisites-installed'
if (-not (Test-Path $Prerequisites)) {
    $Vc = Join-Path $env:TEMP 'vc_redist.x64.exe'
    Invoke-WebRequest 'https://aka.ms/vs/17/release/vc_redist.x64.exe' -OutFile $Vc -UseBasicParsing -TimeoutSec 300
    $VcExit = Install-Component $Vc @('/install', '/quiet', '/norestart')
    $Driver = Join-Path $env:TEMP 'nvidia-grid.exe'
    Invoke-WebRequest 'https://storage.googleapis.com/compute-gpu-installation-us/windows/596.86_grid_win10_win11_server2022_server_2025_dch_64bit_international.exe' `
        -OutFile $Driver -UseBasicParsing -TimeoutSec 300
    if ((Get-FileHash $Driver -Algorithm SHA256).Hash -ine '4b926d9e8f6ba1f391da436a1724e1d104081d6ea6a9604c77738caa201a9627') {
        throw 'NVIDIA GRID driver SHA256 mismatch.'
    }
    $DriverExit = Install-Component $Driver @('/s', '/n')
    New-Item -ItemType File -Path $Prerequisites | Out-Null
    Remove-Item $Vc, $Driver -Force
    if ($VcExit -eq 3010 -or $DriverExit -eq 3010) {
        Restart-Computer -Force
        return
    }
}
$Gpu = & "$env:WINDIR\System32\nvidia-smi.exe" --query-gpu=name,driver_model.current --format=csv,noheader
if ($LASTEXITCODE -ne 0 -or $Gpu -notmatch 'T4.*WDDM') {
    throw "A T4 with a WDDM driver is required: $Gpu"
}
$Directory = 'C:\actions-runner'
New-Item -ItemType Directory -Path $Directory -Force | Out-Null
$Archive = Join-Path $env:TEMP 'actions-runner.zip'
Invoke-WebRequest "https://github.com/actions/runner/releases/download/v$Version/actions-runner-win-x64-$Version.zip" `
    -OutFile $Archive -UseBasicParsing -TimeoutSec 300
if ((Get-FileHash $Archive -Algorithm SHA256).Hash -ine $ExpectedHash) {
    throw 'GitHub runner archive SHA256 mismatch.'
}
Expand-Archive $Archive $Directory -Force
Remove-Item $Archive -Force

# Give the runner its own user profile instead of sharing the GCE guest agent's SYSTEM profile.
$Bytes = New-Object byte[] 24
$Random = [Security.Cryptography.RandomNumberGenerator]::Create()
try { $Random.GetBytes($Bytes) } finally { $Random.Dispose() }
$Password = [Convert]::ToBase64String($Bytes) + 'aA1!'
New-LocalUser runner -Password (ConvertTo-SecureString $Password -AsPlainText -Force) -PasswordNeverExpires | Out-Null
Add-LocalGroupMember -SID 'S-1-5-32-544' -Member runner
$Task = @{
    TaskName = 'tdml-runner'
    Action = New-ScheduledTaskAction -Execute "$Directory\run.cmd" -Argument "--jitconfig $Jit" -WorkingDirectory $Directory
    Settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes 50)
    User = 'runner'
    Password = $Password
    RunLevel = 'Highest'
}
Register-ScheduledTask @Task | Out-Null
Start-ScheduledTask tdml-runner
