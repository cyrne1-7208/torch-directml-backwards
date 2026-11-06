param(
    [string]$Root = 'C:\tdml',
    [Parameter(Mandatory = $true)][string]$Revision
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$Root = [IO.Path]::GetFullPath($Root)
$Results = Join-Path $Root 'results'
$Status = [ordered]@{ revision = $Revision; outcome = 'running'; targets = @() }

function Save-Status {
    $Status | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $Results 'status.json') -Encoding UTF8
}

function Assert-Hash([string]$Path, [string]$Expected) {
    if ((Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash -ine $Expected) {
        throw "SHA256 mismatch: $Path"
    }
}

function Download([string]$Url, [string]$Path, [string]$Hash) {
    if (-not (Test-Path $Path)) {
        Invoke-WebRequest $Url -OutFile $Path -UseBasicParsing -TimeoutSec 300
    }
    Assert-Hash $Path $Hash
}

function Disable-ServiceIfPresent([string]$Name) {
    if (Get-CimInstance Win32_Service -Filter "Name='$Name'") {
        Stop-Service -Name $Name -Force
        Set-Service -Name $Name -StartupType Disabled
    }
}

function Invoke-Check([string]$Python, [string]$Target, [string]$Name, [string]$NativeChecks) {
    $Stdout = Join-Path $Results "$Target-$Name.stdout.txt"
    $Stderr = Join-Path $Results "$Target-$Name.stderr.txt"
    $Script = Join-Path $Root "tests\$Name.py"
    $Watch = [Diagnostics.Stopwatch]::StartNew()
    $Arguments = @('-u', ('"{0}"' -f $Script))
    if ($Name -eq 'check_backend') {
        $Arguments += @('--native-checks', ('"{0}"' -f (Join-Path $Root $NativeChecks)))
    }
    $Process = Start-Process $Python -ArgumentList $Arguments `
        -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -PassThru
    # Cache the handle before waiting; otherwise PowerShell 5.1 can lose ExitCode.
    $null = $Process.Handle
    try {
        $TimedOut = -not $Process.WaitForExit(120000)
        if ($TimedOut) {
            Stop-Process -Id $Process.Id -Force
            $null = $Process.WaitForExit(5000)
            $ExitCode = -1
        }
        else {
            $Process.WaitForExit()
            $ExitCode = $Process.ExitCode
        }
    }
    finally { $Process.Dispose() }
    Get-Content $Stdout, $Stderr | ForEach-Object { Write-Host $_ }
    return [ordered]@{
        script = "$Name.py"; exit_code = $ExitCode; timed_out = $TimedOut
        elapsed_seconds = [Math]::Round($Watch.Elapsed.TotalSeconds, 3)
    }
}

New-Item -ItemType Directory -Path $Results -Force | Out-Null
Start-Transcript -Path (Join-Path $Results 'setup.log') | Out-Null
$ExitCode = 1
try {
    $Manifest = Get-Content (Join-Path $Root 'manifest.json') -Raw | ConvertFrom-Json
    $MetadataRevision = Invoke-RestMethod 'http://metadata.google.internal/computeMetadata/v1/instance/attributes/tdml-ci-revision' `
        -Headers @{ 'Metadata-Flavor' = 'Google' } -TimeoutSec 10
    if ($Manifest.revision -cne $Revision -or $MetadataRevision -cne $Revision -or -not $Manifest.targets -or -not $Manifest.checks) {
        throw 'The payload must contain targets and checks for the requested CI revision.'
    }
    foreach ($Entry in $Manifest.hashes.PSObject.Properties) {
        $Path = [IO.Path]::GetFullPath((Join-Path $Root $Entry.Name))
        if (-not $Path.StartsWith($Root.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) {
            throw "Payload path escapes its root: $($Entry.Name)"
        }
        Assert-Hash $Path $Entry.Value
    }

    # Service tuning is adapted from runner-images; see LICENSE.runner-images.
    $Policy = 'HKLM:\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate\AU'
    New-Item $Policy -Force | Out-Null
    New-ItemProperty $Policy -Name NoAutoUpdate -Value 1 -PropertyType DWord -Force | Out-Null
    foreach ($Service in @('wuauserv', 'SysMain', 'DiagTrack')) { Disable-ServiceIfPresent $Service }
    Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, BuildNumber, OSArchitecture | `
        ConvertTo-Json | Set-Content (Join-Path $Results 'os.json') -Encoding UTF8
    & nvidia-smi --query-gpu=name,driver_version,driver_model.current,memory.total --format=csv | Set-Content (Join-Path $Results 'gpu.txt') -Encoding UTF8
    if ($LASTEXITCODE -ne 0) { throw 'nvidia-smi failed.' }

    $Downloads = Join-Path $Root 'downloads'
    New-Item -ItemType Directory -Path $Downloads -Force | Out-Null
    $Runtimes = @{}
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    foreach ($Target in $Manifest.targets) {
        $Record = [ordered]@{ id = $Target.id; outcome = 'running'; tests = @() }
        $Status.targets += $Record
        Save-Status
        try {
            if (-not $Runtimes.ContainsKey($Target.abi)) {
                $Sdk = Join-Path $Downloads "python.$($Target.python).nupkg"
                Download $Target.python_url $Sdk $Target.python_sha256
                $Runtime = Join-Path $Root "runtimes\$($Target.abi)"
                [IO.Compression.ZipFile]::ExtractToDirectory($Sdk, $Runtime)
                $Runtimes[$Target.abi] = Join-Path $Runtime 'tools\python.exe'
            }
            $Venv = Join-Path $Root "venvs\$($Target.id)"
            & ($Runtimes[$Target.abi]) -m venv $Venv
            if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
            $Python = Join-Path $Venv 'Scripts\python.exe'
            $Wheel = Join-Path $Root $Target.wheel
            Assert-Hash $Wheel $Target.wheel_sha256
            & $Python -m pip install --no-compile --disable-pip-version-check -c (Join-Path $Root 'constraints.txt') $Target.torch_url $Wheel
            if ($LASTEXITCODE -ne 0) { throw 'Wheel installation failed.' }
            & $Python -m pip check
            if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed.' }
            $Code = 'import hashlib,json,pathlib,sys,torch,torch_directml,torch_directml_native; p=pathlib.Path(torch_directml_native.__file__).resolve(); print(json.dumps(dict(python=sys.version.split()[0],torch=torch.__version__.split("+")[0],native_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),native_path=str(p))))'
            # Stdin preserves Python source quotes in Windows PowerShell 5.1.
            $Installed = $Code | & $Python - | ConvertFrom-Json
            if ($LASTEXITCODE -ne 0 -or $Installed.python -ne $Target.python -or $Installed.torch -ne $Target.torch -or $Installed.native_sha256 -ne $Target.native_sha256) {
                throw 'The installed extension does not match the CPU-checked wheel.'
            }
            $Record['environment'] = $Installed
            foreach ($Name in $Manifest.checks) {
                $Record.tests += Invoke-Check $Python $Target.id $Name $Target.native_checks
                Save-Status
            }
            $Record.outcome = if (@($Record.tests | Where-Object { $_.exit_code -ne 0 }).Count) { 'failed' } else { 'passed' }
            Remove-Item $Venv -Recurse -Force
        }
        catch {
            $Record.outcome = 'failed'
            $Record['error'] = ($_ | Out-String)
            Write-Warning "Target $($Target.id) failed: $($Record.error)"
        }
        Save-Status
    }
    if (@($Status.targets | Where-Object { $_.outcome -ne 'passed' }).Count) {
        throw 'One or more GPU targets failed; inspect the saved reports.'
    }
    $Status.outcome = 'passed'
    $ExitCode = 0
}
catch {
    $Status.outcome = 'failed'
    $Status['error'] = ($_ | Out-String)
    Write-Warning $Status.error
}
finally {
    Save-Status
    Stop-Transcript | Out-Null
}
exit $ExitCode
