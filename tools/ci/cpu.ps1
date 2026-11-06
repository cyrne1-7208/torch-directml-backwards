param(
    [Parameter(Mandatory = $true)][string]$Revision,
    [string[]]$Abis = @()
)

$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $Root
$Python = Join-Path $Root '.venv\Scripts\python.exe'
$Results = Join-Path $Root '.work\ci\results'
New-Item -ItemType Directory -Path $Results -Force | Out-Null
Start-Transcript -Path (Join-Path $Results 'cpu.log') | Out-Null

try {
    $Matrix = & $Python tools/build.py matrix | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0) { throw 'Could not read the build matrix.' }
    $Targets = @($Matrix.include | Where-Object { -not $Abis.Count -or $_.abi -in $Abis })
    if (-not $Targets.Count) { throw 'No supported targets match the selected Python ABIs.' }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    foreach ($Target in $Targets) {
        $Id = "$($Target.abi)-torch$($Target.torch)-win_amd64"
        $Output = Join-Path $Root ".work\ci\artifacts\$Id"
        $ManifestPath = Join-Path $Output 'manifest.json'
        $Manifest = Get-Content $ManifestPath -Raw | ConvertFrom-Json
        if ($Manifest.revision -cne $Revision) { throw "Build revision differs: $Id" }
        $Runtime = Join-Path $Root ".work\ci\runtimes\$($Target.abi)"
        if (-not (Test-Path $Runtime)) {
            $Sdk = Join-Path $Root "reference\cpython\$($Target.python)\python.$($Target.python).nupkg"
            New-Item -ItemType Directory -Path $Runtime -Force | Out-Null
            [System.IO.Compression.ZipFile]::ExtractToDirectory($Sdk, $Runtime)
        }
        $Venv = Join-Path $Root ".work\ci\venvs\$Id"
        & (Join-Path $Runtime 'tools\python.exe') -m venv $Venv
        if ($LASTEXITCODE -ne 0) { throw "Could not create the test environment: $Id" }
        $TestPython = Join-Path $Venv 'Scripts\python.exe'
        $Wheel = Join-Path $Output $Manifest.wheel
        & $TestPython -m pip install --no-compile --disable-pip-version-check -c (Join-Path $Output 'constraints.txt') $Manifest.torch_url $Wheel
        if ($LASTEXITCODE -ne 0) { throw "Wheel installation failed: $Id" }
        & $TestPython -m pip check
        if ($LASTEXITCODE -ne 0) { throw "Dependency check failed: $Id" }
        # Check the installed extension rather than importing a source-tree copy.
        $Code = 'import hashlib,json,pathlib,sys,torch,torch_directml,torch_directml_native; p=pathlib.Path(torch_directml_native.__file__).resolve(); print(json.dumps(dict(python=sys.version.split()[0],torch=torch.__version__.split("+")[0],native_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),native_path=str(p))))'
        # Stdin preserves Python source quotes in Windows PowerShell 5.1.
        $Installed = $Code | & $TestPython - | ConvertFrom-Json
        if ($LASTEXITCODE -ne 0 -or $Installed.python -ne $Target.python -or $Installed.torch -ne $Target.torch -or $Installed.native_sha256 -ne $Manifest.native_sha256) {
            throw "Installed wheel does not match its build manifest: $Id"
        }
        $Installed | ConvertTo-Json | Set-Content (Join-Path $Results "$Id.json") -Encoding UTF8
        & $TestPython -u (Join-Path $Root 'tests\check_backend.py') --cpu-only --native-checks (Join-Path $Output 'native_checks.dll')
        if ($LASTEXITCODE -ne 0) { throw "CPU native check failed: $Id" }
        & $TestPython -u (Join-Path $Root 'tests\check_amp.py') --cpu-only
        if ($LASTEXITCODE -ne 0) { throw "CPU AMP check failed: $Id" }
        $Manifest | Add-Member -NotePropertyName cpu_checked -NotePropertyValue $true -Force
        $Manifest | ConvertTo-Json -Depth 5 | Set-Content $ManifestPath -Encoding UTF8
        Write-Host "PASS wheel installation and CPU checks: $Id"
        Remove-Item $Venv -Recurse -Force
    }
}
finally {
    Stop-Transcript | Out-Null
}
