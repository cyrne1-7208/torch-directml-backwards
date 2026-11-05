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
Start-Transcript -Path (Join-Path $Results 'build.log') | Out-Null

try {
    # Initialize the x64 toolchain directly in the current PowerShell process.
    $VsWhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    $VisualStudio = & $VsWhere -latest -products '*' -version '[17.1,)' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $VisualStudio) {
        throw 'Visual Studio 2022 17.1 or newer with the x64 C++ tools is required.'
    }
    $DevShell = Join-Path $VisualStudio 'Common7\Tools\Launch-VsDevShell.ps1'
    & $DevShell -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
    $env:PATH = (Join-Path $env:ProgramFiles 'LLVM\bin') + ';' + $env:PATH
    & clang-cl --version
    if ($LASTEXITCODE -ne 0) { throw 'clang-cl is required.' }
    & lld-link --version
    if ($LASTEXITCODE -ne 0) { throw 'lld-link is required.' }
    Write-Host "VCToolsVersion=$env:VCToolsVersion WindowsSDKVersion=$env:WindowsSDKVersion"

    $Matrix = & $Python tools/build.py matrix | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0) { throw 'Could not read the build matrix.' }
    $Targets = @($Matrix.include | Where-Object { -not $Abis.Count -or $_.abi -in $Abis })
    if (-not $Targets.Count) { throw 'No supported targets match the selected Python ABIs.' }
    foreach ($Target in $Targets) {
        $Id = "$($Target.abi)-torch$($Target.torch)-win_amd64"
        $Output = Join-Path $Root ".work\ci\artifacts\$Id"
        & $Python tools/build.py build $Target.abi $Target.torch --revision $Revision --output-dir $Output
        if ($LASTEXITCODE -ne 0) { throw "Build failed: $Id" }
        Write-Host "PASS build: $Id"
    }
}
finally {
    Stop-Transcript | Out-Null
}
