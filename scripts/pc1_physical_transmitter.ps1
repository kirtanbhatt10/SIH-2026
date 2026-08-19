# PC1 Physical Acoustic Transmitter
# Run on PC1 BEFORE or while PC2 receiver is capturing.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts/pc1_physical_transmitter.ps1 -Payload "SIH_PC1_PC2_TEST"
#   powershell -ExecutionPolicy Bypass -File scripts/pc1_physical_transmitter.ps1 -Payload "SIH_PC1_PC2_TEST" -Device 5

param(
    [string]$Payload = "SIH_PC1_PC2_TEST",
    [int]$Device = 5
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "========================================"
Write-Host "PC1 PHYSICAL ACOUSTIC TRANSMITTER"
Write-Host "========================================"
Write-Host "Ensure PC2 receiver is already capturing!"
Write-Host ""

$startTs = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Write-Host "[PC1] transmission_start=$startTs"

function Resolve-PythonWithSoundfile {
    $candidates = [System.Collections.Generic.List[string]]::new()

    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { $candidates.Add($cmd.Source) }

    if ($env:CONDA_PREFIX) {
        $candidates.Add((Join-Path $env:CONDA_PREFIX "python.exe"))
    }

    if ($env:CONDA_EXE) {
        $condaRoot = Split-Path (Split-Path $env:CONDA_EXE -Parent) -Parent
        Get-ChildItem (Join-Path $condaRoot "envs\*\python.exe") -ErrorAction SilentlyContinue |
            ForEach-Object { $candidates.Add($_.FullName) }
    }

    foreach ($exe in (where.exe python 2>$null)) {
        $candidates.Add($exe)
    }

    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if (-not (Test-Path $candidate)) { continue }
        try {
            $null = & $candidate -c "import soundfile" 2>&1
            if ($LASTEXITCODE -eq 0) { return $candidate }
        } catch {
            continue
        }
    }

    return "python"
}

$Python = Resolve-PythonWithSoundfile

$args_list = @(
    "scripts/pc1_transmitter_test.py",
    "--payload", $Payload,
    "--play",
    "--device", "$Device"
)

& $Python @args_list
$exitCode = $LASTEXITCODE

$endTs = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Write-Host "[PC1] transmission_end=$endTs"

if ($exitCode -eq 0) {
    Write-Host "[PC1] PHYSICAL TRANSMIT: SPEAKER PLAYED"
} else {
    Write-Host "[PC1] PHYSICAL TRANSMIT: FAILED"
}

exit $exitCode
