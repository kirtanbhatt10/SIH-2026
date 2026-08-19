# PC2 Physical Acoustic Receiver
# Start this on PC2 BEFORE PC1 plays through speaker.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts/pc2_physical_receiver.ps1
#   powershell -ExecutionPolicy Bypass -File scripts/pc2_physical_receiver.ps1 -Duration 20

param(
    [double]$Duration = 20.0,
    [int]$Device = -1,
    [string]$BackendUrl = "http://127.0.0.1:8000"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "========================================"
Write-Host "PC2 PHYSICAL ACOUSTIC RECEIVER"
Write-Host "========================================"
Write-Host "Start PC1 transmitter AFTER this begins capturing."
Write-Host "Duration: ${Duration}s"
Write-Host ""

$args_list = @(
    "scripts/pc2_receiver_test.py",
    "--mode", "mic",
    "--duration", "$Duration",
    "--backend-url", $BackendUrl
)

if ($Device -ge 0) {
    $args_list += @("--device", "$Device")
}

$startTs = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Write-Host "[PC2] receiver_start=$startTs"

python @args_list
$exitCode = $LASTEXITCODE

$endTs = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Write-Host "[PC2] receiver_end=$endTs"

if ($exitCode -eq 0) {
    Write-Host "[PC2] PHYSICAL RECEIVE + AI + BACKEND: PASS"
} else {
    Write-Host "[PC2] PHYSICAL RECEIVE + AI + BACKEND: FAIL"
    Write-Host "Note: Physical test may fail if PC1 did not play during capture window."
}

exit $exitCode
