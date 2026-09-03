# PC1 → PC2 test orchestration (single-machine FILE REPLAY mode).
# Physical acoustic test requires two PCs — use pc1_physical_transmitter.ps1
# and pc2_physical_receiver.ps1 independently.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts/run_pc1_pc2_test.ps1

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$Payload = "SIH_PC1_PC2_TEST"
$BackendUrl = "http://127.0.0.1:8000"
$Python = "python"

Write-Host "========================================"
Write-Host "PC1 -> PC2 TEST RUNNER (FILE REPLAY)"
Write-Host "========================================"
Write-Host "Project root: $ProjectRoot"
Write-Host "Mode: FILE REPLAY (not physical acoustic)"
Write-Host ""

# 1. Check Python
try {
    $pyVer = & $Python --version 2>&1
    Write-Host "[CHECK] Python: $pyVer"
} catch {
    Write-Host "[FAIL] Python not found"
    exit 1
}

# 2. Check dependencies
& $Python -c "import fastapi, uvicorn, numpy, scipy, httpx, sounddevice" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Missing dependencies. Run: pip install -r requirements.txt"
    exit 1
}
Write-Host "[CHECK] Dependencies: OK"

# 3. Check project structure
$required = @(
    "backend/main.py",
    "backend/services/payload_service.py",
    "ai/dsp/dsp_api.py",
    "integration/backend_client.py",
    "scripts/pc1_transmitter_test.py",
    "scripts/pc2_receiver_test.py",
    "scripts/run_file_replay_e2e.py"
)
foreach ($path in $required) {
    if (-not (Test-Path $path)) {
        Write-Host "[FAIL] Missing: $path"
        exit 1
    }
}
Write-Host "[CHECK] Project structure: OK"

# 4. Start backend if not running
$backendRunning = $false
try {
    $r = Invoke-WebRequest -Uri "$BackendUrl/api/system-status" -UseBasicParsing -TimeoutSec 3
    if ($r.StatusCode -eq 200) { $backendRunning = $true }
} catch { }

if (-not $backendRunning) {
    Write-Host "[START] Launching FastAPI backend..."
    $backendJob = Start-Job -ScriptBlock {
        param($root)
        Set-Location $root
        python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
    } -ArgumentList $ProjectRoot

    $deadline = (Get-Date).AddSeconds(30)
    while ((Get-Date) -lt $deadline) {
        try {
            $r = Invoke-WebRequest -Uri "$BackendUrl/api/system-status" -UseBasicParsing -TimeoutSec 2
            if ($r.StatusCode -eq 200) {
                Write-Host "[START] Backend ready"
                break
            }
        } catch {
            Start-Sleep -Seconds 1
        }
    }
    try {
        $r = Invoke-WebRequest -Uri "$BackendUrl/api/system-status" -UseBasicParsing -TimeoutSec 2
        if ($r.StatusCode -ne 200) { throw "not ready" }
    } catch {
        Write-Host "[FAIL] Backend did not start within 30s"
        if ($backendJob) { Stop-Job $backendJob; Remove-Job $backendJob }
        exit 1
    }
} else {
    Write-Host "[CHECK] Backend already running"
    $backendJob = $null
}

# 5. PC1 generate WAV
Write-Host ""
Write-Host "[PC1] Generating acoustic payload..."
& $Python scripts/pc1_transmitter_test.py --payload $Payload
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] PC1 transmitter failed"
    if ($backendJob) { Stop-Job $backendJob; Remove-Job $backendJob }
    exit 1
}

# Find latest WAV in generated_payloads
$wavDir = Join-Path $ProjectRoot "generated_payloads"
$latestWav = Get-ChildItem -Path $wavDir -Filter "*.wav" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if (-not $latestWav) {
    Write-Host "[FAIL] No WAV found in generated_payloads/"
    if ($backendJob) { Stop-Job $backendJob; Remove-Job $backendJob }
    exit 1
}
Write-Host "[PC1] WAV: $($latestWav.FullName)"

# 6. PC2 file replay + AI + backend
Write-Host ""
Write-Host "[PC2] File replay E2E..."
& $Python scripts/run_file_replay_e2e.py --wav $latestWav.FullName --backend-url $BackendUrl
$e2eExit = $LASTEXITCODE

# Cleanup backend job if we started it
if ($backendJob) {
    Stop-Job $backendJob -ErrorAction SilentlyContinue
    Remove-Job $backendJob -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "========================================"
Write-Host "ORCHESTRATION COMPLETE"
Write-Host "========================================"
Write-Host "Mode: FILE REPLAY"
Write-Host "Physical PC1->PC2: NOT EXECUTED (requires 2 PCs)"
if ($e2eExit -eq 0) {
    Write-Host "File replay E2E: PASS"
    exit 0
} else {
    Write-Host "File replay E2E: FAIL"
    exit 1
}
