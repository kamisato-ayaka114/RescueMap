$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$colmapSource = "E:\colmap"
$releaseRoot = Join-Path $projectRoot "dist\RescueMap"
$colmapTarget = Join-Path $releaseRoot "engines\colmap"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment not found: $python"
}
if (-not (Test-Path -LiteralPath (Join-Path $colmapSource "COLMAP.bat"))) {
    throw "COLMAP not found: $colmapSource"
}

& $python -m PyInstaller `
    --noconfirm `
    --clean `
    --windowed `
    --name RescueMap `
    --exclude-module torch `
    --exclude-module PyQt5 `
    --exclude-module PyQt6 `
    --exclude-module PySide2 `
    --exclude-module trame `
    --exclude-module trame_pyvista `
    --exclude-module IPython `
    --exclude-module jupyter `
    --exclude-module notebook `
    --exclude-module pandas `
    --exclude-module botocore `
    --exclude-module boto3 `
    --exclude-module zmq `
    --exclude-module tkinter `
    --exclude-module _tkinter `
    --exclude-module lxml `
    --exclude-module fsspec `
    (Join-Path $projectRoot "main.py")
if ($LASTEXITCODE -ne 0) {
    throw "RescueMap packaging failed."
}

New-Item -ItemType Directory -Path $colmapTarget -Force | Out-Null
robocopy $colmapSource $colmapTarget /E /R:2 /W:1 /NFL /NDL /NJH /NJS
if ($LASTEXITCODE -ge 8) {
    throw "COLMAP copy failed. Robocopy exit code: $LASTEXITCODE"
}

$launcher = Join-Path $releaseRoot "Start RescueMap.bat"
Set-Content -LiteralPath $launcher -Encoding ASCII -Value @"
@echo off
cd /d "%~dp0"
start "" "RescueMap.exe"
"@

Write-Host "Release created: $releaseRoot"
