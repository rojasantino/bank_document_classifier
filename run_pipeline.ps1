param (
    [switch]$NoPretrained,
    [int]$Epochs = 8,
    [int]$PerClass = 60,
    [switch]$Cpu
)

$ExtraTrainArgs = ""
if ($NoPretrained) {
    $ExtraTrainArgs += " --no_pretrained"
}

if ($Cpu) {
    $env:FORCE_CPU = "1"
}

Write-Host "==> Step 0/4: Checking dependencies"
python -c "import cv2, torch, torchvision, fastapi" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "`nERROR: Required Python packages are missing."
    Write-Host "Please ensure you have activated your venv and installed dependencies:"
    Write-Host "    python -m pip install -r requirements.txt"
    exit 1
}

Write-Host "==> Step 1/4: Generating synthetic data (skip if data/raw/ already populated)"
$rawFiles = Get-ChildItem -Path "data/raw" -ErrorAction SilentlyContinue
if ($null -eq $rawFiles -or $rawFiles.Count -eq 0) {
    python data/generate_synthetic_data.py --per_class $PerClass
} else {
    Write-Host "data/raw already has files -- skipping synthetic generation."
}

Write-Host "==> Step 2/4: Preprocessing (denoise / deskew / crop)"
python src/data_preprocessing.py

Write-Host "==> Step 3/4: Training all models ($Epochs epochs each)"
$trainCmd = "python -m src.train --model all --epochs $Epochs$ExtraTrainArgs"
Invoke-Expression $trainCmd

Write-Host "==> Step 4/4: Evaluating and comparing all models"
python -m src.evaluate

Write-Host "`nPipeline complete."
Write-Host "  - Trained checkpoints : outputs/models/"
Write-Host "  - Comparison table    : outputs/reports/model_comparison.md"
Write-Host "  - Confusion matrices  : outputs/reports/confusion_matrix_*.png`n"
Write-Host "Start the API with:  uvicorn api.main:app --reload --port 8000"
