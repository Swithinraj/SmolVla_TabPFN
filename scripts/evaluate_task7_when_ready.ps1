param(
    [int]$TrainingPid = 29924,
    [string]$RunName = 'task7_base_aug_20261116_1748'
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$run = Join-Path (Get-Location) ('outputs\' + $RunName)
$firstModel = Join-Path $run 'checkpoints\012178\pretrained_model'
$finalModel = Join-Path $run 'checkpoints\024356\pretrained_model'
$status = Join-Path $run 'evaluation_status.txt'
try {
    Wait-Process -Id $TrainingPid -ErrorAction SilentlyContinue
    foreach ($model in @($firstModel, $finalModel)) {
        if (-not (Test-Path -LiteralPath (Join-Path $model 'model.safetensors'))) {
            throw "Checkpoint was not saved: $model; inspect the training logs in .tmp."
        }
    }
    $env:HF_HOME = Join-Path (Get-Location) '.cache\huggingface'
    $env:HF_LEROBOT_HOME = Join-Path (Get-Location) '.cache\lerobot'
    $env:HF_HUB_OFFLINE = '1'
    $env:TRANSFORMERS_OFFLINE = '1'
    # Windows Python packages can emit harmless stderr warnings. Check exit codes instead.
    $ErrorActionPreference = 'Continue'
    & '.\.venv\Scripts\python.exe' scripts\evaluate_tuning.py `
        --tuned $firstModel `
        --baseline-dir outputs\task7_base_baseline\pretrained_model `
        --test-episodes 53,347 `
        --samples-per-episode 16 2>&1 | Tee-Object -FilePath (Join-Path $run 'evaluation.log')
    if ($LASTEXITCODE -ne 0) { throw "Evaluation exited with code $LASTEXITCODE" }
    & '.\.venv\Scripts\python.exe' scripts\evaluate_tuning.py `
        --tuned $finalModel `
        --baseline-report (Join-Path $run 'evaluation_32_012178.json') `
        --test-episodes 53,347 `
        --samples-per-episode 16 2>&1 | Tee-Object -FilePath (Join-Path $run 'evaluation.log') -Append
    if ($LASTEXITCODE -ne 0) { throw "Final evaluation exited with code $LASTEXITCODE" }
    Set-Content -LiteralPath $status -Value 'completed' -Encoding UTF8
} catch {
    Set-Content -LiteralPath $status -Value ('failed: ' + $_.Exception.Message) -Encoding UTF8
    throw
}
