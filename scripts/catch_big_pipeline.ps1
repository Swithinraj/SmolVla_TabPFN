Set-Location (Split-Path $PSScriptRoot -Parent)
$env:HF_HOME = (Join-Path (Get-Location) '.cache\huggingface'); $env:HF_LEROBOT_HOME = (Join-Path (Get-Location) '.cache\lerobot')
$env:TABPFN_MODEL_CACHE_DIR = (Join-Path (Get-Location) '.cache\tabpfn'); $env:TABPFN_NO_BROWSER = '1'
$out = "outputs\catch_big"
$init = "outputs\catch_pilot_s2\checkpoints\003000\pretrained_model"
Write-Host "=== big train $(Get-Date -Format o)"
powershell -NoProfile -ExecutionPolicy Bypass -File .\train_ball.ps1 -Init $init -Dataset data\so101_catch_v2 -Steps 12000 -BatchSize 4 -SaveFreq 4000 -OutputDir $out
Write-Host "=== big eval $(Get-Date -Format o)"
$env:HF_HUB_OFFLINE = '0'
& .\.venv\Scripts\python.exe challenge\sim\eval_catch.py --checkpoints "big=$out\checkpoints\012000\pretrained_model" --splits hard,indist --n 24 --modes oracle,tabpfn,none --variants smooth,smoother,safe --realism 0,1 --safety --gifs 2 --out "$out\eval"
Write-Host "=== big done $(Get-Date -Format o)"
