# Wait for the two randomized recording shards, merge, then run the PILOT (short training + evaluation).
$ErrorActionPreference = 'Continue'
Set-Location 'd:\smolVLA_Enhance'
$env:HF_LEROBOT_HOME = 'd:\smolVLA_Enhance\.cache\lerobot'; $env:HF_HOME = 'd:\smolVLA_Enhance\.cache\huggingface'
$env:TABPFN_MODEL_CACHE_DIR = 'd:\smolVLA_Enhance\.cache\tabpfn'; $env:TABPFN_NO_BROWSER = '1'; $env:PYTHONUNBUFFERED = '1'
Write-Host "=== waiting for shards $(Get-Date -Format o)"
while (-not ((Test-Path data\catch_v2_A\sidecar.json) -and (Test-Path data\catch_v2_B\sidecar.json))) { Start-Sleep 30 }
Write-Host "=== aggregate $(Get-Date -Format o)"
$env:HF_HUB_OFFLINE = '1'
& .\.venv\Scripts\python.exe scripts\aggregate_v2.py
$s2 = (Get-ChildItem -Directory outputs\so101_ball_2* | Where-Object { $_.Name -match '^so101_ball_\d{8}_\d{6}$' } | Sort-Object Name | Select-Object -First 1).FullName
$init = "$s2\checkpoints\015000\pretrained_model"
$out = 'd:\smolVLA_Enhance\outputs\catch_pilot_s2'
Write-Host "=== pilot train $(Get-Date -Format o)"
powershell -NoProfile -ExecutionPolicy Bypass -File .\train_ball.ps1 -Init $init -Dataset data\so101_catch_v2 -Steps 3000 -BatchSize 4 -SaveFreq 3000 -OutputDir $out
Write-Host "=== pilot eval $(Get-Date -Format o)"
$env:HF_HUB_OFFLINE = '0'
& .\.venv\Scripts\python.exe challenge\sim\eval_catch.py --checkpoints "pilot=$out\checkpoints\003000\pretrained_model" --splits hard,indist --n 16 --modes oracle,tabpfn --variants raw,smooth --realism 0,1 --references --gifs 2 --out "$out\eval"
Write-Host "=== pilot done $(Get-Date -Format o)"
