# Catch task: record forecast-conditioned catch-and-hold demos -> fine-tune from the stage-2 checkpoint.
$ErrorActionPreference = 'Continue'
Set-Location 'd:\smolVLA_Enhance'
$env:HF_LEROBOT_HOME = 'd:\smolVLA_Enhance\.cache\lerobot'
$env:HF_HOME = 'd:\smolVLA_Enhance\.cache\huggingface'
$env:TABPFN_MODEL_CACHE_DIR = 'd:\smolVLA_Enhance\.cache\tabpfn'
$env:TABPFN_NO_BROWSER = '1'
$env:PYTHONUNBUFFERED = '1'
$s2 = (Get-ChildItem -Directory outputs\so101_ball_2* | Where-Object { $_.Name -match '^so101_ball_\d{8}_\d{6}$' } | Sort-Object Name | Select-Object -First 1).FullName
$init = "$s2\checkpoints\015000\pretrained_model"
$out = 'd:\smolVLA_Enhance\outputs\so101_catch'
Write-Host "init: $init"
Write-Host "=== record $(Get-Date -Format o)"
$env:HF_HUB_OFFLINE = '0'
& .\.venv\Scripts\python.exe challenge\sim\record_catch_dataset.py --episodes 360 --root data\so101_catch_train --overwrite
Write-Host "=== train $(Get-Date -Format o)"
$env:HF_HUB_OFFLINE = '1'
powershell -NoProfile -ExecutionPolicy Bypass -File .\train_ball.ps1 -Init $init -Dataset data\so101_catch_train -Steps 12000 -BatchSize 4 -SaveFreq 4000 -OutputDir $out
Write-Host "=== done $(Get-Date -Format o)"
