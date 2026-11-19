# Stage 3: record forecast-conditioned demos -> fine-tune from the stage-2 final checkpoint -> compare old vs new.
$ErrorActionPreference = 'Continue'
Set-Location 'd:\smolVLA_Enhance'
$env:HF_LEROBOT_HOME = 'd:\smolVLA_Enhance\.cache\lerobot'
$env:HF_HOME = 'd:\smolVLA_Enhance\.cache\huggingface'
$env:TABPFN_MODEL_CACHE_DIR = 'd:\smolVLA_Enhance\.cache\tabpfn'
$env:TABPFN_NO_BROWSER = '1'
$env:PYTHONUNBUFFERED = '1'
$s2 = (Get-ChildItem -Directory outputs\so101_ball_2* | Where-Object { $_.Name -match '^so101_ball_\d{8}_\d{6}$' } | Sort-Object Name | Select-Object -First 1).FullName
$init = "$s2\checkpoints\015000\pretrained_model"
$out = 'd:\smolVLA_Enhance\outputs\so101_ball_stage3'
Write-Host "stage2 run: $s2"
Write-Host "=== record $(Get-Date -Format o)"
$env:HF_HUB_OFFLINE = '0'
& .\.venv\Scripts\python.exe challenge\sim\record_forecast_dataset.py --episodes 400 --root data\so101_ball_fc_train --overwrite
Write-Host "=== train $(Get-Date -Format o)"
$env:HF_HUB_OFFLINE = '1'
powershell -NoProfile -ExecutionPolicy Bypass -File .\train_ball.ps1 -Init $init -Dataset data\so101_ball_fc_train -Steps 10000 -BatchSize 4 -SaveFreq 5000 -OutputDir $out
Write-Host "=== eval $(Get-Date -Format o)"
$env:HF_HUB_OFFLINE = '0'
$new = "$out\checkpoints\010000\pretrained_model"
& .\.venv\Scripts\python.exe challenge\sim\eval_closed_loop.py --checkpoints "stage2=$init" "stage3=$new" --splits hard,indist --n 50 --modes tabpfn,cv,oracle,none --references --gifs 2 --out "$out\closed_loop_compare"
Write-Host "finished $(Get-Date -Format o)"
