# Waits for the stage-2 ball training to finish, then evaluates every checkpoint (offline + closed loop).
$ErrorActionPreference = 'Continue'
Set-Location 'd:\smolVLA_Enhance'
$info = Get-Content outputs\run_logs\active_ball.json -Raw | ConvertFrom-Json
Write-Host "waiting for training pid $($info.pid)"
while (Get-Process -Id $info.pid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 30 }
Write-Host "training exited at $(Get-Date -Format o)"
$run = $info.run
$ck = Get-ChildItem "$run\checkpoints" -Directory | Where-Object { $_.Name -match '^\d+$' } | Sort-Object Name
Write-Host "checkpoints: $($ck.Name -join ', ')"
$models = $ck | ForEach-Object { "step$([int]$_.Name)=$($_.FullName)\pretrained_model" }
$eps = (0..59) -join ','
Write-Host '=== offline held-out eval'
& .\.venv\Scripts\python.exe scripts\evaluate_so101.py --dataset data\so101_ball_heldout --repo local/so101_ball_heldout --episodes $eps --rename top:camera1,front:camera2 --samples 3 --out "$run\heldout_offline_eval.json" --models @models
Write-Host '=== closed-loop eval'
& .\.venv\Scripts\python.exe challenge\sim\eval_closed_loop.py --checkpoints @models --n 30 --n-action-steps 5 --references --out "$run\closed_loop"
Write-Host "finished at $(Get-Date -Format o)"
