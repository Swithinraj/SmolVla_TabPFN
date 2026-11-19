param(
    [string]$OutputDir = ''
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$env:HF_HOME = Join-Path $PSScriptRoot '.cache\huggingface'
$env:HF_LEROBOT_HOME = Join-Path $PSScriptRoot '.cache\lerobot'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'

$split = Get-Content -LiteralPath 'data\libero\task7_split.json' -Raw | ConvertFrom-Json
$info = Get-Content -LiteralPath 'data\libero\meta\info.json' -Raw | ConvertFrom-Json
$heldOut = @($split.test_episodes | ForEach-Object { [int]$_ })
if ($info.total_episodes -ne 1693 -or $heldOut.Count -ne 2) {
    throw 'Unexpected LIBERO dataset or held-out split'
}
$episodes = @(0..($info.total_episodes - 1) | Where-Object { $heldOut -notcontains $_ })
if ($episodes.Count -ne 1691) { throw 'Unexpected training episode count' }

$freeBytes = [System.IO.DriveInfo]::new('D').AvailableFreeSpace
if ($freeBytes -lt 105GB) { throw 'D: must have at least 105 GiB free before training' }

if ($OutputDir) { $output = $OutputDir }
else { $output = Join-Path $PSScriptRoot ('outputs\libero_paper_' + (Get-Date -Format 'yyyyMMdd_HHmmss')) }
$dataset = Join-Path $PSScriptRoot 'data\libero'
$episodeText = '[' + ($episodes -join ',') + ']'

# LeRobot 0.7 counts microbatches. 64 microbatches = one batch-64 optimizer update.
# 6,400,000 microbatches = the paper's 100,000 optimizer updates.
$microsteps = 6400000
$accumulation = 64
$trainArgs = @(
    '--policy.type=smolvla',
    '--policy.load_vlm_weights=true',
    '--policy.device=cuda',
    '--policy.train_expert_only=true',
    '--policy.freeze_vision_encoder=true',
    '--policy.n_action_steps=1',
    '--policy.chunk_size=50',
    '--policy.push_to_hub=false',
    '--policy.optimizer_lr=0.0001',
    '--policy.scheduler_warmup_steps=6400',
    '--policy.scheduler_decay_steps=6400000',
    '--policy.scheduler_decay_lr=0.0000025',
    '--dataset.repo_id=lerobot/libero',
    "--dataset.root=$dataset",
    "--dataset.episodes=$episodeText",
    '--dataset.video_backend=pyav',
    '--dataset.image_transforms.enable=false',
    "--output_dir=$output",
    "--steps=$microsteps",
    '--batch_size=1',
    "--accelerator.gradient_accumulation.steps=$accumulation",
    '--accelerator.mixed_precision=bf16',
    '--num_workers=0',
    '--env_eval_freq=0',
    '--log_freq=64',
    '--save_freq=64000',
    '--seed=1000',
    '--trackio.enable=false'
)

Write-Host "LIBERO paper run: $($episodes.Count) episodes; held out: $($heldOut -join ','); output: $output"
Write-Host "GPU batch 1 x gradient accumulation $accumulation = global batch 64"
Write-Host "Target: $microsteps microbatches = 100000 optimizer updates"
Write-Host "D: free: $([math]::Round($freeBytes / 1GB, 1)) GiB"

$ErrorActionPreference = 'Continue'
& '.\.venv\Scripts\lerobot-train.exe' @trainArgs
exit $LASTEXITCODE
