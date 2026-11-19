param(
    # Initial weights. Default: the stage-1 SO-101 pick-place checkpoint (set -Init to smolvla_base to skip stage 1).
    [string]$Init = '',
    [string]$Dataset = 'data\so101_ball_train',
    [int]$Steps = 15000,
    [int]$BatchSize = 4,
    [double]$LearningRate = 0.0001,
    [int]$SaveFreq = 5000,
    [int]$LogFreq = 50,
    [string]$OutputDir = ''
)
# Stage 2: fine-tune on simulated SO-101 red-ball blocking demonstrations (goal marker drawn in the image).
$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot
$env:HF_HOME = Join-Path $PSScriptRoot '.cache\huggingface'
$env:HF_LEROBOT_HOME = Join-Path $PSScriptRoot '.cache\lerobot'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
if (-not $Init) { throw 'Pass -Init <path to pretrained_model dir>' }
if ($OutputDir) { $output = $OutputDir }
else { $output = Join-Path $PSScriptRoot ('outputs\so101_ball_' + (Get-Date -Format 'yyyyMMdd_HHmmss')) }
$data = Join-Path $PSScriptRoot $Dataset
# top camera -> camera1, front camera -> camera2 (LeRobot SmolVLA key names)
$renameMap = '{observation.images.top: observation.images.camera1, observation.images.front: observation.images.camera2}'
# Photometric jitter only (sim-to-real: broad lighting range; geometric shifts would move the marker relative to the table).
$transforms = '{brightness: {weight: 1, type: ColorJitter, kwargs: {brightness: [0.8, 1.2]}}, contrast: {weight: 1, type: ColorJitter, kwargs: {contrast: [0.8, 1.2]}}}'
$lrText = $LearningRate.ToString('G', [System.Globalization.CultureInfo]::InvariantCulture)
$decayText = ($LearningRate / 40).ToString('G', [System.Globalization.CultureInfo]::InvariantCulture)
$trainArgs = @(
    "--policy.path=$Init",
    '--policy.device=cuda',
    '--policy.input_features=null',
    '--policy.output_features=null',
    '--policy.train_expert_only=true',
    '--policy.freeze_vision_encoder=true',
    '--policy.push_to_hub=false',
    "--policy.optimizer_lr=$lrText",
    '--policy.scheduler_warmup_steps=200',
    "--policy.scheduler_decay_steps=$Steps",
    "--policy.scheduler_decay_lr=$decayText",
    '--dataset.repo_id=local/so101_ball_train',
    "--dataset.root=$data",
    '--dataset.video_backend=pyav',
    '--dataset.image_transforms.enable=true',
    '--dataset.image_transforms.backend=dataloader',
    '--dataset.image_transforms.max_num_transforms=1',
    "--dataset.image_transforms.tfs=$transforms",
    "--rename_map=$renameMap",
    "--output_dir=$output",
    "--steps=$Steps",
    "--batch_size=$BatchSize",
    '--num_workers=0',
    '--env_eval_freq=0',
    "--log_freq=$LogFreq",
    "--save_freq=$SaveFreq"
)
Write-Host "Init: $Init; dataset: $data; output: $output; steps: $Steps; batch: $BatchSize"
& '.\.venv\Scripts\lerobot-train.exe' @trainArgs
exit $LASTEXITCODE
