param(
    [int]$Steps = 20000,
    [int]$BatchSize = 4,
    [double]$LearningRate = 0.0001,
    [int]$SaveFreq = 5000,
    [int]$LogFreq = 50,
    [string]$OutputDir = ''
)
# Fine-tune smolvla_base on the real SO-101 pick-place dataset (lerobot/svla_so101_pickplace).
# Held-out episodes 5, 25, 45 are excluded from training.
$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot
$env:HF_HOME = Join-Path $PSScriptRoot '.cache\huggingface'
$env:HF_LEROBOT_HOME = Join-Path $PSScriptRoot '.cache\lerobot'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$held = 5, 25, 45
$train = (0..49) | Where-Object { $held -notcontains $_ }
$episodes = '[' + ($train -join ',') + ']'
if ($OutputDir) { $output = $OutputDir }
else { $output = Join-Path $PSScriptRoot ('outputs\so101_pickplace_' + (Get-Date -Format 'yyyyMMdd_HHmmss')) }
$model = Join-Path $PSScriptRoot 'models\smolvla_base'
$dataset = Join-Path $PSScriptRoot 'data\svla_so101_pickplace'
$renameMap = '{observation.images.up: observation.images.camera1, observation.images.side: observation.images.camera2}'
$transforms = '{brightness: {weight: 1, type: ColorJitter, kwargs: {brightness: [0.9, 1.1]}}, contrast: {weight: 1, type: ColorJitter, kwargs: {contrast: [0.9, 1.1]}}}'
$lrText = $LearningRate.ToString('G', [System.Globalization.CultureInfo]::InvariantCulture)
$decayText = ($LearningRate / 40).ToString('G', [System.Globalization.CultureInfo]::InvariantCulture)
$trainArgs = @(
    "--policy.path=$model",
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
    '--dataset.repo_id=lerobot/svla_so101_pickplace',
    "--dataset.root=$dataset",
    "--dataset.episodes=$episodes",
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
Write-Host "Training episodes: $($train.Count); held-out: $($held -join ','); output: $output; steps: $Steps; batch: $BatchSize"
& '.\.venv\Scripts\lerobot-train.exe' @trainArgs
exit $LASTEXITCODE
