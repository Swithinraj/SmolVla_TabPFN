param(
    [int]$Steps = 24356,
    [double]$LearningRate = 0.00001,
    [int]$SaveFreq = 12178,
    [int]$LogFreq = 100,
    [string]$OutputDir = ''
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$env:HF_HOME = Join-Path $PSScriptRoot '.cache\huggingface'
$env:HF_LEROBOT_HOME = Join-Path $PSScriptRoot '.cache\lerobot'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$split = Get-Content -LiteralPath 'data\libero\task7_split.json' -Raw | ConvertFrom-Json
if ($split.train_episodes.Count -ne 47 -or $split.test_episodes.Count -ne 2) { throw 'Unexpected task split' }
foreach ($episode in $split.test_episodes) {
    if ($split.train_episodes -contains $episode) { throw "Test episode $episode is in training set" }
}
if ([System.IO.DriveInfo]::new('D').AvailableFreeSpace -lt 105000000000) {
    throw 'Less than 105 GB free on D: before training'
}
$episodes = '[' + (($split.train_episodes | ForEach-Object { [string]$_ }) -join ',') + ']'
if ($OutputDir) { $output = $OutputDir }
else { $output = Join-Path $PSScriptRoot ('outputs\task7_base_aug_' + (Get-Date -Format 'yyyyMMdd_HHmmss')) }
$model = Join-Path $PSScriptRoot 'models\smolvla_base'
$dataset = Join-Path $PSScriptRoot 'data\libero'
$renameMap = '{observation.images.image: observation.images.camera1, observation.images.image2: observation.images.camera2}'
# The paper uses 512x512 inputs; camera key mapping is for LeRobot compatibility.
# LeRobot's optional training-time photometric transforms are kept mild; no geometric shifts.
$transforms = '{brightness: {weight: 1, type: ColorJitter, kwargs: {brightness: [0.9, 1.1]}}, contrast: {weight: 1, type: ColorJitter, kwargs: {contrast: [0.9, 1.1]}}}'
$lrText = $LearningRate.ToString('G', [System.Globalization.CultureInfo]::InvariantCulture)
$decayText = ($LearningRate / 10).ToString('G', [System.Globalization.CultureInfo]::InvariantCulture)
$trainArgs = @(
    "--policy.path=$model",
    '--policy.device=cuda',
    '--policy.input_features=null',
    '--policy.output_features=null',
    '--policy.train_expert_only=true',
    '--policy.freeze_vision_encoder=true',
    '--policy.push_to_hub=false',
    "--policy.optimizer_lr=$lrText",
    '--policy.scheduler_warmup_steps=100',
    "--policy.scheduler_decay_steps=$Steps",
    "--policy.scheduler_decay_lr=$decayText",
    '--dataset.repo_id=lerobot/libero',
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
    '--batch_size=1',
    '--num_workers=0',
    '--env_eval_freq=0',
    "--log_freq=$LogFreq",
    "--save_freq=$SaveFreq"
)
Write-Host "Task: $($split.task); training episodes: $($split.train_episodes.Count); held-out episodes: $($split.test_episodes -join ',')"
Write-Host "Training frames: $($split.train_frames); output: $output; steps: $Steps"
& '.\.venv\Scripts\lerobot-train.exe' @trainArgs
exit $LASTEXITCODE
