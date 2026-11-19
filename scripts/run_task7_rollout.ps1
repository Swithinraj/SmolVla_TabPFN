param(
    [int]$ActionSteps = 50,
    [string]$CheckpointPath = '',
    [string]$OutputName = ''
)
$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$env:HF_HOME = Join-Path $workspace '.cache\huggingface'
$env:XDG_CACHE_HOME = Join-Path $workspace '.cache\xdg'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'

$checkpoint = if ($CheckpointPath) { Join-Path $workspace $CheckpointPath } else { Join-Path $workspace 'outputs\task7_base_aug_20261116_1748\checkpoints\024356\pretrained_model' }
$resultName = if ($OutputName) { $OutputName } elseif ($ActionSteps -eq 50) { 'task7_libero_rollout' } else { "task7_libero_rollout_replan$ActionSteps" }
$output = Join-Path $workspace "outputs\$resultName"
$log = Join-Path $output 'eval.log'
$python = Join-Path $workspace '.venv\Scripts\python.exe'
$eval = Join-Path $workspace 'scripts\eval_with_video_fix.py'
if (-not (Test-Path $checkpoint)) { throw "Missing checkpoint: $checkpoint" }
if (-not (Test-Path $python)) { throw "Missing Python environment: $python" }
if (-not (Test-Path $eval)) { throw "Missing LeRobot evaluator: $eval" }

$args = @(
    "--policy.path=$checkpoint"
    '--policy.device=cuda'
    "--policy.n_action_steps=$ActionSteps"
    '--env.type=libero'
    '--env.task=libero_10'
    '--env.task_ids=[1]'
    '--env.fps=20'
    '--env.episode_length=520'
    '--env.observation_height=256'
    '--env.observation_width=256'
    '--env.server.endpoint=tcp/127.0.0.1:7448'
    '--rename_map={observation.images.image: observation.images.camera1, observation.images.image2: observation.images.camera2}'
    '--eval.n_episodes=1'
    '--eval.batch_size=1'
    '--eval.max_episode_steps=520'
    '--seed=1000'
    "--output_dir=$output"
)

New-Item -ItemType Directory -Path $output -Force | Out-Null
try {
    $ErrorActionPreference = 'Continue'
    & $python $eval @args *> $log
} finally {
    $ErrorActionPreference = 'Stop'
}
if ($LASTEXITCODE -ne 0) {
    Get-Content $log -Tail 50
    throw "LIBERO rollout failed with exit code $LASTEXITCODE; see $log"
}
Get-Content $log -Tail 18
