param(
    [ValidateSet('cuda', 'cpu')][string]$Device = 'cuda',
    [int]$Steps = 100,
    [int]$BatchSize = 1,
    [string]$OutputDir = '',
    [string]$EpisodeIds = '',
    [int]$LogFreq = 10,
    [int]$SaveFreq = 20000,
    [double]$LearningRate = 0.000001
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$env:HF_HOME = Join-Path $PSScriptRoot '.cache\huggingface'
$env:HF_LEROBOT_HOME = Join-Path $PSScriptRoot '.cache\lerobot'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$manifest = Get-Content -LiteralPath 'data\libero\subset.json' -Raw | ConvertFrom-Json
if ($EpisodeIds) {
    $selectedEpisodes = $EpisodeIds.Split(',') | ForEach-Object { [int]$_.Trim() }
    foreach ($episode in $selectedEpisodes) {
        if ($manifest.episodes -notcontains $episode) { throw "Episode $episode is not in the downloaded subset" }
    }
} else { $selectedEpisodes = $manifest.episodes }
$episodeList = '[' + (($selectedEpisodes | ForEach-Object { [string]$_ }) -join ',') + ']'
$model = Join-Path $PSScriptRoot 'models\smolvla_libero'
$dataset = Join-Path $PSScriptRoot 'data\libero'
if ($OutputDir) { $output = $OutputDir }
else { $output = Join-Path $PSScriptRoot ('outputs\smolvla_libero_subset_' + (Get-Date -Format 'yyyyMMdd_HHmmss')) }
$renameMap = '{observation.images.image: observation.images.camera1, observation.images.image2: observation.images.camera2}'
$trainArgs = @(
    "--policy.path=$model",
    "--policy.device=$Device",
    '--policy.train_expert_only=true',
    '--policy.push_to_hub=false',
    '--dataset.repo_id=lerobot/libero',
    "--dataset.root=$dataset",
    "--dataset.episodes=$episodeList",
    '--dataset.video_backend=pyav',
    "--rename_map=$renameMap",
    "--output_dir=$output",
    "--steps=$Steps",
    "--batch_size=$BatchSize",
    '--num_workers=0',
    '--env_eval_freq=0',
    "--log_freq=$LogFreq",
    "--save_freq=$SaveFreq"
)
if ($LearningRate -gt 0) {
    $lrText = $LearningRate.ToString('G', [System.Globalization.CultureInfo]::InvariantCulture)
    $decayText = ($LearningRate / 10).ToString('G', [System.Globalization.CultureInfo]::InvariantCulture)
    $trainArgs += "--policy.optimizer_lr=$lrText"
    $trainArgs += "--policy.scheduler_decay_lr=$decayText"
}
& '.\.venv\Scripts\lerobot-train.exe' @trainArgs
$trainingExitCode = $LASTEXITCODE
exit $trainingExitCode
