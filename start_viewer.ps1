param([int]$Limit = 8)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { $python = 'python' }
$extraEpisodes = @()
if (Test-Path -LiteralPath 'data\libero\task7_split.json') {
    $split = Get-Content -LiteralPath 'data\libero\task7_split.json' -Raw | ConvertFrom-Json
    $extraEpisodes = @($split.test_episodes | ForEach-Object { [string]$_ })
}
if ($extraEpisodes.Count -gt 0) {
    & $python scripts\prepare_viewer.py --limit $Limit --episode-ids ($extraEpisodes -join ',')
} else {
    & $python scripts\prepare_viewer.py --limit $Limit
}
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python scripts\serve_viewer.py
