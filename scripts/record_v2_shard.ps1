param([string]$Shard, [int]$Seed0, [int]$Episodes)
Set-Location 'd:\smolVLA_Enhance'
$env:HF_LEROBOT_HOME = 'd:\smolVLA_Enhance\.cache\lerobot'; $env:HF_HOME = 'd:\smolVLA_Enhance\.cache\huggingface'
$env:TABPFN_MODEL_CACHE_DIR = 'd:\smolVLA_Enhance\.cache\tabpfn'; $env:TABPFN_NO_BROWSER = '1'; $env:PYTHONUNBUFFERED = '1'; $env:HF_HUB_OFFLINE = '0'
& .\.venv\Scripts\python.exe challenge\sim\record_catch_dataset_v2.py --episodes $Episodes --seed0 $Seed0 --root "data\catch_v2_$Shard" --repo-id "local/catch_v2_$Shard" --overwrite
