# LIBERO SmolVLA paper-style run

Started 2026-11-17 at 15:33 Europe/Berlin from `train_libero_paper.ps1`. **Cancelled at 15:44 at the user's request.** The last logged progress was microstep 960 (15 optimizer updates); the training process and GPU worker have exited. No checkpoint had yet been saved.

- Output target: `outputs/libero_paper_20261117_153321`
- Live logs: `outputs/run_logs/libero_paper_20261117_153321.stderr.log` (LeRobot metrics), `.stdout.log` (startup)
- Dataset: `lerobot/libero`, 1,691 training episodes / 272,941 frames / 40 tasks. Episodes 53 and 347 are excluded as the prior task-7 holdout. Original dataset metadata statistics still include these episodes; they are excluded from gradient updates.
- Initialization: pretrained SmolVLM2 vision-language backbone, new SmolVLA action expert; VLM frozen, action expert trained.
- 50-action predictions; 1 action executed before replanning at evaluation, 10 flow steps. 512x512 images; bfloat16 mixed precision.
- GPU batch 1, accumulation 64, global effective batch 64. LeRobot 0.7 counts *microbatches*, so 6,400,000 microbatches represent 100,000 optimizer updates.
- AdamW defaults: learning rate 1e-4, beta (0.9, 0.95). 100 optimizer-update warmup = 6,400 microbatches. Cosine decay to 2.5e-6 at 6,400,000 microbatches.
- Checkpoint every 64,000 microbatches (1,000 optimizer updates). About 100 checkpoints; previous checkpoint size was 1.232 GiB, so expected total is about 123 GiB. Free space at start: 383.9 GiB, leaving about 260 GiB if checkpoint size is similar.
- On the RTX 3050 Laptop 4 GiB, observed startup speed was about 3.2 microbatches/s, projecting about 23 days. This estimate will change with system load. Keep Windows awake and the GPU available.
- Initial log: step 64 loss 2.353, step 128 loss 2.304, step 192 loss 2.379. These are noisy training losses, not validation or rollout success.

Check progress in PowerShell:

```powershell
$j = Get-Content outputs\run_logs\active_libero_paper.json -Raw | ConvertFrom-Json
Select-String -Path $j.stderr -Pattern 'step:[0-9]+ smpl:' | Select-Object -Last 1
Get-Process -Id $j.pid
```

The run follows the [SmolVLA paper](https://arxiv.org/pdf/2506.01844) and [LeRobot LIBERO training guide](https://huggingface.co/docs/lerobot/libero) in model initialization, task coverage, global batch, update count, optimizer schedule, precision, and inference design. Its two excluded episodes and batch accumulation are adaptations for this experiment and 4 GiB GPU. The paper did not specify a required color or geometric augmentation for LIBERO, so none is applied here.
