#!/bin/bash
# Keep the workspace under a disk budget (default 300 GB). Sourced by env.sh / env_cpu.sh.
# The measured folder is DISK_WS (default: the repo root). Put machine-specific values, e.g. DISK_WS=/path/to/workspace,
# into scripts/cluster/local.env (git-ignored); it is read first.
#  - at job start: refuse to run (return 1) if usage >= DISK_START_LIMIT_GB (default 270)
#  - inside a Slurm job: one background watchdog per job checks every 10 min and cancels the job at DISK_KILL_LIMIT_GB (290)
[ -f "$ROOT/scripts/cluster/local.env" ] && source "$ROOT/scripts/cluster/local.env"
DISK_WS="${DISK_WS:-$ROOT}"
DISK_START_LIMIT_GB="${DISK_START_LIMIT_GB:-270}"
DISK_KILL_LIMIT_GB="${DISK_KILL_LIMIT_GB:-290}"
disk_usage_gb() { du -s --block-size=1G "$DISK_WS" 2>/dev/null | cut -f1; }
_u=$(disk_usage_gb)
echo "disk guard: $DISK_WS uses ${_u:-?} GB (start limit $DISK_START_LIMIT_GB, kill limit $DISK_KILL_LIMIT_GB, hard cap 300)"
if [ -n "$_u" ] && [ "$_u" -ge "$DISK_START_LIMIT_GB" ]; then
  echo "DISK GUARD: workspace at ${_u} GB >= ${DISK_START_LIMIT_GB} GB, not starting (free space first)"
  return 1 2>/dev/null || exit 1
fi
if [ -n "$SLURM_JOB_ID" ] && mkdir -p "$ROOT/logs" && mkdir "/tmp/disk_watchdog_${SLURM_JOB_ID}_$USER" 2>/dev/null; then
  (
    while sleep 600; do
      u=$(disk_usage_gb)
      if [ -n "$u" ] && [ "$u" -ge "$DISK_KILL_LIMIT_GB" ]; then
        echo "DISK GUARD: workspace at ${u} GB >= ${DISK_KILL_LIMIT_GB} GB, cancelling job $SLURM_JOB_ID"
        scancel "$SLURM_JOB_ID"
      fi
    done
  ) </dev/null >>"$ROOT/logs/disk_guard_${SLURM_JOB_ID}.log" 2>&1 &   # never hold the caller's pipes open
  disown 2>/dev/null
fi
unset _u
