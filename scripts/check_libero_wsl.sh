#!/usr/bin/env bash
ps -eo pid,etime,pcpu,stat,cmd | grep -E 'pip install|setup_libero_wsl' | grep -v grep || true
df -h /mnt/d
