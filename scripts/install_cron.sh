#!/usr/bin/env bash
# Install a daily cron job for paper-tracker at midnight UTC
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_SCRIPT="$SCRIPT_DIR/run.sh"
CRON_JOB="0 0 * * * bash $RUN_SCRIPT"

# Check if already installed
if crontab -l 2>/dev/null | grep -qF "$RUN_SCRIPT"; then
    echo "Cron job already installed:"
    crontab -l | grep -F "$RUN_SCRIPT"
    exit 0
fi

# Append to existing crontab
(crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -
echo "Cron job installed: $CRON_JOB"
echo "Verify with: crontab -l"
