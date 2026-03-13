#!/usr/bin/env bash
# Cron wrapper: sets up PATH and runs paper-tracker
# Usage: bash /home/shurui/wkspace/codex-test/paper-tracker/scripts/run.sh

set -euo pipefail

# Ensure uv, claude, codex are in PATH
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
TODAY=$(date -u +%Y-%m-%d)

exec uv run paper-tracker >> "$LOG_DIR/run-${TODAY}.log" 2>&1
