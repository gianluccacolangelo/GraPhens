#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

status=0

vulture src training tools/vulture_whitelist.py --exclude "*/__pycache__/*" || status=$?
vulture src training tools/vulture_whitelist.py --exclude "*/__pycache__/*" --min-confidence 90 || status=$?

exit "$status"
