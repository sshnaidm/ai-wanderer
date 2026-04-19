#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="${1:-$SCRIPT_DIR/config.yaml}"

if [[ ! -f "$CONFIG" ]]; then
    echo "Config not found: $CONFIG" >&2
    echo "Usage: $0 [config.yaml]" >&2
    exit 1
fi

exec python -m ai_free_swap --config "$CONFIG"
