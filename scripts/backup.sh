#!/usr/bin/env sh
set -eu
mkdir -p backups
sqlite3 "${DIBLE_DB_PATH:-./data/dible.db}" ".backup 'backups/dible-$(date +%Y%m%dT%H%M%S).db'"
