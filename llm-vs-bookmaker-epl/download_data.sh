#!/usr/bin/env bash
# Download the Football-Data.co.uk CSVs used by this study into data/ (not redistributed here; see DATA.md).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data
for s in 2122 2223 2324 2425 2526; do
  curl -fsSL -A "Mozilla/5.0" "https://www.football-data.co.uk/mmz4281/${s}/E0.csv" -o "data/E0_${s}.csv"
  echo "ok data/E0_${s}.csv"
done
