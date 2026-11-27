#!/usr/bin/env bash
# Reproduce every reported experiment. Each script is also a standalone command.
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/download_mvtec.py
python scripts/run_zeroshot.py "$@"
python scripts/run_retrieval.py "$@"
python scripts/run_localize.py "$@"
python scripts/run_prompt_ablation.py "$@"
python scripts/run_backbone_comparison.py "$@"
python scripts/run_fewshot.py "$@"
python scripts/run_localize.py "$@" --model RN50 --stem localization_rn50
