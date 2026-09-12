#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
benchmark="${project_dir}/build/cpp/tron_gpu_benchmark"
output="${project_dir}/results/rtx4050-direct-loose.json"

if [[ ! -x "${benchmark}" ]]; then
  echo "Benchmark binary not found. Build it first with:" >&2
  echo "  bash scripts/build_cpp_benchmark.sh" >&2
  exit 1
fi

mkdir -p "${project_dir}/results"

"${benchmark}" \
  --prefix 'New?' \
  --suffix 'adr?S?' \
  --case-mode ignore \
  --warmup-seconds 10 \
  --benchmark-seconds 60 \
  --validation-hits 20 \
  --validation-timeout 300 \
  --output "${output}"
