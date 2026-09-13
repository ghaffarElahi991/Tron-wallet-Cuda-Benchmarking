#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
benchmark="${project_dir}/build/cpp/tron_gpu_benchmark"

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <2x6|3x5|4x4>" >&2
  exit 2
fi

pattern_size="$1"
case "${pattern_size}" in
  2x6)
    prefix='Ne'
    suffix='adreSS'
    ;;
  3x5)
    prefix='New'
    suffix='adreS'
    ;;
  4x4)
    prefix='NewW'
    suffix='adre'
    ;;
  *)
    echo "Unsupported pattern size: ${pattern_size}" >&2
    echo "Choose one of: 2x6, 3x5, or 4x4." >&2
    exit 2
    ;;
esac

output="${project_dir}/results/multi-gpu-${pattern_size}-loose.json"

if [[ ! -x "${benchmark}" ]]; then
  echo "Benchmark binary not found. Build it first with:" >&2
  echo "  bash scripts/build_cpp_benchmark.sh" >&2
  exit 1
fi

mkdir -p "${project_dir}/results"

echo "Running ${pattern_size} case-insensitive benchmark: ^T${prefix}...${suffix}$"

"${benchmark}" \
  --prefix "${prefix}" \
  --suffix "${suffix}" \
  --case-mode ignore \
  --devices all \
  --debug-math \
  --warmup-seconds 10 \
  --benchmark-seconds 60 \
  --validation-hits 20 \
  --validation-timeout 300 \
  --output "${output}"
