#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${project_dir}/build/cpp"

if ! compgen -G "${project_dir}/.venv/lib/python*/site-packages/nvidia/cuda_nvrtc/include/nvrtc.h" >/dev/null; then
  echo "CUDA runtime headers are missing. Run: bash scripts/setup_gpu_benchmark.sh" >&2
  exit 1
fi

cmake -S "${project_dir}/cpp" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" --parallel
ctest --test-dir "${build_dir}" --output-on-failure

echo "C++ benchmark ready: ${build_dir}/tron_gpu_benchmark"
