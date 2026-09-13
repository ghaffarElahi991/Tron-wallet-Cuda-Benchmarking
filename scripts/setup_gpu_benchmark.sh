#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${project_dir}/.venv"
vendor_dir="${project_dir}/vendor/tron_vanity_cuda"
kernel_patch="${project_dir}/patches/tron_vanity_cuda_fixed_width_matcher.patch"

if [[ ! -f "${vendor_dir}/requirements.txt" ]]; then
  if ! command -v git >/dev/null 2>&1; then
    echo "git is required to download the CUDA source dependency." >&2
    exit 1
  fi

  echo "Initializing CUDA source dependency..."
  git -C "${project_dir}" submodule sync --recursive
  git -C "${project_dir}" submodule update --init --recursive
fi

if [[ ! -f "${vendor_dir}/requirements.txt" || ! -f "${vendor_dir}/kernels.cu" ]]; then
  echo "CUDA source dependency is incomplete: ${vendor_dir}" >&2
  exit 1
fi

if git -C "${vendor_dir}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if git -C "${vendor_dir}" apply --reverse --check --ignore-space-change \
    "${kernel_patch}" >/dev/null 2>&1; then
    echo "CUDA matcher patch is already applied."
  elif git -C "${vendor_dir}" apply --check --ignore-space-change \
    "${kernel_patch}" >/dev/null 2>&1; then
    echo "Applying CUDA fixed-width matcher patch..."
    git -C "${vendor_dir}" apply --ignore-space-change "${kernel_patch}"
  else
    echo "Cannot apply CUDA matcher patch; vendor source has unexpected changes." >&2
    exit 1
  fi
elif grep -q "u64 prefix_masks\[5\]" "${vendor_dir}/kernels.cu"; then
  echo "CUDA matcher patch is already present in the packaged source."
else
  echo "Packaged CUDA kernel does not contain the fixed-width matcher." >&2
  exit 1
fi

python3 -m venv "${venv_dir}"
"${venv_dir}/bin/python" -m pip install --upgrade pip
"${venv_dir}/bin/python" -m pip install -r "${project_dir}/vendor/tron_vanity_cuda/requirements.txt"

echo "Setup complete. Activate with: source ${venv_dir}/bin/activate"
