#!/usr/bin/env python3
"""Benchmark live, case-insensitive 4x6 TRON matching on an NVIDIA GPU.

This program reuses an MIT-licensed CUDA TRON derivation kernel and adds a
fixed-width mask matcher. It never prints or writes private keys. Any GPU hit is
re-derived on the CPU and then its temporary Python integer is discarded.
"""

from __future__ import annotations

import argparse
import base58
import hashlib
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("CUPY_CACHE_DIR", str(PROJECT_ROOT / ".cache" / "cupy"))

try:
    import cupy as cp
    import coincurve
    from Crypto.Hash import keccak as keccak_lib
except ImportError as error:  # pragma: no cover - depends on target GPU environment
    raise SystemExit(
        "GPU dependencies are missing. Run ./scripts/setup_gpu_benchmark.sh, "
        "activate .venv, and try again. Missing: {}".format(error)
    )

from tron_bench import BASE58, PatternError, estimate, match_probability, parse_fixed_pattern


ROOT = PROJECT_ROOT
VENDOR = ROOT / "vendor" / "tron_vanity_cuda"
KERNEL_PATH = VENDOR / "kernels.cu"
sys.path.insert(0, str(VENDOR))

from cpu_worker import gen_startpoints_batch  # noqa: E402


SECP256K1_N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141


def cpu_priv_to_address(private_int: int) -> str:
    public = coincurve.PublicKey.from_valid_secret(private_int.to_bytes(32, "big")).format(
        compressed=False
    )
    digest = keccak_lib.new(digest_bits=256)
    digest.update(public[1:])
    payload = b"\x41" + digest.digest()[-20:]
    checksum = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
    return base58.b58encode(payload + checksum).decode("ascii")


THREADS_PER_BLOCK = 64
BLOCKS_PER_SM = 4
MAX_MATCHES = 64

PATTERN_DTYPE = np.dtype(
    [
        ("prefix_len", np.int32),
        ("suffix_len", np.int32),
        ("repeat_n", np.int32),
        ("reserved", np.int32),
        ("prefix_masks", np.uint64, 4),
        ("suffix_masks", np.uint64, 6),
    ],
    align=True,
)

MATCH_DTYPE = np.dtype(
    [
        ("thread_id", np.uint32),
        ("_pad", np.uint32),
        ("step", np.uint64),
        ("address", np.uint8, 34),
        ("_pad2", np.uint8, 6),
    ],
    align=True,
)


def compile_pattern(prefix: str, suffix: str, ignore_case: bool) -> np.ndarray:
    prefix_tokens = parse_fixed_pattern(prefix, ignore_case)
    suffix_tokens = parse_fixed_pattern(suffix, ignore_case)
    if len(prefix_tokens) != 4 or len(suffix_tokens) != 6:
        raise PatternError(
            "this milestone requires exactly four prefix tokens after T and six suffix tokens"
        )
    result = np.zeros(1, dtype=PATTERN_DTYPE)
    result["prefix_len"] = len(prefix_tokens)
    result["suffix_len"] = len(suffix_tokens)
    for token_index, accepted in enumerate(prefix_tokens):
        result["prefix_masks"][0, token_index] = _mask(accepted)
    for token_index, accepted in enumerate(suffix_tokens):
        result["suffix_masks"][0, token_index] = _mask(accepted)
    return result


def compile_reduced_pattern(prefix: str, suffix: str, ignore_case: bool) -> np.ndarray:
    """Compile a validation pattern up to 4x6, padding unused masks with zero."""

    prefix_tokens = parse_fixed_pattern(prefix, ignore_case)
    suffix_tokens = parse_fixed_pattern(suffix, ignore_case)
    if len(prefix_tokens) > 4 or len(suffix_tokens) > 6:
        raise PatternError("validation pattern exceeds the 4x6 matcher capacity")
    result = np.zeros(1, dtype=PATTERN_DTYPE)
    result["prefix_len"] = len(prefix_tokens)
    result["suffix_len"] = len(suffix_tokens)
    for token_index, accepted in enumerate(prefix_tokens):
        result["prefix_masks"][0, token_index] = _mask(accepted)
    for token_index, accepted in enumerate(suffix_tokens):
        result["suffix_masks"][0, token_index] = _mask(accepted)
    return result


def _mask(accepted: Sequence[str]) -> int:
    mask = 0
    for character in accepted:
        mask |= 1 << BASE58.index(character)
    return mask


def representative_literals(
    pattern: str, ignore_case: bool, count: int, first_after_t: bool = False
) -> str:
    tokens = parse_fixed_pattern(pattern, ignore_case)[:count]
    result: list[str] = []
    possible_second = frozenset("9ABCDEFGHJKLMNPQRSTUVWXYZ")
    for index, accepted in enumerate(tokens):
        choices = accepted
        if first_after_t and index == 0:
            choices = accepted.intersection(possible_second)
            if not choices:
                raise PatternError("first prefix token cannot occur immediately after T")
        result.append(next(char for char in BASE58 if char in choices))
    return "".join(result)


def load_kernel(architecture: str, points_per_thread: int):
    source = KERNEL_PATH.read_text(encoding="utf-8")
    module = cp.RawModule(
        code=source,
        options=(
            "-std=c++14",
            "--use_fast_math",
            f"-DPOINTS_PER_THREAD={points_per_thread}",
            f"-arch={architecture}",
        ),
        backend="nvrtc",
        name_expressions=("vanity_kernel",),
    )
    return module.get_function("vanity_kernel")


def _public_point_arrays(private_keys: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    xs = np.empty(len(private_keys) * 4, dtype=np.uint64)
    ys = np.empty(len(private_keys) * 4, dtype=np.uint64)
    for point_index, private_key in enumerate(private_keys):
        public = coincurve.PublicKey.from_valid_secret(private_key.to_bytes(32, "big")).format(
            compressed=False
        )
        x = int.from_bytes(public[1:33], "big")
        y = int.from_bytes(public[33:65], "big")
        for limb in range(4):
            xs[point_index * 4 + limb] = (x >> (64 * limb)) & 0xFFFFFFFFFFFFFFFF
            ys[point_index * 4 + limb] = (y >> (64 * limb)) & 0xFFFFFFFFFFFFFFFF
    return xs, ys


def choose_points_per_thread(architecture: str, pattern: np.ndarray) -> tuple[int, object]:
    rates: list[tuple[float, int, object]] = []
    for points_per_thread in (16, 8):
        kernel = load_kernel(architecture, points_per_thread)
        chains = 256 * points_per_thread
        xs, ys = _public_point_arrays(range(1, chains + 1))
        x_device = cp.asarray(xs)
        y_device = cp.asarray(ys)
        pattern_device = cp.asarray(pattern)
        matches = cp.zeros(MAX_MATCHES, dtype=MATCH_DTYPE)
        count = cp.zeros(1, dtype=cp.uint32)
        args = (
            x_device,
            y_device,
            np.int32(32),
            np.uint64(0),
            pattern_device,
            matches,
            count,
            np.uint32(MAX_MATCHES),
        )
        kernel((4,), (THREADS_PER_BLOCK,), args)
        cp.cuda.Stream.null.synchronize()
        count.fill(0)
        started = time.perf_counter()
        kernel((4,), (THREADS_PER_BLOCK,), args)
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - started
        rates.append((chains * 32 / elapsed, points_per_thread, kernel))
        print(f"M={points_per_thread}: {rates[-1][0] / 1e6:.3f} M addr/s tuning sample")
    _, selected, kernel = max(rates, key=lambda row: row[0])
    return selected, kernel


def create_start_points(chains: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    process_count = max(1, min(mp.cpu_count(), 8))
    chunk_count = process_count * 2
    base_size, remainder = divmod(chains, chunk_count)
    tasks = [base_size + (1 if index < remainder else 0) for index in range(chunk_count)]
    tasks = [task for task in tasks if task]
    print(f"Preparing {chains:,} independent GPU chains with {process_count} CPU workers...")
    with mp.get_context("spawn").Pool(processes=process_count) as pool:
        chunks = pool.map(gen_startpoints_batch, tasks)

    xs = np.empty(chains * 4, dtype=np.uint64)
    ys = np.empty(chains * 4, dtype=np.uint64)
    base_keys: list[int] = []
    output_index = 0
    for x_bytes, y_bytes, private_keys in chunks:
        for local_index, private_key in enumerate(private_keys):
            x = int.from_bytes(x_bytes[local_index * 32 : (local_index + 1) * 32], "big")
            y = int.from_bytes(y_bytes[local_index * 32 : (local_index + 1) * 32], "big")
            for limb in range(4):
                xs[output_index * 4 + limb] = (x >> (64 * limb)) & 0xFFFFFFFFFFFFFFFF
                ys[output_index * 4 + limb] = (y >> (64 * limb)) & 0xFFFFFFFFFFFFFFFF
            base_keys.append(private_key)
            output_index += 1
    return xs, ys, base_keys


class GpuRunner:
    def __init__(
        self,
        kernel,
        blocks: int,
        points_per_thread: int,
        xs: np.ndarray,
        ys: np.ndarray,
        base_keys: list[int],
    ) -> None:
        self.kernel = kernel
        self.blocks = blocks
        self.points_per_thread = points_per_thread
        self.chains = blocks * THREADS_PER_BLOCK * points_per_thread
        self.x_device = cp.asarray(xs)
        self.y_device = cp.asarray(ys)
        self.base_keys = base_keys
        self.pattern_device = cp.zeros(1, dtype=PATTERN_DTYPE)
        self.matches_device = cp.zeros(MAX_MATCHES, dtype=MATCH_DTYPE)
        self.count_device = cp.zeros(1, dtype=cp.uint32)
        self.step_offset = 0

    def set_pattern(self, pattern: np.ndarray) -> None:
        self.pattern_device.set(pattern)

    def launch(self, steps: int) -> tuple[int, np.ndarray, float]:
        self.count_device.fill(0)
        started = time.perf_counter()
        self.kernel(
            (self.blocks,),
            (THREADS_PER_BLOCK,),
            (
                self.x_device,
                self.y_device,
                np.int32(steps),
                np.uint64(self.step_offset),
                self.pattern_device,
                self.matches_device,
                self.count_device,
                np.uint32(MAX_MATCHES),
            ),
        )
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - started
        self.step_offset += steps
        count = int(self.count_device.get()[0])
        records = self.matches_device[: min(count, MAX_MATCHES)].get() if count else np.empty(0)
        return count, records, elapsed

    def verify_records(
        self, records: np.ndarray, prefix: str, suffix: str, ignore_case: bool
    ) -> int:
        prefix_tokens = parse_fixed_pattern(prefix, ignore_case)
        suffix_tokens = parse_fixed_pattern(suffix, ignore_case)
        verified = 0
        for record in records:
            packed = int(record["thread_id"])
            thread_index = packed >> 4
            point_index = packed & 0xF
            if thread_index >= self.blocks * THREADS_PER_BLOCK:
                continue
            if point_index >= self.points_per_thread:
                continue
            base_index = thread_index * self.points_per_thread + point_index
            private_int = (self.base_keys[base_index] + int(record["step"])) % SECP256K1_N
            address = bytes(record["address"]).decode("ascii")
            derived = cpu_priv_to_address(private_int)
            private_int = 0
            if derived != address:
                raise RuntimeError(f"GPU/CPU derivation mismatch for public address {address}")
            if not all(address[index + 1] in accepted for index, accepted in enumerate(prefix_tokens)):
                raise RuntimeError(f"GPU prefix matcher returned a false positive: {address}")
            suffix_start = len(address) - len(suffix_tokens)
            if not all(
                address[suffix_start + index] in accepted
                for index, accepted in enumerate(suffix_tokens)
            ):
                raise RuntimeError(f"GPU suffix matcher returned a false positive: {address}")
            verified += 1
        return verified


def calibrate_steps(runner: GpuRunner) -> int:
    sample_steps = 16
    _, _, elapsed = runner.launch(sample_steps)
    target_seconds = 0.5
    steps = round(sample_steps * target_seconds / max(elapsed, 1e-6))
    return max(8, min(4096, steps))


def timed_throughput(runner: GpuRunner, steps: int, seconds: float) -> tuple[float, int, float]:
    started = time.perf_counter()
    candidates = 0
    while time.perf_counter() - started < seconds:
        _, _, _ = runner.launch(steps)
        candidates += runner.chains * steps
    elapsed = time.perf_counter() - started
    return candidates / elapsed, candidates, elapsed


def validation_hits(
    runner: GpuRunner,
    pattern: np.ndarray,
    prefix: str,
    suffix: str,
    ignore_case: bool,
    steps: int,
    wanted_hits: int,
    timeout_seconds: float,
) -> dict[str, float | int]:
    runner.set_pattern(pattern)
    started = time.perf_counter()
    candidates = 0
    hits = 0
    verified = 0
    while hits < wanted_hits and time.perf_counter() - started < timeout_seconds:
        count, records, _ = runner.launch(steps)
        candidates += runner.chains * steps
        if count > MAX_MATCHES:
            raise RuntimeError("match buffer overflow; use a harder validation pattern")
        hits += count
        verified += runner.verify_records(records, prefix, suffix, ignore_case)
    elapsed = time.perf_counter() - started
    return {
        "hits": hits,
        "verified_hits": verified,
        "candidates": candidates,
        "elapsed_seconds": elapsed,
        "observed_seconds_per_hit": elapsed / hits if hits else math.inf,
        "observed_candidates_per_hit": candidates / hits if hits else math.inf,
    }


def first_requested_hit(
    runner: GpuRunner,
    pattern: np.ndarray,
    prefix: str,
    suffix: str,
    ignore_case: bool,
    steps: int,
    timeout_seconds: float,
) -> dict[str, float | int | bool | None]:
    if timeout_seconds <= 0:
        return {
            "attempted": False,
            "found": False,
            "verified": False,
            "candidates": 0,
            "elapsed_seconds": 0.0,
            "public_address": None,
        }
    runner.set_pattern(pattern)
    started = time.perf_counter()
    candidates = 0
    while time.perf_counter() - started < timeout_seconds:
        count, records, _ = runner.launch(steps)
        candidates += runner.chains * steps
        if count:
            if count > MAX_MATCHES:
                raise RuntimeError("requested-pattern match buffer overflow")
            verified = runner.verify_records(records, prefix, suffix, ignore_case)
            address = bytes(records[0]["address"]).decode("ascii") if verified else None
            return {
                "attempted": True,
                "found": verified > 0,
                "verified": verified > 0,
                "candidates": candidates,
                "elapsed_seconds": time.perf_counter() - started,
                "public_address": address,
            }
    return {
        "attempted": True,
        "found": False,
        "verified": False,
        "candidates": candidates,
        "elapsed_seconds": time.perf_counter() - started,
        "public_address": None,
    }


def gpu_snapshot() -> dict[str, str]:
    fields = "name,driver_version,power.limit,temperature.gpu,utilization.gpu,memory.total"
    try:
        output = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        values = [value.strip() for value in output.splitlines()[0].split(",")]
        return dict(zip(fields.split(","), values))
    except (OSError, subprocess.SubprocessError, IndexError):
        return {}


def write_result(path: Path, result: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(descriptor, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RTX TRON 4x6 live-generation benchmark")
    parser.add_argument("--prefix", required=True, help="four fixed-width tokens after T")
    parser.add_argument("--suffix", required=True, help="six fixed-width suffix tokens")
    case = parser.add_mutually_exclusive_group()
    case.add_argument("--ignore-case", dest="ignore_case", action="store_true", default=True)
    case.add_argument("--case-sensitive", dest="ignore_case", action="store_false")
    parser.add_argument("--warmup-seconds", type=float, default=10.0)
    parser.add_argument("--benchmark-seconds", type=float, default=60.0)
    parser.add_argument("--validation-prefix", default=None)
    parser.add_argument("--validation-suffix", default=None)
    parser.add_argument("--validation-hits", type=int, default=20)
    parser.add_argument("--validation-timeout", type=float, default=300.0)
    parser.add_argument(
        "--full-search-seconds",
        type=float,
        default=0.0,
        help="optionally search the actual 4x6 request for one verified hit",
    )
    parser.add_argument("--sla-seconds", type=float, default=30.0)
    parser.add_argument("--output", type=Path, default=Path("results/gpu-live-benchmark.json"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.warmup_seconds <= 0 or args.benchmark_seconds <= 0:
        raise SystemExit("warmup and benchmark durations must be positive")
    if args.validation_hits <= 0 or args.validation_timeout <= 0:
        raise SystemExit("validation hits and timeout must be positive")
    if args.full_search_seconds < 0:
        raise SystemExit("full-search seconds cannot be negative")

    try:
        requested_pattern = compile_pattern(args.prefix, args.suffix, args.ignore_case)
        favorable, _, _ = match_probability(args.prefix, args.suffix, args.ignore_case)
        if favorable == 0:
            raise PatternError("requested pattern cannot occur in a TRON address")
    except PatternError as error:
        raise SystemExit(f"invalid requested pattern: {error}") from error

    validation_prefix = args.validation_prefix
    validation_suffix = args.validation_suffix
    if validation_prefix is None:
        validation_prefix = representative_literals(
            args.prefix, args.ignore_case, 2, first_after_t=True
        )
    if validation_suffix is None:
        validation_suffix = representative_literals(args.suffix, args.ignore_case, 3)
    try:
        validation_pattern = compile_reduced_pattern(
            validation_prefix, validation_suffix, args.ignore_case
        )
    except PatternError as error:
        raise SystemExit(f"invalid validation pattern: {error}") from error

    try:
        device_count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as error:
        raise SystemExit(
            "CUDA runtime cannot access the NVIDIA driver from this shell. "
            "Run the command in the same host terminal where nvidia-smi works. "
            f"CUDA error: {error}"
        ) from error
    if device_count < 1:
        raise SystemExit("no CUDA GPU detected")
    properties = cp.cuda.runtime.getDeviceProperties(0)
    name_value = properties["name"]
    gpu_name = name_value.decode() if isinstance(name_value, bytes) else str(name_value)
    major, minor = int(properties["major"]), int(properties["minor"])
    architecture = f"sm_{major}{minor}"
    sm_count = int(properties["multiProcessorCount"])
    blocks = sm_count * BLOCKS_PER_SM
    print(f"GPU: {gpu_name}; architecture {architecture}; {sm_count} SMs")

    points_per_thread, kernel = choose_points_per_thread(architecture, requested_pattern)
    chains = blocks * THREADS_PER_BLOCK * points_per_thread
    xs, ys, base_keys = create_start_points(chains)
    runner = GpuRunner(kernel, blocks, points_per_thread, xs, ys, base_keys)
    runner.set_pattern(requested_pattern)
    steps = calibrate_steps(runner)
    print(f"Selected M={points_per_thread}; {chains:,} chains; {steps} steps/launch")

    print(f"Warming GPU for {args.warmup_seconds:g} seconds...")
    timed_throughput(runner, steps, args.warmup_seconds)
    print(f"Measuring sustained throughput for {args.benchmark_seconds:g} seconds...")
    rate, candidates, measured_seconds = timed_throughput(
        runner, steps, args.benchmark_seconds
    )
    projection = estimate(
        args.prefix,
        args.suffix,
        args.ignore_case,
        rate,
        1,
        args.sla_seconds,
    )

    print(
        f"Validating real matches with ^T{validation_prefix}...{validation_suffix}$ "
        f"(up to {args.validation_timeout:g}s)..."
    )
    validation = validation_hits(
        runner,
        validation_pattern,
        validation_prefix,
        validation_suffix,
        args.ignore_case,
        steps,
        args.validation_hits,
        args.validation_timeout,
    )
    validation_projection = estimate(
        validation_prefix,
        validation_suffix,
        args.ignore_case,
        rate,
        1,
        args.sla_seconds,
    )
    full_search = first_requested_hit(
        runner,
        requested_pattern,
        args.prefix,
        args.suffix,
        args.ignore_case,
        steps,
        args.full_search_seconds,
    )

    result: dict[str, object] = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": gpu_snapshot(),
        "cuda_architecture": architecture,
        "sm_count": sm_count,
        "points_per_thread": points_per_thread,
        "chains": chains,
        "steps_per_launch": steps,
        "requested_pattern": {
            "prefix_after_t": args.prefix,
            "suffix": args.suffix,
            "ignore_case": args.ignore_case,
        },
        "throughput": {
            "candidates": candidates,
            "elapsed_seconds": measured_seconds,
            "full_tron_addresses_per_second": rate,
            "million_addresses_per_second": rate / 1e6,
        },
        "projection": asdict(projection),
        "full_search": full_search,
        "validation_pattern": {
            "prefix_after_t": validation_prefix,
            "suffix": validation_suffix,
            "projection": asdict(validation_projection),
            **validation,
        },
        "private_keys_written_to_disk": False,
    }
    write_result(args.output, result)

    print()
    print(f"Measured GPU rate : {rate / 1e6:,.3f} M full TRON addr/s")
    print(f"4x6 mean          : {projection.mean_seconds:,.3f} seconds")
    print(f"4x6 median        : {projection.median_seconds:,.3f} seconds")
    print(f"4x6 p95           : {projection.p95_seconds:,.3f} seconds")
    print(
        f"Validation         : {validation['verified_hits']}/{validation['hits']} GPU hits "
        f"verified on CPU"
    )
    if full_search["attempted"]:
        status = "verified hit" if full_search["found"] else "no hit before timeout"
        print(
            f"Full 4x6 trial     : {status} after "
            f"{float(full_search['elapsed_seconds']):,.3f} seconds"
        )
    print(f"Result             : {args.output}")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
