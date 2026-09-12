#!/usr/bin/env python3
"""TRON vanity-address probability and latency benchmark calculator.

The calculator intentionally does not generate or persist private keys.  It models
the Base58Check search space and turns a measured full-TRON-address rate into
latency/SLA numbers for a fixed-width prefix/suffix request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal, getcontext
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence


BASE58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
BASE58_SET = frozenset(BASE58)
BASE58_INDEX = {char: index for index, char in enumerate(BASE58)}
ADDRESS_LENGTH = 34
TRON_PREFIX_BYTE = 0x41
CHECKSUM_BYTES = 4
PAYLOAD_RANDOM_BYTES = 20

# The 25-byte Base58Check integer occupies this interval. The last four bytes
# are a checksum in real addresses; treating them as uniformly distributed is
# the standard cryptographic model used for suffix probabilities.
RAW_LOW = TRON_PREFIX_BYTE * 256**24
RAW_HIGH = (TRON_PREFIX_BYTE + 1) * 256**24 - 1
RAW_SPACE = RAW_HIGH - RAW_LOW + 1

getcontext().prec = 80


class PatternError(ValueError):
    """Raised when a fixed-width pattern is invalid."""


def _case_variants(char: str) -> frozenset[str]:
    variants = {char}
    if char.isascii() and char.isalpha():
        variants.update((char.lower(), char.upper()))
    return frozenset(variant for variant in variants if variant in BASE58_SET)


def parse_fixed_pattern(pattern: str, case_insensitive: bool) -> list[frozenset[str]]:
    """Parse literals, '?' wildcards, and '[abc]' character classes.

    Each token consumes exactly one address character. This restricted grammar
    maps directly to an efficient GPU matcher and permits an exact digit-DP
    probability calculation in the Base58-space model.
    """

    tokens: list[frozenset[str]] = []
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char == "?":
            accepted = BASE58_SET
            index += 1
        elif char == "[":
            close = pattern.find("]", index + 1)
            if close == -1:
                raise PatternError("unclosed '[' character class")
            content = pattern[index + 1 : close]
            if not content:
                raise PatternError("empty '[]' character class")
            if any(item in "[]?" for item in content):
                raise PatternError("character classes contain Base58 literals only")
            accepted = frozenset(content)
            index = close + 1
        elif char in "]*+{}()|.^$\\":
            raise PatternError(
                f"unsupported regex operator {char!r}; use literals, '?', or '[abc]'"
            )
        else:
            accepted = frozenset((char,))
            index += 1

        if case_insensitive:
            expanded: set[str] = set()
            for item in accepted:
                expanded.update(_case_variants(item))
            accepted = frozenset(expanded)
        else:
            invalid = sorted(set(accepted) - BASE58_SET)
            if invalid:
                raise PatternError(
                    f"characters not present in Base58: {''.join(invalid)!r}"
                )

        if not accepted:
            raise PatternError("token cannot match any Base58 character")
        tokens.append(accepted)

    return tokens


def _base58_digits(number: int, width: int = ADDRESS_LENGTH) -> tuple[int, ...]:
    digits = [0] * width
    for index in range(width - 1, -1, -1):
        number, remainder = divmod(number, 58)
        digits[index] = remainder
    if number:
        raise ValueError("number does not fit requested Base58 width")
    return tuple(digits)


def _count_at_most(bound: int, allowed: Sequence[frozenset[int]]) -> int:
    if bound < 0:
        return 0
    bound_digits = _base58_digits(bound, len(allowed))

    @lru_cache(maxsize=None)
    def visit(position: int, tight: bool) -> int:
        if position == len(allowed):
            return 1
        maximum = bound_digits[position] if tight else 57
        total = 0
        for digit in allowed[position]:
            if digit <= maximum:
                total += visit(position + 1, tight and digit == maximum)
        return total

    return visit(0, True)


def match_probability(
    prefix: str, suffix: str, case_insensitive: bool
) -> tuple[int, int, list[frozenset[str]]]:
    """Return favorable raw integers, denominator, and position match sets.

    ``prefix`` starts immediately after TRON's invariant leading ``T``.
    """

    prefix_tokens = parse_fixed_pattern(prefix, case_insensitive)
    suffix_tokens = parse_fixed_pattern(suffix, case_insensitive)
    if 1 + len(prefix_tokens) + len(suffix_tokens) > ADDRESS_LENGTH:
        raise PatternError("fixed T + prefix + suffix exceeds the 34-character address")

    positions: list[frozenset[str]] = [BASE58_SET for _ in range(ADDRESS_LENGTH)]
    positions[0] = frozenset(("T",))

    for offset, accepted in enumerate(prefix_tokens, start=1):
        positions[offset] = positions[offset].intersection(accepted)
    suffix_start = ADDRESS_LENGTH - len(suffix_tokens)
    for offset, accepted in enumerate(suffix_tokens, start=suffix_start):
        positions[offset] = positions[offset].intersection(accepted)

    if any(not accepted for accepted in positions):
        return 0, RAW_SPACE, positions

    allowed_indices = [
        frozenset(BASE58_INDEX[char] for char in accepted) for accepted in positions
    ]
    favorable = _count_at_most(RAW_HIGH, allowed_indices) - _count_at_most(
        RAW_LOW - 1, allowed_indices
    )
    return favorable, RAW_SPACE, positions


def quantile_attempts(probability: float, quantile: float) -> int:
    if not 0.0 < probability <= 1.0:
        raise ValueError("probability must be in (0, 1]")
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be in (0, 1)")
    if probability == 1.0:
        return 1
    return math.ceil(math.log1p(-quantile) / math.log1p(-probability))


def format_duration(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.3g} us"
    if seconds < 1:
        return f"{seconds * 1e3:.3g} ms"
    if seconds < 60:
        return f"{seconds:.3g} s"
    if seconds < 3600:
        return f"{seconds / 60:.3g} min"
    if seconds < 86400:
        return f"{seconds / 3600:.3g} h"
    return f"{seconds / 86400:.3g} days"


@dataclass(frozen=True)
class Estimate:
    prefix_after_t: str
    suffix: str
    case_insensitive: bool
    effective_rate_per_second: float
    favorable_space: int
    total_space: int
    probability_per_candidate: float
    expected_attempts: float
    mean_seconds: float
    median_seconds: float
    p95_seconds: float
    p99_seconds: float
    sla_seconds: float
    probability_within_sla: float


def estimate(
    prefix: str,
    suffix: str,
    case_insensitive: bool,
    rate_per_second: float,
    gpus: int,
    sla_seconds: float,
) -> Estimate:
    if rate_per_second <= 0:
        raise ValueError("rate must be greater than zero")
    if gpus <= 0:
        raise ValueError("GPU count must be greater than zero")
    if sla_seconds < 0:
        raise ValueError("SLA seconds cannot be negative")

    favorable, total, _ = match_probability(prefix, suffix, case_insensitive)
    if favorable == 0:
        raise PatternError("pattern cannot occur in a TRON Base58Check address")

    probability_decimal = Decimal(favorable) / Decimal(total)
    probability = float(probability_decimal)
    expected_attempts_decimal = Decimal(total) / Decimal(favorable)
    expected_attempts = float(expected_attempts_decimal)
    effective_rate = rate_per_second * gpus
    mean_seconds = expected_attempts / effective_rate
    median_seconds = quantile_attempts(probability, 0.5) / effective_rate
    p95_seconds = quantile_attempts(probability, 0.95) / effective_rate
    p99_seconds = quantile_attempts(probability, 0.99) / effective_rate
    attempts_in_sla = math.floor(effective_rate * sla_seconds)
    probability_within_sla = -math.expm1(attempts_in_sla * math.log1p(-probability))

    return Estimate(
        prefix_after_t=prefix,
        suffix=suffix,
        case_insensitive=case_insensitive,
        effective_rate_per_second=effective_rate,
        favorable_space=favorable,
        total_space=total,
        probability_per_candidate=probability,
        expected_attempts=expected_attempts,
        mean_seconds=mean_seconds,
        median_seconds=median_seconds,
        p95_seconds=p95_seconds,
        p99_seconds=p99_seconds,
        sla_seconds=sla_seconds,
        probability_within_sla=probability_within_sla,
    )


def _print_estimate(result: Estimate) -> None:
    mode = "case-insensitive" if result.case_insensitive else "case-sensitive"
    print(f"Pattern       : ^T{result.prefix_after_t}...{result.suffix}$ ({mode})")
    print(f"Measured rate : {result.effective_rate_per_second / 1e6:,.3f} M addr/s")
    print(f"Match chance  : {result.probability_per_candidate:.12g} per candidate")
    print(f"Mean attempts : {result.expected_attempts:,.6g}")
    print(f"Mean latency  : {result.mean_seconds:,.3f} s ({format_duration(result.mean_seconds)})")
    print(f"Median        : {result.median_seconds:,.3f} s ({format_duration(result.median_seconds)})")
    print(f"p95           : {result.p95_seconds:,.3f} s ({format_duration(result.p95_seconds)})")
    print(f"p99           : {result.p99_seconds:,.3f} s ({format_duration(result.p99_seconds)})")
    print(
        f"P(hit <= {result.sla_seconds:g}s): "
        f"{result.probability_within_sla * 100:.9g}%"
    )
    print("Model         : exact Base58 interval; uniform-checksum assumption for suffix digits")


def _load_trials(paths: Iterable[Path]) -> list[dict[str, object]]:
    trials: list[dict[str, object]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: each JSONL row must be an object")
                trials.append(value)
    return trials


def summarize(paths: Iterable[Path]) -> dict[str, object]:
    records = _load_trials(paths)
    successful = [record for record in records if record.get("found", True)]
    elapsed = [float(record["elapsed_seconds"]) for record in successful]
    if not elapsed:
        raise ValueError("no successful rows containing elapsed_seconds")
    if any(value <= 0 for value in elapsed):
        raise ValueError("elapsed_seconds values must be positive")

    candidates = [
        float(record["candidates"])
        for record in successful
        if record.get("candidates") is not None
    ]
    result: dict[str, object] = {
        "total_rows": len(records),
        "successful_trials": len(elapsed),
        "mean_seconds": statistics.fmean(elapsed),
        "median_seconds": statistics.median(elapsed),
        "minimum_seconds": min(elapsed),
        "maximum_seconds": max(elapsed),
        "sample_stdev_seconds": statistics.stdev(elapsed) if len(elapsed) > 1 else None,
        "standard_error_seconds": (
            statistics.stdev(elapsed) / math.sqrt(len(elapsed)) if len(elapsed) > 1 else None
        ),
    }
    if len(candidates) == len(elapsed):
        result["aggregate_rate_per_second"] = sum(candidates) / sum(elapsed)
    return result


def _stop_process_group(process: subprocess.Popen[bytes]) -> None:
    """Terminate only the process group created for a measured command."""

    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=3)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def _run_once(command: Sequence[str], timeout_seconds: float) -> tuple[float, int | None, bool]:
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    timed_out = False
    try:
        return_code = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        _stop_process_group(process)
        return_code = None
    elapsed = time.perf_counter() - started
    return elapsed, return_code, timed_out


def measure_trials(
    command: Sequence[str],
    output: Path,
    trials: int,
    warmups: int,
    timeout_seconds: float,
    success_exit_code: int,
    pattern_label: str,
    gpu_label: str,
) -> list[dict[str, object]]:
    """Measure cold process-to-hit latency without retaining child output."""

    if not command:
        raise ValueError("a generator command is required after '--'")
    if trials <= 0:
        raise ValueError("trials must be greater than zero")
    if warmups < 0:
        raise ValueError("warmups cannot be negative")
    if timeout_seconds <= 0:
        raise ValueError("timeout must be greater than zero")

    for _ in range(warmups):
        _run_once(command, timeout_seconds)

    output.parent.mkdir(parents=True, exist_ok=True)
    command_digest = hashlib.sha256(b"\0".join(os.fsencode(arg) for arg in command)).hexdigest()
    rows: list[dict[str, object]] = []
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    descriptor = os.open(output, flags, 0o600)
    os.fchmod(descriptor, 0o600)
    with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
        for trial_number in range(1, trials + 1):
            elapsed, return_code, timed_out = _run_once(command, timeout_seconds)
            row: dict[str, object] = {
                "schema_version": 1,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "trial": trial_number,
                "mode": "cold_process_to_hit",
                "pattern": pattern_label,
                "gpu": gpu_label,
                "elapsed_seconds": elapsed,
                "found": not timed_out and return_code == success_exit_code,
                "timed_out": timed_out,
                "return_code": return_code,
                "command_sha256": command_digest,
            }
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            handle.flush()
            rows.append(row)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TRON case-insensitive prefix/suffix latency calculator"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    estimate_parser = subparsers.add_parser("estimate", help="project latency from measured rate")
    estimate_parser.add_argument(
        "--prefix",
        default="",
        help="fixed-width pattern immediately after the invariant leading T",
    )
    estimate_parser.add_argument("--suffix", default="", help="anchored suffix pattern")
    case_group = estimate_parser.add_mutually_exclusive_group()
    case_group.add_argument(
        "--case-insensitive", dest="case_insensitive", action="store_true", default=True
    )
    case_group.add_argument(
        "--case-sensitive", dest="case_insensitive", action="store_false"
    )
    rate_group = estimate_parser.add_mutually_exclusive_group(required=True)
    rate_group.add_argument("--rate", type=float, help="measured full TRON addresses/second")
    rate_group.add_argument("--rate-mh", type=float, help="measured million TRON addresses/second")
    estimate_parser.add_argument("--gpus", type=int, default=1)
    estimate_parser.add_argument("--sla-seconds", type=float, default=30.0)
    estimate_parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    summarize_parser = subparsers.add_parser("summarize", help="summarize JSONL latency trials")
    summarize_parser.add_argument("paths", type=Path, nargs="+")
    summarize_parser.add_argument("--json", action="store_true")

    measure_parser = subparsers.add_parser(
        "measure", help="measure a generator command without retaining its output"
    )
    measure_parser.add_argument("--output", type=Path, required=True)
    measure_parser.add_argument("--trials", type=int, default=30)
    measure_parser.add_argument("--warmups", type=int, default=1)
    measure_parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    measure_parser.add_argument("--success-exit-code", type=int, default=0)
    measure_parser.add_argument("--pattern-label", default="")
    measure_parser.add_argument("--gpu-label", default="")
    measure_parser.add_argument(
        "generator_command",
        nargs=argparse.REMAINDER,
        help="generator command and arguments, preceded by '--'",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "estimate":
            rate = args.rate if args.rate is not None else args.rate_mh * 1e6
            result = estimate(
                prefix=args.prefix,
                suffix=args.suffix,
                case_insensitive=args.case_insensitive,
                rate_per_second=rate,
                gpus=args.gpus,
                sla_seconds=args.sla_seconds,
            )
            if args.json:
                print(json.dumps(asdict(result), indent=2, sort_keys=True))
            else:
                _print_estimate(result)
        elif args.command == "summarize":
            result = summarize(args.paths)
            if args.json:
                print(json.dumps(result, indent=2, sort_keys=True))
            else:
                for key, value in result.items():
                    print(f"{key}: {value}")
        elif args.command == "measure":
            command = list(args.generator_command)
            if command and command[0] == "--":
                command.pop(0)
            rows = measure_trials(
                command=command,
                output=args.output,
                trials=args.trials,
                warmups=args.warmups,
                timeout_seconds=args.timeout_seconds,
                success_exit_code=args.success_exit_code,
                pattern_label=args.pattern_label,
                gpu_label=args.gpu_label,
            )
            successes = sum(bool(row["found"]) for row in rows)
            print(f"Recorded {len(rows)} trials ({successes} successful) in {args.output}")
        else:  # pragma: no cover - argparse enforces this
            parser.error("unknown command")
    except (PatternError, ValueError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    sys.exit(main())
