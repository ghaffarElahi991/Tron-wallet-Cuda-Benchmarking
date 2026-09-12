# TRON live 4x6 feasibility report

Date: 2026-09-10

## Decision

A case-insensitive 4x6 prefix/suffix request is not suitable for synchronous,
live generation on one RTX 4090. Case-insensitivity is a major improvement, but
the remaining search space is still hundreds of trillions of complete address
candidates for an ordinary all-letter pattern.

No RTX 4090 was available in this workspace, so the figures below are model
projections, not locally measured results. `750 M addr/s` is a third-party
estimate for a full TRON pipeline on an RTX 4090. `2,224 M addr/s` is included
only as an optimistic ceiling illustration: its publisher describes that number
as an EVM unreachable-target benchmark, so it must not be treated as measured
TRON Base58Check throughput.

## Definition and representative request

The unambiguous 4x6 definition used here is:

```text
^T<four requested characters>...<six requested characters>$
```

The invariant leading `T` is not one of the four requested prefix characters.
The representative pattern `^TAbcd...WxyzAB$` uses ten ordinary letters; each
has two valid Base58 case variants except at the first position after `T`, where
TRON's fixed network byte permits only the uppercase form.

| Rate assumption | Mean | Median | p95 | Chance within 30 s |
|---|---:|---:|---:|---:|
| 750 M full TRON addr/s | 451,423 s (5.22 d) | 312,903 s (3.62 d) | 1,352,344 s (15.7 d) | 0.00665% |
| 2,224 M addr/s optimistic ceiling | 152,234 s (1.76 d) | 105,520 s (1.22 d) | 456,051 s (5.28 d) | 0.0197% |

If "four-character prefix" includes the fixed `T`, there are three variable
prefix characters plus six suffix characters. At 750 M addr/s the projected
mean is still 15,566 s (4.32 h), with a 12.95-hour p95.

## Mathematics

For independent candidates with per-candidate hit probability `p` and sustained
rate `r`:

```text
expected attempts = 1 / p
mean seconds      = 1 / (r * p)
P(hit by t)       = 1 - (1 - p) ^ floor(r * t)
quantile(q)       = ceil(log(1-q) / log(1-p)) / r
```

For an ordinary alphabetic digit away from the beginning, case-insensitive
matching has probability `2/58 = 1/29`. Digits have probability `1/58`.
Because Base58 omits uppercase `I`/`O` and lowercase `l`, the folded classes for
`i`, `o`, and `L` also have only one valid symbol and therefore cost `1/58`.
Every digit or singleton-case letter in place of an ordinary letter doubles the
mean search time.

The first character after `T` is special. It can only be `9` or an uppercase
Base58 letter, and its distribution has edge bias. The calculator counts the
accepted Base58 integer intervals directly, so it handles this constraint rather
than applying a naive `1/29` factor. For the representative request it returns:

```text
p                 = 2.95362017222e-15
expected attempts = 338,567,568,506,697 (approximately)
```

Case-insensitivity improves this representative TRON 4x6 pattern by 512x rather
than 1024x: nine positions gain two case variants, while the first post-`T`
position cannot be lowercase.

## Practical synchronous envelope at 750 M addr/s

These examples use ordinary letters and count the prefix after the fixed `T`.

| Split | Requested positions | Mean | p95 | Chance within 30 s |
|---|---:|---:|---:|---:|
| 2x3 | 5 | 0.022 s | 0.066 s | ~100% |
| 3x3 | 6 | 0.638 s | 1.91 s | ~100% |
| 3x4 | 7 | 18.5 s | 55.4 s | 80.2% |
| 4x4 | 8 | 536.8 s | 1,608 s | 5.44% |
| 4x6 | 10 | 451,423 s | 1,352,344 s | 0.00665% |

This puts a reliable live product at roughly six requested ordinary-letter
positions per RTX 4090. Seven positions may work as an explicitly variable-wait
tier. Ten positions needs an asynchronous job or a fundamentally looser pattern.
At 750 M addr/s, meeting a 30-second mean for true 4x6 would require roughly
15,048 equivalent GPUs; meeting a 30-second p95 would require about 45,079.

## Measurement protocol

1. Benchmark complete TRON candidates, including secp256k1, Keccak-256, the
   Base58Check work required for matching, and the case-insensitive matcher.
2. Verify GPU outputs against an independent CPU derivation before measuring.
3. Record a 60-second warmup and a minimum 180-second sustained throughput run.
4. Validate the probability model with at least 100 reachable 2x3 and 3x3 hits.
5. Keep the GPU worker resident to measure production-like warm request latency;
   process startup and CUDA compilation should be reported separately as cold
   latency.
6. Expect high variance: geometric waiting time has a standard deviation close
   to its mean. Thirty hits give about 18% relative standard error, 100 give 10%,
   and 400 give 5%, before systematic effects.
7. Never fund benchmark keys. Do not retain child output or plaintext keys in
   logs, telemetry, shell history, screenshots, or cloud-synced storage.

Use `tron_bench.py estimate` for pattern-specific projections and
`tron_bench.py summarize` for measured JSONL trial data. The exact requested
pattern matters; changing letters to digits or to `i`/`o`/`L` changes the result.

## Local GPU runner

`gpu_live_benchmark.py` now measures this pipeline directly on an NVIDIA GPU. A
normal run benchmarks the exact 4x6 matching workload, validates real reduced
2x3 hits against an independent CPU derivation, and writes a key-free JSON
report. `--full-search-seconds N` optionally adds a bounded attempt to extract a
real requested 4x6 address. The expected average remains the mathematical
projection from the sustained measured throughput; enough real 4x6 samples to
estimate the average directly would take an impractical amount of GPU time.
