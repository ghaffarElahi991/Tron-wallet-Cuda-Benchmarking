# TRON live vanity benchmark — milestone 0

This workspace currently contains the feasibility and benchmark layer for live,
on-demand TRON vanity generation. It uses transient private scalars to derive
real addresses, but deliberately does **not** print or store private keys.

## C++ delivery build

The production-shaped benchmark is available as a standalone C++20 host with a
runtime-compiled CUDA kernel. It uses the CUDA Driver API, NVRTC, OpenSSL
secp256k1, the requested 25/58-position probability approximation, and
independent CPU verification.
The full CUDA toolkit and `nvcc` are not required; the existing setup script
installs the required runtime headers and NVRTC library into `.venv`.

Build and test it:

```bash
cd ~/Desktop/work/upwork/TaiBrown
bash scripts/setup_gpu_benchmark.sh  # only if .venv is not already prepared
bash scripts/build_cpp_benchmark.sh
```

Run the client-specified case-insensitive `TNewW...adreSS` request:

```bash
build/cpp/tron_gpu_benchmark \
  --prefix 'NewW' \
  --suffix 'adreSS' \
  --ignore-case \
  --warmup-seconds 10 \
  --benchmark-seconds 60 \
  --validation-hits 20 \
  --validation-timeout 300 \
  --output results/rtx4050-TNewW-adreSS-cpp.json
```

The JSON output is created with mode `0600`, contains public benchmark data
only, and records `private_keys_written_to_disk: false`. Add
`--full-search-seconds 300` when you also want a bounded real extraction trial.
The C++ projection treats the character after `T` as uniform over its 25
possible symbols and every later position as uniform over the 58-symbol Base58
alphabet. It counts the actual valid case variants in each mask, so letters such
as `L/l` do not incorrectly receive a two-case factor. For this target the model
is exactly `p = 1 / (25 * 29^9)` and
`E = 362,678,649,396,725` addresses. The measured GPU rate remains dynamic.

### Direct loose patterns

Case handling can be selected explicitly with `--case-mode ignore` or
`--case-mode exact` (`--ignore-case` and `--case-sensitive` remain supported).
Put `?` wildcards or `[abc]` character classes directly in the prefix and
suffix arguments:

```bash
build/cpp/tron_gpu_benchmark \
  --prefix 'AB??' \
  --suffix 'XYZ???' \
  --case-mode ignore \
  --output results/rtx4050-AB-XYZ-loose.json
```

This produces the fixed-width pattern `^TAB??...XYZ???$`. Each `?` accepts one
Base58 character. For a restricted position, use a class directly, such as
`--prefix 'AB[17][2K]'`.

### Multi-GPU execution

The C++ benchmark accepts 4x4, 3x5, and other fixed-width requests up to four
prefix tokens and six suffix tokens. It uses all visible CUDA GPUs concurrently
by default. Select specific CUDA device ordinals with `--devices 0,1`, or state
the default explicitly with `--devices all`:

```bash
build/cpp/tron_gpu_benchmark \
  --prefix 'AB??' \
  --suffix 'XYZ???' \
  --case-mode ignore \
  --devices 0,1 \
  --warmup-seconds 10 \
  --benchmark-seconds 60 \
  --output results/two-gpu-AB-XYZ.json
```

Each GPU owns an independent CUDA context and independently seeded candidate
chains. GPU workers run in parallel host threads. The report contains each
device's measured rate and uses the sum of those rates for the projected search
times. Use `--devices 0` when a single-GPU baseline is required.

The convenience script requires the pattern size as its argument:

```bash
# Prefix AB?? (4), suffix XY?? (4)
bash scripts/run_case_insensitive_loose_benchmark.sh 4x4

# Prefix AB? (3), suffix XYZ?? (5)
bash scripts/run_case_insensitive_loose_benchmark.sh 3x5
```

Results are written to `results/multi-gpu-4x4-loose.json` or
`results/multi-gpu-3x5-loose.json`. The script enables `--debug-math`, which
prints every probability factor, expected attempts, per-GPU rate equation,
combined rate, and the mean/median/p95 latency calculations.

## Run the real GPU benchmark

The RTX benchmark worker is now included. It uses the full TRON pipeline on the
GPU: secp256k1 point generation, Keccak-256, double SHA-256, Base58Check suffix
extraction, and the fixed-width mask matcher. It prints and stores no private
keys; GPU hits are independently re-derived on the CPU before being counted.

Install the isolated Python environment once (the CUDA packages are large):

```bash
cd ~/Desktop/work/upwork/TaiBrown
bash scripts/setup_gpu_benchmark.sh
source .venv/bin/activate
```

Then run the representative case-insensitive 4x6 benchmark:

```bash
python3 gpu_live_benchmark.py \
  --prefix Abcd \
  --suffix WxyzAB \
  --ignore-case \
  --warmup-seconds 10 \
  --benchmark-seconds 60 \
  --validation-hits 20 \
  --validation-timeout 300 \
  --output results/rtx4050-4x6.json
```

This performs three separate checks:

1. Measures sustained throughput using the actual requested 4x6 matcher.
2. Finds real reduced 2x3 matches and verifies them on the CPU, demonstrating
   that the GPU matcher returns valid TRON addresses.
3. Uses the measured rate and the exact pattern probability to print the 4x6
   mean, median, and p95 in seconds.

The default run does not wait for an actual 4x6 hit because that may take weeks
on a laptop GPU. To add a bounded real extraction attempt, for example one hour:

```bash
python3 gpu_live_benchmark.py \
  --prefix Abcd --suffix WxyzAB --ignore-case \
  --full-search-seconds 3600 \
  --output results/rtx4050-4x6-one-hour.json
```

The CUDA implementation is based on the MIT-licensed
`Daniel-Wu-1/tron_vanity_address_generation` source at commit
`59dfbc1d8d971898c360af8912a9f22c6fd1de7e`; the local kernel modification adds
four prefix and six suffix bitmasks for case-insensitive literals, `?` wildcards,
and `[abc]` character classes.

## Pattern contract

- A TRON Base58Check address has 34 characters and an invariant leading `T`.
- `--prefix` starts **after** that `T`. Therefore a 4x6 request means four
  variable prefix characters plus six suffix characters: `^T<prefix>...<suffix>$`.
- Matching is case-insensitive by default.
- The GPU-friendly loose-pattern grammar is fixed-width: literals, `?` for any
  Base58 character, and `[abc]` for a character class.
- Full variable-width regex operators (`*`, `+`, alternation, captures) are
  intentionally rejected. They make GPU kernels less predictable and make the
  expected-work calculation ambiguous.

Examples:

```bash
# True 4x6, all requested characters are letters
python3 tron_bench.py estimate \
  --prefix Abcd \
  --suffix WxyzAB \
  --case-insensitive \
  --rate-mh 750 \
  --sla-seconds 30

# Four visible prefix characters including T means only three go in --prefix
python3 tron_bench.py estimate \
  --prefix Abc \
  --suffix WxyzAB \
  --case-insensitive \
  --rate-mh 750

# Flexible fixed-width classes
python3 tron_bench.py estimate \
  --prefix '[AB]b?d' \
  --suffix '[Ww]xyz[AB]?' \
  --rate-mh 750 \
  --json
```

Always use the rate from the same full TRON pipeline and matching mode that will
serve requests. Do not substitute SHA/Keccak-only figures or an EVM hex-address
benchmark. The rate must include secp256k1 point generation, Keccak-256, TRON
Base58Check work needed by the matcher, and match checking.

## Why the probability is not simply `29^N`

For ordinary alphabetic characters, case-insensitive matching changes the
per-position ideal probability from `1/58` to `2/58 = 1/29`. Digits do not get
that benefit. The Base58 alphabet also omits `0`, uppercase `I`/`O`, and lowercase
`l`, so the case-fold classes for `i`, `L`, and `o` contain only one valid symbol.

There is a second TRON-specific wrinkle: the first character after `T` is not
uniform across all 58 symbols. The fixed `0x41` network byte constrains it to `9`
or an uppercase Base58 letter, with edge symbols having smaller probability.
`tron_bench.py` handles this with digit dynamic programming over the actual
25-byte TRON Base58 interval instead of charging every position as `1/58`.

Suffix digits include the four-byte Base58Check checksum. The calculator uses
the standard uniform cryptographic-checksum assumption for those digits. Final
architecture decisions must be calibrated against a measured GPU rate and a
small set of real-match trials.

## Target RTX 4090 protocol

1. Use a dedicated, headless worker with persistence mode enabled and no display
   workload. Record exact GPU model, driver, power limit, clocks, temperature,
   CUDA version, generator commit, and build flags.
2. Verify at least 4,096 generated GPU results against an independent CPU TRON
   derivation implementation before collecting speed data.
3. Warm the process and GPU for 60 seconds.
4. Run an unreachable-pattern throughput test for at least 180 seconds. Report
   full TRON addresses tested per second, not raw hash operations.
5. Run reachable reduced-difficulty 2x3 and 3x3 patterns for at least 100 hits.
   Check that observed inter-arrival means agree with the probability model.
6. Project 4x6 latency using the measured sustained rate. A full empirical 4x6
   average is usually economically impractical because match latency follows a
   geometric distribution and has a standard deviation approximately equal to
   its mean.
7. If doing full trials anyway, use at least 30 independent warm trials and keep
   private keys out of logs, telemetry, shell history, and synced files.

Trial data can be summarized from JSON Lines. Each successful row needs
`elapsed_seconds`; `candidates` is optional:

```json
{"gpu":"RTX 4090","pattern":"^TAb...xyz$","found":true,"elapsed_seconds":12.4,"candidates":918273645}
```

```bash
python3 tron_bench.py summarize results/rtx4090.jsonl --json
```

For a generator that exits when it finds one match, the cold end-to-end wrapper
can collect trials directly. Child stdout/stderr are discarded so a printed key
is not copied into the benchmark log. The generator itself must have a
benchmark-only/no-key-file mode; this wrapper cannot stop an external program
from writing its own files.

```bash
python3 tron_bench.py measure \
  --output results/rtx4090-cold.jsonl \
  --trials 30 \
  --warmups 1 \
  --timeout-seconds 3600 \
  --gpu-label 'RTX 4090' \
  --pattern-label '^TAb...xyz$ / ignore-case' \
  -- ./your-generator --benchmark-only --prefix Ab --suffix xyz --ignore-case
```

This measures cold process-to-hit time. A production on-demand service should
keep a GPU worker resident; export that worker's per-request elapsed times as the
same JSONL fields and use `summarize` for the warm latency result.

## Security boundary

Benchmark keys must never receive funds. A production generator should run on a
trusted, isolated worker, use an operating-system CSPRNG for independent secret
starting points, independently re-derive every winning address from its private
key, and send the secret through an authenticated encrypted channel. “Wiping” a
managed-language string is not a reliable memory-erasure guarantee; avoid
creating unnecessary plaintext copies in the first place.
# Tron-wallet-Cuda-Benchmarking
