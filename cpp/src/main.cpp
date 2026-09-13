#include "tron/crypto.hpp"
#include "tron/gpu.hpp"
#include "tron/pattern.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

constexpr int kThreadsPerBlock = 64;
constexpr int kBlocksPerSm = 4;
constexpr std::uint32_t kMaxMatches = 64;
std::mutex output_mutex;

struct Options {
  std::string prefix;
  std::string suffix;
  bool ignore_case = true;
  double warmup_seconds = 10.0;
  double benchmark_seconds = 60.0;
  std::string validation_prefix;
  std::string validation_suffix;
  int validation_hits = 20;
  double validation_timeout = 300.0;
  double full_search_seconds = 0.0;
  double sla_seconds = 30.0;
  std::filesystem::path output = "results/cpp-gpu-live-benchmark.json";
  std::string kernel = TRON_DEFAULT_KERNEL_PATH;
  std::string devices = "all";
  bool debug_math = false;
};

struct Throughput {
  double rate{};
  std::uint64_t candidates{};
  double seconds{};
};

struct Projection {
  long double probability{};
  long double attempts{};
  double mean{};
  double median{};
  double p95{};
  double p99{};
  double probability_within_sla{};
};

struct Validation {
  std::uint64_t candidates{};
  std::uint64_t hits{};
  std::uint64_t verified{};
  double seconds{};
};

struct FullSearch {
  bool attempted{};
  bool found{};
  bool verified{};
  std::uint64_t candidates{};
  double seconds{};
  std::string address;
};

struct DeviceBenchmark {
  tron::GpuInfo gpu;
  double tuning_rate_16{};
  double tuning_rate_8{};
  int points_per_thread{};
  std::size_t chains{};
  int steps{};
  Throughput throughput;
  Validation validation;
  FullSearch full_search;
};

struct MathFactor {
  std::string label;
  unsigned accepted{};
  unsigned domain{};
};

class StartGate {
 public:
  explicit StartGate(std::size_t participants) : participants_(participants) {}

  bool arrive_and_wait() {
    std::unique_lock lock(mutex_);
    ++ready_;
    if (ready_ == participants_) {
      released_ = true;
      condition_.notify_all();
    } else {
      condition_.wait(lock, [&] { return released_ || cancelled_; });
    }
    return !cancelled_;
  }

  void cancel() {
    std::lock_guard lock(mutex_);
    cancelled_ = true;
    condition_.notify_all();
  }

 private:
  std::mutex mutex_;
  std::condition_variable condition_;
  std::size_t participants_{};
  std::size_t ready_{};
  bool released_{};
  bool cancelled_{};
};

[[noreturn]] void usage(int status, std::string_view error = {}) {
  std::ostream& stream = status == 0 ? std::cout : std::cerr;
  if (!error.empty()) stream << "error: " << error << "\n\n";
  stream
      << "Usage: tron_gpu_benchmark --prefix PATTERN --suffix PATTERN [options]\n\n"
      << "Pattern capacity: up to 5 prefix and 6 suffix tokens.\n"
      << "Syntax: Base58 literals, ? wildcards, and [abc] classes.\n"
      << "The prefix starts after TRON's invariant leading T.\n\n"
      << "Options:\n"
      << "  --ignore-case              Match ASCII letter case (default)\n"
      << "  --case-sensitive           Match exact case\n"
      << "  --case-mode MODE           MODE is ignore or exact\n"
      << "  --devices LIST             CUDA ordinals, e.g. 0,1, or all (default all)\n"
      << "  --debug-math               Print probability and latency calculations\n"
      << "  --warmup-seconds N         GPU warm-up duration (default 10)\n"
      << "  --benchmark-seconds N      Throughput measurement (default 60)\n"
      << "  --validation-prefix P      Easier validation prefix (default derived)\n"
      << "  --validation-suffix P      Easier validation suffix (default derived)\n"
      << "  --validation-hits N        Verified hits required (default 20)\n"
      << "  --validation-timeout N     Validation limit (default 300)\n"
      << "  --full-search-seconds N    Optional requested-pattern extraction trial\n"
      << "  --sla-seconds N            SLA probability window (default 30)\n"
      << "  --output PATH              Private-mode JSON result path\n"
      << "  --kernel PATH              CUDA kernel source override\n";
  std::exit(status);
}

std::string require_value(int argc, char** argv, int& index) {
  if (index + 1 >= argc) usage(2, std::string("missing value for ") + argv[index]);
  return argv[++index];
}

double parse_double(const std::string& value, std::string_view option) {
  std::size_t used = 0;
  double result = 0;
  try {
    result = std::stod(value, &used);
  } catch (const std::exception&) {
    usage(2, std::string(option) + " requires a number");
  }
  if (used != value.size() || !std::isfinite(result)) {
    usage(2, std::string(option) + " requires a finite number");
  }
  return result;
}

int parse_int(const std::string& value, std::string_view option) {
  std::size_t used = 0;
  long result = 0;
  try {
    result = std::stol(value, &used);
  } catch (const std::exception&) {
    usage(2, std::string(option) + " requires an integer");
  }
  if (used != value.size() || result < std::numeric_limits<int>::min() ||
      result > std::numeric_limits<int>::max()) {
    usage(2, std::string(option) + " requires an integer");
  }
  return static_cast<int>(result);
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--help" || argument == "-h") usage(0);
    if (argument == "--prefix") options.prefix = require_value(argc, argv, index);
    else if (argument == "--suffix") options.suffix = require_value(argc, argv, index);
    else if (argument == "--ignore-case") options.ignore_case = true;
    else if (argument == "--case-sensitive") options.ignore_case = false;
    else if (argument == "--case-mode") {
      const std::string mode = require_value(argc, argv, index);
      if (mode == "ignore" || mode == "insensitive") options.ignore_case = true;
      else if (mode == "exact" || mode == "sensitive") options.ignore_case = false;
      else usage(2, "--case-mode must be ignore or exact");
    } else if (argument == "--warmup-seconds") {
      options.warmup_seconds = parse_double(require_value(argc, argv, index), argument);
    } else if (argument == "--benchmark-seconds") {
      options.benchmark_seconds = parse_double(require_value(argc, argv, index), argument);
    } else if (argument == "--validation-prefix") {
      options.validation_prefix = require_value(argc, argv, index);
    } else if (argument == "--validation-suffix") {
      options.validation_suffix = require_value(argc, argv, index);
    } else if (argument == "--validation-hits") {
      options.validation_hits = parse_int(require_value(argc, argv, index), argument);
    } else if (argument == "--validation-timeout") {
      options.validation_timeout = parse_double(require_value(argc, argv, index), argument);
    } else if (argument == "--full-search-seconds") {
      options.full_search_seconds = parse_double(require_value(argc, argv, index), argument);
    } else if (argument == "--sla-seconds") {
      options.sla_seconds = parse_double(require_value(argc, argv, index), argument);
    } else if (argument == "--output") {
      options.output = require_value(argc, argv, index);
    } else if (argument == "--kernel") {
      options.kernel = require_value(argc, argv, index);
    } else if (argument == "--devices") {
      options.devices = require_value(argc, argv, index);
    } else if (argument == "--debug-math") {
      options.debug_math = true;
    } else {
      usage(2, "unknown option: " + argument);
    }
  }
  if (options.prefix.empty() || options.suffix.empty()) {
    usage(2, "--prefix and --suffix are required");
  }
  if (options.warmup_seconds <= 0 || options.benchmark_seconds <= 0 ||
      options.validation_hits <= 0 || options.validation_timeout <= 0 ||
      options.full_search_seconds < 0 || options.sla_seconds < 0) {
    usage(2, "durations/hit count are outside their valid range");
  }
  return options;
}

std::vector<int> parse_device_ordinals(std::string_view selection, int available) {
  if (available < 1) throw std::runtime_error("no CUDA GPU detected");
  std::vector<int> devices;
  if (selection == "all") {
    devices.reserve(static_cast<std::size_t>(available));
    for (int ordinal = 0; ordinal < available; ++ordinal) devices.push_back(ordinal);
    return devices;
  }

  std::size_t start = 0;
  while (start <= selection.size()) {
    const std::size_t comma = selection.find(',', start);
    const std::size_t end = comma == std::string_view::npos ? selection.size() : comma;
    if (end == start) usage(2, "--devices contains an empty device ordinal");
    const int ordinal = parse_int(std::string(selection.substr(start, end - start)), "--devices");
    if (ordinal < 0 || ordinal >= available) {
      usage(2, "--devices ordinal is outside the visible CUDA device range");
    }
    if (std::find(devices.begin(), devices.end(), ordinal) != devices.end()) {
      usage(2, "--devices contains a duplicate device ordinal");
    }
    devices.push_back(ordinal);
    if (comma == std::string_view::npos) break;
    start = comma + 1U;
  }
  if (devices.empty()) usage(2, "--devices requires all or a comma-separated list");
  return devices;
}

void device_log(int ordinal, const std::string& message) {
  std::lock_guard lock(output_mutex);
  std::cout << "[GPU " << ordinal << "] " << message << '\n' << std::flush;
}

std::vector<MathFactor> probability_factors(const Options& options) {
  const auto prefix = tron::parse_fixed_pattern(options.prefix, options.ignore_case);
  const auto suffix = tron::parse_fixed_pattern(options.suffix, options.ignore_case);
  if (prefix.empty()) throw tron::PatternError("prefix must contain at least one token");

  tron::Token second_character_domain = 0;
  constexpr std::string_view possible_second = "9ABCDEFGHJKLMNPQRSTUVWXYZ";
  for (const char character : possible_second) {
    second_character_domain |=
        tron::Token{1} << static_cast<unsigned>(tron::base58_index(character));
  }

  std::vector<MathFactor> factors;
  factors.reserve(prefix.size() + suffix.size());
  factors.push_back(MathFactor{
      "prefix[1] immediately after T",
      static_cast<unsigned>(std::popcount(prefix.front() & second_character_domain)), 25});
  for (std::size_t index = 1; index < prefix.size(); ++index) {
    factors.push_back(MathFactor{
        "prefix[" + std::to_string(index + 1U) + "]",
        static_cast<unsigned>(std::popcount(prefix[index])), 58});
  }
  for (std::size_t index = 0; index < suffix.size(); ++index) {
    factors.push_back(MathFactor{
        "suffix[" + std::to_string(index + 1U) + "]",
        static_cast<unsigned>(std::popcount(suffix[index])), 58});
  }
  return factors;
}

Projection project(const tron::Probability& probability, double rate, double sla_seconds) {
  const auto quantile_seconds = [&](long double quantile) {
    if (probability.probability == 1.0L) return 1.0 / rate;
    const long double attempts =
        std::ceil(std::log1pl(-quantile) / std::log1pl(-probability.probability));
    return static_cast<double>(attempts / static_cast<long double>(rate));
  };
  const long double sla_attempts =
      std::floor(static_cast<long double>(rate) * static_cast<long double>(sla_seconds));
  const long double sla_probability =
      -std::expm1l(sla_attempts * std::log1pl(-probability.probability));
  return Projection{probability.probability,
                    probability.expected_attempts,
                    static_cast<double>(probability.expected_attempts / rate),
                    quantile_seconds(0.5L),
                    quantile_seconds(0.95L),
                    quantile_seconds(0.99L),
                    static_cast<double>(sla_probability)};
}

void print_math_debug(const Options& options,
                      const std::vector<DeviceBenchmark>& devices,
                      const Throughput& throughput,
                      const Projection& projection) {
  const auto prefix = tron::parse_fixed_pattern(options.prefix, options.ignore_case);
  const auto suffix = tron::parse_fixed_pattern(options.suffix, options.ignore_case);
  const auto factors = probability_factors(options);
  const std::size_t middle_characters =
      34U - 1U - prefix.size() - suffix.size();

  long double calculated_probability = 1.0L;
  std::ostringstream formula;
  formula << "1";
  for (const auto& factor : factors) {
    calculated_probability *=
        static_cast<long double>(factor.accepted) /
        static_cast<long double>(factor.domain);
    formula << " * (" << factor.accepted << '/' << factor.domain << ')';
  }
  formula << " * 1";

  const long double difference =
      std::abs(calculated_probability - projection.probability);
  const long double tolerance =
      std::max(1e-30L, std::abs(projection.probability) * 1e-15L);

  std::cout << "\n========== MATH DEBUG ==========\n"
            << "Address layout     : T(1) + prefix(" << prefix.size()
            << ") + middle(" << middle_characters << ") + suffix("
            << suffix.size() << ") = 34 characters\n"
            << "Base58 alphabet    : 58 symbols\n"
            << "Second-char domain : 25 symbols (simplified TRON model)\n"
            << "Case mode          : "
            << (options.ignore_case ? "case-insensitive" : "case-sensitive")
            << "\n\nProbability factors:\n"
            << "  leading T                     = 1 (always present)\n";
  for (const auto& factor : factors) {
    std::cout << "  " << std::left << std::setw(29) << factor.label << std::right
              << " = " << factor.accepted << '/' << factor.domain;
    if (factor.accepted == factor.domain) std::cout << " = 1 (unrestricted)";
    std::cout << '\n';
  }
  std::cout << "  middle[" << middle_characters
            << "] unrestricted          = 1\n"
            << "\nProbability:\n"
            << "  p = " << formula.str() << '\n'
            << std::scientific << std::setprecision(12)
            << "  p = " << calculated_probability << " per candidate\n"
            << std::fixed << std::setprecision(3)
            << "  E = 1 / p = " << projection.attempts << " attempts\n"
            << "  Probability-model check = "
            << (difference <= tolerance ? "PASS" : "FAIL") << "\n\n"
            << "Measured throughput:\n";

  for (const auto& device : devices) {
    std::cout << "  R_gpu" << device.gpu.ordinal << " = "
              << device.throughput.candidates << " / "
              << device.throughput.seconds << " = "
              << device.throughput.rate << " addr/s\n";
  }
  std::cout << "  R_total = ";
  for (std::size_t index = 0; index < devices.size(); ++index) {
    if (index != 0U) std::cout << " + ";
    std::cout << devices[index].throughput.rate;
  }
  std::cout << " = " << throughput.rate << " addr/s\n\n"
            << "CUDA observation batches:\n";
  for (const auto& device : devices) {
    const std::uint64_t candidates_per_launch =
        static_cast<std::uint64_t>(device.chains) *
        static_cast<std::uint64_t>(device.steps);
    const double approximate_launch_seconds =
        static_cast<double>(candidates_per_launch) / device.throughput.rate;
    const long double expected_hits_per_launch =
        static_cast<long double>(candidates_per_launch) * projection.probability;
    std::cout << "  GPU " << device.gpu.ordinal << ": "
              << candidates_per_launch << " candidates/launch, approximately "
              << approximate_launch_seconds << "s/launch, expected hits/launch = "
              << static_cast<double>(expected_hits_per_launch) << '\n';
  }
  std::cout << "  Note: E/R assumes candidates are observed continuously; live detection\n"
            << "        is returned at a CUDA launch boundary and can be batch-limited.\n\n"
            << "Latency projection:\n"
            << "  mean   = E / R_total = " << projection.attempts << " / "
            << throughput.rate << " = " << projection.mean << " seconds\n";
  if (projection.probability < 1.0L) {
    std::cout << "  median = ceil(ln(1-0.50) / ln(1-p)) / R_total = "
              << projection.median << " seconds\n"
              << "  p95    = ceil(ln(1-0.95) / ln(1-p)) / R_total = "
              << projection.p95 << " seconds\n";
  } else {
    std::cout << "  median = one attempt / R_total = " << projection.median
              << " seconds\n"
              << "  p95    = one attempt / R_total = " << projection.p95
              << " seconds\n";
  }
  std::cout << "  P(hit <= " << options.sla_seconds << "s) = "
            << projection.probability_within_sla * 100.0 << "%\n"
            << "================================\n" << std::defaultfloat;
}

int calibrate_steps(tron::GpuRunner& runner) {
  constexpr int sample_steps = 16;
  const auto sample = runner.launch(sample_steps);
  const long long proposed = std::llround(sample_steps * 0.5 / std::max(sample.elapsed_seconds, 1e-6));
  return static_cast<int>(std::clamp<long long>(proposed, 8, 4096));
}

Throughput timed_throughput(tron::GpuRunner& runner, int steps, double requested_seconds) {
  const auto started = std::chrono::steady_clock::now();
  std::uint64_t candidates = 0;
  double elapsed = 0;
  do {
    runner.launch(steps);
    candidates += static_cast<std::uint64_t>(runner.chains()) *
                  static_cast<std::uint64_t>(steps);
    elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  } while (elapsed < requested_seconds);
  return Throughput{static_cast<double>(candidates) / elapsed, candidates, elapsed};
}

std::uint64_t verify_records(const tron::GpuRunner& runner,
                             const tron::StartPoints& points,
                             const std::vector<tron::MatchRecord>& records,
                             std::string_view prefix, std::string_view suffix,
                             bool ignore_case) {
  const auto prefix_tokens = tron::parse_fixed_pattern(prefix, ignore_case);
  const auto suffix_tokens = tron::parse_fixed_pattern(suffix, ignore_case);
  std::uint64_t verified = 0;
  for (const auto& record : records) {
    const std::uint32_t thread = record.thread_id >> 4U;
    const std::uint32_t point = record.thread_id & 0xFU;
    if (thread >= static_cast<std::uint32_t>(runner.blocks() * kThreadsPerBlock) ||
        point >= static_cast<std::uint32_t>(runner.points_per_thread())) {
      throw std::runtime_error("GPU returned an invalid chain identifier");
    }
    const std::size_t base_index =
        static_cast<std::size_t>(thread) * static_cast<std::size_t>(runner.points_per_thread()) +
        point;
    const std::string gpu_address(record.address, sizeof(record.address));
    const std::string cpu_address =
        tron::address_for_offset(points.private_keys.at(base_index), record.step);
    if (gpu_address != cpu_address) {
      throw std::runtime_error("GPU/CPU derivation mismatch for public address " + gpu_address);
    }
    if (!tron::matches(gpu_address, prefix_tokens, suffix_tokens)) {
      throw std::runtime_error("GPU matcher returned a false positive: " + gpu_address);
    }
    ++verified;
  }
  return verified;
}

Validation validate(tron::GpuRunner& runner, const tron::StartPoints& points,
                    const tron::PatternParams& pattern, std::string_view prefix,
                    std::string_view suffix, bool ignore_case, int steps,
                    int wanted_hits, double timeout_seconds) {
  runner.set_pattern(pattern);
  const auto started = std::chrono::steady_clock::now();
  Validation result;
  while (result.hits < static_cast<std::uint64_t>(wanted_hits)) {
    result.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    if (result.seconds >= timeout_seconds) break;
    const auto launch = runner.launch(steps);
    result.candidates += static_cast<std::uint64_t>(runner.chains()) *
                         static_cast<std::uint64_t>(steps);
    if (launch.count > kMaxMatches) {
      throw std::runtime_error("match buffer overflow; use a harder validation pattern");
    }
    result.hits += launch.count;
    result.verified +=
        verify_records(runner, points, launch.records, prefix, suffix, ignore_case);
  }
  result.seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  return result;
}

FullSearch search_requested(tron::GpuRunner& runner, const tron::StartPoints& points,
                            const tron::PatternParams& pattern, std::string_view prefix,
                            std::string_view suffix, bool ignore_case, int steps,
                            double timeout_seconds,
                            std::atomic<bool>* another_device_found = nullptr) {
  FullSearch result;
  if (timeout_seconds <= 0) return result;
  result.attempted = true;
  runner.set_pattern(pattern);
  const auto started = std::chrono::steady_clock::now();
  while (true) {
    if (another_device_found != nullptr &&
        another_device_found->load(std::memory_order_acquire)) {
      break;
    }
    result.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    if (result.seconds >= timeout_seconds) break;
    const auto launch = runner.launch(steps);
    result.candidates += static_cast<std::uint64_t>(runner.chains()) *
                         static_cast<std::uint64_t>(steps);
    if (launch.count > kMaxMatches) throw std::runtime_error("requested match buffer overflow");
    if (launch.count > 0) {
      result.verified =
          verify_records(runner, points, launch.records, prefix, suffix, ignore_case) > 0;
      result.found = result.verified;
      if (result.found) {
        result.address.assign(launch.records.front().address, 34);
        if (another_device_found != nullptr) {
          another_device_found->store(true, std::memory_order_release);
        }
      }
      break;
    }
  }
  result.seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  return result;
}

DeviceBenchmark benchmark_device(const Options& options,
                                 const tron::PatternParams& requested,
                                 const tron::PatternParams& validation_pattern,
                                 std::string_view validation_prefix,
                                 std::string_view validation_suffix,
                                 int validation_hits, int ordinal,
                                 StartGate& throughput_gate,
                                 std::atomic<bool>& full_search_stop) {
  DeviceBenchmark result;
  tron::CudaContext context(ordinal);
  result.gpu = context.info();

  {
    std::ostringstream message;
    message << result.gpu.name << "; architecture sm_" << result.gpu.major
            << result.gpu.minor << "; " << result.gpu.sm_count << " SMs";
    device_log(ordinal, message.str());
  }

  result.tuning_rate_16 =
      tron::tune_points_per_thread(context, options.kernel, requested, 16);
  {
    std::ostringstream message;
    message << std::fixed << std::setprecision(3) << "M=16: "
            << result.tuning_rate_16 / 1e6 << " M addr/s tuning sample";
    device_log(ordinal, message.str());
  }
  result.tuning_rate_8 =
      tron::tune_points_per_thread(context, options.kernel, requested, 8);
  {
    std::ostringstream message;
    message << std::fixed << std::setprecision(3) << "M=8: "
            << result.tuning_rate_8 / 1e6 << " M addr/s tuning sample";
    device_log(ordinal, message.str());
  }

  result.points_per_thread =
      result.tuning_rate_16 >= result.tuning_rate_8 ? 16 : 8;
  const int blocks = result.gpu.sm_count * kBlocksPerSm;
  result.chains = static_cast<std::size_t>(blocks) * kThreadsPerBlock *
                  static_cast<std::size_t>(result.points_per_thread);
  {
    std::ostringstream message;
    message << "Preparing " << result.chains << " independent GPU chains...";
    device_log(ordinal, message.str());
  }

  auto points = tron::create_start_points(result.chains, true);
  tron::GpuRunner runner(context, options.kernel, result.points_per_thread, blocks,
                         points.x, points.y);
  runner.set_pattern(requested);
  result.steps = calibrate_steps(runner);
  {
    std::ostringstream message;
    message << "Selected M=" << result.points_per_thread << "; " << result.chains
            << " chains; " << result.steps << " steps/launch";
    device_log(ordinal, message.str());
  }

  {
    std::ostringstream message;
    message << "Warming GPU for " << options.warmup_seconds << " seconds...";
    device_log(ordinal, message.str());
  }
  timed_throughput(runner, result.steps, options.warmup_seconds);
  device_log(ordinal, "Ready for synchronized multi-GPU measurement.");
  if (!throughput_gate.arrive_and_wait()) {
    throw std::runtime_error("multi-GPU measurement cancelled after another device failed");
  }
  {
    std::ostringstream message;
    message << "Measuring sustained throughput for " << options.benchmark_seconds
            << " seconds...";
    device_log(ordinal, message.str());
  }
  result.throughput =
      timed_throughput(runner, result.steps, options.benchmark_seconds);

  {
    std::ostringstream message;
    message << "Validating ^T" << validation_prefix << "..." << validation_suffix
            << "$ (up to " << options.validation_timeout << "s)...";
    device_log(ordinal, message.str());
  }
  result.validation =
      validate(runner, points, validation_pattern, validation_prefix,
               validation_suffix, options.ignore_case, result.steps,
               validation_hits, options.validation_timeout);
  result.full_search =
      search_requested(runner, points, requested, options.prefix, options.suffix,
                       options.ignore_case, result.steps,
                       options.full_search_seconds, &full_search_stop);
  points.wipe_private_keys();

  {
    std::ostringstream message;
    message << std::fixed << std::setprecision(3) << "Completed: "
            << result.throughput.rate / 1e6 << " M addr/s; validation "
            << result.validation.verified << "/" << result.validation.hits;
    device_log(ordinal, message.str());
  }
  return result;
}

std::string json_escape(std::string_view input) {
  std::ostringstream output;
  for (const unsigned char character : input) {
    switch (character) {
      case '\\': output << "\\\\"; break;
      case '"': output << "\\\""; break;
      case '\n': output << "\\n"; break;
      case '\r': output << "\\r"; break;
      case '\t': output << "\\t"; break;
      default:
        if (character < 0x20U) {
          output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                 << static_cast<unsigned>(character) << std::dec;
        } else {
          output << static_cast<char>(character);
        }
    }
  }
  return output.str();
}

std::string timestamp_utc() {
  const std::time_t current = std::time(nullptr);
  std::tm broken_down{};
  gmtime_r(&current, &broken_down);
  std::ostringstream output;
  output << std::put_time(&broken_down, "%Y-%m-%dT%H:%M:%SZ");
  return output.str();
}

void write_private_json(const std::filesystem::path& path, const std::string& contents) {
  if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
  const int descriptor = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
  if (descriptor < 0) throw std::runtime_error("cannot create result file: " + path.string());
  if (::fchmod(descriptor, 0600) != 0) {
    ::close(descriptor);
    throw std::runtime_error("cannot protect result file: " + path.string());
  }
  std::size_t written = 0;
  while (written < contents.size()) {
    const ssize_t count = ::write(descriptor, contents.data() + written, contents.size() - written);
    if (count <= 0) {
      ::close(descriptor);
      throw std::runtime_error("cannot write result file: " + path.string());
    }
    written += static_cast<std::size_t>(count);
  }
  if (::close(descriptor) != 0) throw std::runtime_error("cannot close result file");
}

std::string build_json(const Options& options,
                       const std::vector<DeviceBenchmark>& devices,
                       const Throughput& throughput, const Projection& projection,
                       std::string_view validation_prefix, std::string_view validation_suffix,
                       const Validation& validation, const FullSearch& full_search) {
  std::ostringstream json;
  json << std::setprecision(17);
  json << "{\n"
       << "  \"schema_version\": 6,\n"
       << "  \"implementation\": \"c++20-cuda-driver-nvrtc-multi-gpu\",\n"
       << "  \"timestamp_utc\": \"" << timestamp_utc() << "\",\n"
       << "  \"device_count\": " << devices.size() << ",\n"
       << "  \"devices\": [\n";
  for (std::size_t index = 0; index < devices.size(); ++index) {
    const auto& device = devices[index];
    json << "    {\"ordinal\": " << device.gpu.ordinal
         << ", \"name\": \"" << json_escape(device.gpu.name)
         << "\", \"cuda_driver_api_version\": "
         << device.gpu.cuda_driver_api_version
         << ", \"cuda_architecture\": \"sm_" << device.gpu.major
         << device.gpu.minor << "\", \"sm_count\": " << device.gpu.sm_count
         << ", \"points_per_thread\": " << device.points_per_thread
         << ", \"chains\": " << device.chains
         << ", \"steps_per_launch\": " << device.steps
         << ", \"tuning_million_addresses_per_second\": {\"m16\": "
         << device.tuning_rate_16 / 1e6 << ", \"m8\": "
         << device.tuning_rate_8 / 1e6 << "}"
         << ", \"throughput\": {\"candidates\": "
         << device.throughput.candidates << ", \"elapsed_seconds\": "
         << device.throughput.seconds
         << ", \"full_tron_addresses_per_second\": "
         << device.throughput.rate
         << ", \"million_addresses_per_second\": "
         << device.throughput.rate / 1e6 << "}"
         << ", \"validation\": {\"hits\": " << device.validation.hits
         << ", \"verified_hits\": " << device.validation.verified
         << ", \"candidates\": " << device.validation.candidates
         << ", \"elapsed_seconds\": " << device.validation.seconds << "}"
         << ", \"full_search\": {\"attempted\": "
         << (device.full_search.attempted ? "true" : "false")
         << ", \"found\": " << (device.full_search.found ? "true" : "false")
         << ", \"verified\": "
         << (device.full_search.verified ? "true" : "false")
         << ", \"candidates\": " << device.full_search.candidates
         << ", \"elapsed_seconds\": " << device.full_search.seconds
         << ", \"public_address\": ";
    if (device.full_search.address.empty()) json << "null";
    else json << "\"" << json_escape(device.full_search.address) << "\"";
    json << "}}" << (index + 1U == devices.size() ? "\n" : ",\n");
  }
  json << "  ],\n"
       << "  \"requested_pattern\": {\"prefix_after_t\": \""
       << json_escape(options.prefix) << "\", \"suffix\": \""
       << json_escape(options.suffix) << "\", \"ignore_case\": "
       << (options.ignore_case ? "true" : "false") << "},\n"
       << "  \"throughput\": {\"aggregation\": \"sum_of_device_rates\""
       << ", \"candidates\": " << throughput.candidates
       << ", \"rate_equivalent_elapsed_seconds\": " << throughput.seconds
       << ", \"full_tron_addresses_per_second\": " << throughput.rate
       << ", \"million_addresses_per_second\": " << throughput.rate / 1e6 << "},\n"
       << "  \"projection\": {\"probability_model\": "
          "\"uniform-25-second-character-and-uniform-58-later-positions\", "
          "\"probability_per_candidate\": "
       << static_cast<double>(projection.probability)
       << ", \"expected_attempts\": " << static_cast<double>(projection.attempts)
       << ", \"mean_seconds\": " << projection.mean
       << ", \"mean_hours\": " << projection.mean / 3600.0
       << ", \"mean_days\": " << projection.mean / 86400.0
       << ", \"median_seconds\": " << projection.median
       << ", \"p95_seconds\": " << projection.p95
       << ", \"p99_seconds\": " << projection.p99
       << ", \"sla_seconds\": " << options.sla_seconds
       << ", \"probability_within_sla\": " << projection.probability_within_sla << "},\n"
       << "  \"validation_pattern\": {\"prefix_after_t\": \""
       << json_escape(validation_prefix) << "\", \"suffix\": \""
       << json_escape(validation_suffix) << "\", \"hits\": " << validation.hits
       << ", \"verified_hits\": " << validation.verified
       << ", \"candidates\": " << validation.candidates
       << ", \"elapsed_seconds\": " << validation.seconds << "},\n"
       << "  \"full_search\": {\"attempted\": "
       << (full_search.attempted ? "true" : "false")
       << ", \"found\": " << (full_search.found ? "true" : "false")
       << ", \"verified\": " << (full_search.verified ? "true" : "false")
       << ", \"candidates\": " << full_search.candidates
       << ", \"elapsed_seconds\": " << full_search.seconds
       << ", \"public_address\": ";
  if (full_search.address.empty()) json << "null";
  else json << "\"" << json_escape(full_search.address) << "\"";
  json << "},\n  \"private_keys_written_to_disk\": false\n}\n";
  return json.str();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const auto requested = tron::compile_pattern(options.prefix, options.suffix,
                                                 options.ignore_case, false);
    const auto probability = tron::match_probability_simplified(
        options.prefix, options.suffix, options.ignore_case);
    const std::string validation_prefix =
        options.validation_prefix.empty()
            ? tron::representative_literals(options.prefix, options.ignore_case, 2, true)
            : options.validation_prefix;
    const std::string validation_suffix =
        options.validation_suffix.empty()
            ? tron::representative_literals(options.suffix, options.ignore_case, 3)
            : options.validation_suffix;
    const auto validation_pattern = tron::compile_pattern(
        validation_prefix, validation_suffix, options.ignore_case, false);

    const int available_devices = tron::cuda_device_count();
    const std::vector<int> ordinals =
        parse_device_ordinals(options.devices, available_devices);
    std::cout << "Using " << ordinals.size() << " of " << available_devices
              << " visible CUDA GPU(s): ";
    for (std::size_t index = 0; index < ordinals.size(); ++index) {
      if (index != 0U) std::cout << ',';
      std::cout << ordinals[index];
    }
    std::cout << '\n';

    std::vector<DeviceBenchmark> devices(ordinals.size());
    std::vector<std::exception_ptr> failures(ordinals.size());
    std::vector<std::thread> workers;
    StartGate throughput_gate(ordinals.size());
    std::atomic<bool> full_search_stop{false};
    workers.reserve(ordinals.size());
    const int validation_hits_per_device = std::max(
        1, (options.validation_hits + static_cast<int>(ordinals.size()) - 1) /
               static_cast<int>(ordinals.size()));
    for (std::size_t index = 0; index < ordinals.size(); ++index) {
      workers.emplace_back([&, index] {
        try {
          devices[index] = benchmark_device(
              options, requested, validation_pattern, validation_prefix,
              validation_suffix, validation_hits_per_device, ordinals[index],
              throughput_gate, full_search_stop);
        } catch (...) {
          failures[index] = std::current_exception();
          throughput_gate.cancel();
        }
      });
    }
    for (auto& worker : workers) worker.join();
    for (std::size_t index = 0; index < failures.size(); ++index) {
      if (!failures[index]) continue;
      try {
        std::rethrow_exception(failures[index]);
      } catch (const std::exception& error) {
        throw std::runtime_error("GPU " + std::to_string(ordinals[index]) +
                                 " failed: " + error.what());
      } catch (...) {
        throw std::runtime_error("GPU " + std::to_string(ordinals[index]) +
                                 " failed with an unknown error");
      }
    }

    Throughput throughput;
    Validation validation;
    FullSearch full_search;
    full_search.attempted = options.full_search_seconds > 0;
    double earliest_found = std::numeric_limits<double>::infinity();
    for (const auto& device : devices) {
      throughput.rate += device.throughput.rate;
      throughput.candidates += device.throughput.candidates;
      validation.candidates += device.validation.candidates;
      validation.hits += device.validation.hits;
      validation.verified += device.validation.verified;
      validation.seconds = std::max(validation.seconds, device.validation.seconds);
      full_search.candidates += device.full_search.candidates;
      full_search.seconds = std::max(full_search.seconds, device.full_search.seconds);
      if (device.full_search.found && device.full_search.seconds < earliest_found) {
        earliest_found = device.full_search.seconds;
        full_search.found = true;
        full_search.verified = device.full_search.verified;
        full_search.address = device.full_search.address;
      }
    }
    throughput.seconds = throughput.rate > 0
                             ? static_cast<double>(throughput.candidates) /
                                   throughput.rate
                             : 0.0;
    if (full_search.found) full_search.seconds = earliest_found;
    const Projection projection =
        project(probability, throughput.rate, options.sla_seconds);

    write_private_json(
        options.output,
        build_json(options, devices, throughput, projection, validation_prefix,
                   validation_suffix, validation, full_search));

    std::cout << "\nRequested pattern : ^T" << options.prefix << "..." << options.suffix
              << "$ (" << (options.ignore_case ? "case-insensitive" : "case-sensitive")
              << ")\n"
              << "Probability model : 1/25 second character; allowed-cases/58 later\n"
              << std::scientific << std::setprecision(12)
              << "Match chance      : " << projection.probability << " per candidate\n"
              << std::fixed << std::setprecision(3)
              << "Expected attempts : " << projection.attempts << " addresses\n"
              << "GPU count         : " << devices.size() << "\n";
    for (const auto& device : devices) {
      std::cout << "GPU " << device.gpu.ordinal << " rate        : "
                << device.throughput.rate / 1e6 << " M full TRON addr/s\n";
    }
    std::cout << "Combined GPU rate : " << throughput.rate / 1e6
              << " M full TRON addr/s\n"
              << "Projected mean    : " << projection.mean << " seconds\n"
              << "Mean hours        : " << projection.mean / 3600.0 << " hours\n"
              << "Mean days         : " << projection.mean / 86400.0 << " days\n"
              << "Projected median  : " << projection.median << " seconds\n"
              << "Projected p95     : " << projection.p95 << " seconds\n"
              << "Validation         : " << validation.verified << "/" << validation.hits
              << " GPU hits verified on CPU\n";
    if (full_search.attempted) {
      std::cout << "Full pattern trial : "
                << (full_search.found ? "verified hit" : "no hit before timeout")
                << " after " << full_search.seconds << " seconds\n";
    }
    if (options.debug_math) {
      print_math_debug(options, devices, throughput, projection);
    }
    std::cout << "Result             : " << options.output.string() << "\n";
    return 0;
  } catch (const tron::PatternError& error) {
    std::cerr << "invalid pattern: " << error.what() << '\n';
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
  }
  return 1;
}
