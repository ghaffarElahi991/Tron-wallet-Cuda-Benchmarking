#include "tron/crypto.hpp"
#include "tron/gpu.hpp"
#include "tron/pattern.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr int kThreadsPerBlock = 64;
constexpr int kBlocksPerSm = 4;
constexpr std::uint32_t kMaxMatches = 64;

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
  std::vector<std::string> loose_rules;
};

struct EffectivePattern {
  std::string prefix;
  std::string suffix;
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

[[noreturn]] void usage(int status, std::string_view error = {}) {
  std::ostream& stream = status == 0 ? std::cout : std::cerr;
  if (!error.empty()) stream << "error: " << error << "\n\n";
  stream
      << "Usage: tron_gpu_benchmark --prefix PATTERN --suffix PATTERN [options]\n\n"
      << "Required 4x6 pattern syntax: Base58 literals, ? wildcards, [abc] classes.\n"
      << "The prefix starts after TRON's invariant leading T.\n\n"
      << "Options:\n"
      << "  --ignore-case              Match ASCII letter case (default)\n"
      << "  --case-sensitive           Match exact case\n"
      << "  --case-mode MODE           MODE is ignore or exact\n"
      << "  --loose-rule RULE          Repeatable position override; examples:\n"
      << "                              prefix:4=?\n"
      << "                              suffix:6=[SsZz]\n"
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
    } else if (argument == "--loose-rule") {
      options.loose_rules.push_back(require_value(argc, argv, index));
    }
    else if (argument == "--warmup-seconds") {
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

std::vector<std::string> split_pattern_tokens(std::string_view pattern,
                                              bool ignore_case) {
  // Validate with the canonical parser first, then retain each token's source
  // spelling so unmodified positions remain readable in reports.
  const auto parsed = tron::parse_fixed_pattern(pattern, ignore_case);
  std::vector<std::string> tokens;
  tokens.reserve(parsed.size());
  for (std::size_t index = 0; index < pattern.size();) {
    if (pattern[index] == '[') {
      const std::size_t close = pattern.find(']', index + 1U);
      tokens.emplace_back(pattern.substr(index, close - index + 1U));
      index = close + 1U;
    } else {
      tokens.emplace_back(1, pattern[index]);
      ++index;
    }
  }
  return tokens;
}

std::string join_pattern_tokens(const std::vector<std::string>& tokens) {
  std::string result;
  for (const auto& token : tokens) result += token;
  return result;
}

EffectivePattern apply_loose_rules(const Options& options) {
  auto prefix = split_pattern_tokens(options.prefix, options.ignore_case);
  auto suffix = split_pattern_tokens(options.suffix, options.ignore_case);

  for (const std::string& rule : options.loose_rules) {
    const std::size_t colon = rule.find(':');
    const std::size_t equals = rule.find('=', colon == std::string::npos ? 0U : colon + 1U);
    if (colon == std::string::npos || equals == std::string::npos ||
        colon == 0U || equals <= colon + 1U || equals + 1U >= rule.size()) {
      throw tron::PatternError(
          "invalid --loose-rule; use prefix:N=? or suffix:N=[characters]");
    }

    const std::string area = rule.substr(0, colon);
    const std::string position_text = rule.substr(colon + 1U, equals - colon - 1U);
    const std::string replacement = rule.substr(equals + 1U);
    std::size_t used = 0;
    unsigned long position = 0;
    try {
      position = std::stoul(position_text, &used);
    } catch (const std::exception&) {
      throw tron::PatternError("loose-rule position must be a positive integer");
    }
    if (used != position_text.size() || position == 0U) {
      throw tron::PatternError("loose-rule position must be a positive integer");
    }
    const auto replacement_tokens =
        tron::parse_fixed_pattern(replacement, options.ignore_case);
    if (replacement_tokens.size() != 1U) {
      throw tron::PatternError("a loose-rule replacement must consume exactly one character");
    }

    std::vector<std::string>* target = nullptr;
    if (area == "prefix") target = &prefix;
    else if (area == "suffix") target = &suffix;
    else throw tron::PatternError("loose-rule area must be prefix or suffix");
    if (position > target->size()) {
      throw tron::PatternError("loose-rule position is outside the selected pattern");
    }
    (*target)[position - 1U] = replacement;
  }

  return EffectivePattern{join_pattern_tokens(prefix), join_pattern_tokens(suffix)};
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
                            double timeout_seconds) {
  FullSearch result;
  if (timeout_seconds <= 0) return result;
  result.attempted = true;
  runner.set_pattern(pattern);
  const auto started = std::chrono::steady_clock::now();
  while (true) {
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
      if (result.found) result.address.assign(launch.records.front().address, 34);
      break;
    }
  }
  result.seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
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

std::string build_json(const Options& options, const tron::GpuInfo& gpu,
                       std::string_view effective_prefix,
                       std::string_view effective_suffix, int points_per_thread,
                       std::size_t chains, int steps,
                       const Throughput& throughput, const Projection& projection,
                       std::string_view validation_prefix, std::string_view validation_suffix,
                       const Validation& validation, const FullSearch& full_search) {
  std::ostringstream json;
  json << std::setprecision(17);
  json << "{\n"
       << "  \"schema_version\": 4,\n"
       << "  \"implementation\": \"c++20-cuda-driver-nvrtc\",\n"
       << "  \"timestamp_utc\": \"" << timestamp_utc() << "\",\n"
       << "  \"gpu\": {\"name\": \"" << json_escape(gpu.name)
       << "\", \"cuda_driver_api_version\": " << gpu.cuda_driver_api_version << "},\n"
       << "  \"cuda_architecture\": \"sm_" << gpu.major << gpu.minor << "\",\n"
       << "  \"sm_count\": " << gpu.sm_count << ",\n"
       << "  \"points_per_thread\": " << points_per_thread << ",\n"
       << "  \"chains\": " << chains << ",\n"
       << "  \"steps_per_launch\": " << steps << ",\n"
       << "  \"requested_pattern\": {\"base_prefix_after_t\": \""
       << json_escape(options.prefix) << "\", \"base_suffix\": \""
       << json_escape(options.suffix) << "\", \"prefix_after_t\": \""
       << json_escape(effective_prefix) << "\", \"suffix\": \""
       << json_escape(effective_suffix) << "\", \"ignore_case\": "
       << (options.ignore_case ? "true" : "false") << ", \"loose_rules\": [";
  for (std::size_t index = 0; index < options.loose_rules.size(); ++index) {
    if (index != 0U) json << ", ";
    json << "\"" << json_escape(options.loose_rules[index]) << "\"";
  }
  json << "]},\n"
       << "  \"throughput\": {\"candidates\": " << throughput.candidates
       << ", \"elapsed_seconds\": " << throughput.seconds
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
    const EffectivePattern effective = apply_loose_rules(options);
    const auto requested = tron::compile_pattern(effective.prefix, effective.suffix,
                                                 options.ignore_case, true);
    const auto probability = tron::match_probability_simplified(
        effective.prefix, effective.suffix, options.ignore_case);
    const std::string validation_prefix =
        options.validation_prefix.empty()
            ? tron::representative_literals(effective.prefix, options.ignore_case, 2, true)
            : options.validation_prefix;
    const std::string validation_suffix =
        options.validation_suffix.empty()
            ? tron::representative_literals(effective.suffix, options.ignore_case, 3)
            : options.validation_suffix;
    const auto validation_pattern = tron::compile_pattern(
        validation_prefix, validation_suffix, options.ignore_case, false);

    tron::CudaContext context;
    const auto& gpu = context.info();
    std::cout << "GPU: " << gpu.name << "; architecture sm_" << gpu.major << gpu.minor
              << "; " << gpu.sm_count << " SMs\n";
    const double rate16 =
        tron::tune_points_per_thread(context, options.kernel, requested, 16);
    std::cout << "M=16: " << std::fixed << std::setprecision(3) << rate16 / 1e6
              << " M addr/s tuning sample\n";
    const double rate8 = tron::tune_points_per_thread(context, options.kernel, requested, 8);
    std::cout << "M=8: " << rate8 / 1e6 << " M addr/s tuning sample\n";
    const int points_per_thread = rate16 >= rate8 ? 16 : 8;
    const int blocks = gpu.sm_count * kBlocksPerSm;
    const std::size_t chains = static_cast<std::size_t>(blocks) * kThreadsPerBlock *
                               static_cast<std::size_t>(points_per_thread);

    std::cout << "Preparing " << chains << " independent GPU chains...\n";
    auto points = tron::create_start_points(chains, true);
    tron::GpuRunner runner(context, options.kernel, points_per_thread, blocks, points.x,
                           points.y);
    runner.set_pattern(requested);
    const int steps = calibrate_steps(runner);
    std::cout << "Selected M=" << points_per_thread << "; " << chains
              << " chains; " << steps << " steps/launch\n";

    std::cout << "Warming GPU for " << options.warmup_seconds << " seconds...\n";
    timed_throughput(runner, steps, options.warmup_seconds);
    std::cout << "Measuring sustained throughput for " << options.benchmark_seconds
              << " seconds...\n";
    const Throughput throughput =
        timed_throughput(runner, steps, options.benchmark_seconds);
    const Projection projection =
        project(probability, throughput.rate, options.sla_seconds);

    std::cout << "Validating real matches with ^T" << validation_prefix << "..."
              << validation_suffix << "$ (up to " << options.validation_timeout
              << "s)...\n";
    const Validation validation =
        validate(runner, points, validation_pattern, validation_prefix, validation_suffix,
                 options.ignore_case, steps, options.validation_hits,
                 options.validation_timeout);
    const FullSearch full_search = search_requested(
        runner, points, requested, effective.prefix, effective.suffix, options.ignore_case,
        steps, options.full_search_seconds);

    write_private_json(
        options.output,
        build_json(options, gpu, effective.prefix, effective.suffix, points_per_thread,
                   chains, steps, throughput, projection, validation_prefix,
                   validation_suffix, validation, full_search));
    points.wipe_private_keys();

    if (!options.loose_rules.empty()) {
      std::cout << "\nBase pattern      : ^T" << options.prefix << "..." << options.suffix
                << "$\n";
    }
    std::cout << "\nRequested pattern : ^T" << effective.prefix << "..." << effective.suffix
              << "$ (" << (options.ignore_case ? "case-insensitive" : "case-sensitive")
              << ")\n"
              << "Probability model : 1/25 second character; allowed-cases/58 later\n"
              << std::scientific << std::setprecision(12)
              << "Match chance      : " << projection.probability << " per candidate\n"
              << std::fixed << std::setprecision(3)
              << "Expected attempts : " << projection.attempts << " addresses\n"
              << "Measured GPU rate : " << throughput.rate / 1e6
              << " M full TRON addr/s\n"
              << "Projected mean    : " << projection.mean << " seconds\n"
              << "Mean hours        : " << projection.mean / 3600.0 << " hours\n"
              << "Mean days         : " << projection.mean / 86400.0 << " days\n"
              << "Projected median  : " << projection.median << " seconds\n"
              << "Projected p95     : " << projection.p95 << " seconds\n"
              << "Validation         : " << validation.verified << "/" << validation.hits
              << " GPU hits verified on CPU\n";
    if (full_search.attempted) {
      std::cout << "Full 4x6 trial     : "
                << (full_search.found ? "verified hit" : "no hit before timeout")
                << " after " << full_search.seconds << " seconds\n";
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
