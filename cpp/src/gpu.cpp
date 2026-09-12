#include "tron/gpu.hpp"

#include "tron/crypto.hpp"

#include <nvrtc.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace tron {
namespace {

constexpr unsigned kThreadsPerBlock = 64;
constexpr std::uint32_t kMaxMatches = 64;

[[noreturn]] void throw_cuda(CUresult result, const char* operation) {
  const char* name = nullptr;
  const char* description = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &description);
  throw std::runtime_error(std::string(operation) + " failed: " +
                           (name ? name : "unknown CUDA error") + " (" +
                           (description ? description : "no description") + ")");
}

void cuda_check(CUresult result, const char* operation) {
  if (result != CUDA_SUCCESS) throw_cuda(result, operation);
}

void nvrtc_check(nvrtcResult result, const char* operation) {
  if (result != NVRTC_SUCCESS) {
    throw std::runtime_error(std::string(operation) + " failed: " +
                             nvrtcGetErrorString(result));
  }
}

std::string read_file(const std::string& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("cannot open CUDA kernel: " + path);
  std::ostringstream contents;
  contents << stream.rdbuf();
  if (!stream.good() && !stream.eof()) throw std::runtime_error("cannot read CUDA kernel: " + path);
  return contents.str();
}

std::string compile_ptx(const std::string& source, const std::string& source_name,
                        int major, int minor, int points_per_thread) {
  nvrtcProgram program{};
  nvrtc_check(nvrtcCreateProgram(&program, source.c_str(), source_name.c_str(), 0, nullptr,
                                 nullptr),
              "nvrtcCreateProgram");
  const std::string architecture = "--gpu-architecture=compute_" + std::to_string(major) +
                                   std::to_string(minor);
  const std::string points = "-DPOINTS_PER_THREAD=" + std::to_string(points_per_thread);
  const std::array<const char*, 4> options = {
      "--std=c++14", "--use_fast_math", architecture.c_str(), points.c_str()};
  const nvrtcResult compile_result =
      nvrtcCompileProgram(program, static_cast<int>(options.size()), options.data());
  std::size_t log_size = 0;
  nvrtcGetProgramLogSize(program, &log_size);
  std::string log(log_size, '\0');
  if (log_size > 1U) nvrtcGetProgramLog(program, log.data());
  if (compile_result != NVRTC_SUCCESS) {
    const std::string message = "NVRTC compilation failed: " +
                                std::string(nvrtcGetErrorString(compile_result)) + "\n" + log;
    nvrtcDestroyProgram(&program);
    throw std::runtime_error(message);
  }
  std::size_t ptx_size = 0;
  nvrtc_check(nvrtcGetPTXSize(program, &ptx_size), "nvrtcGetPTXSize");
  std::string ptx(ptx_size, '\0');
  nvrtc_check(nvrtcGetPTX(program, ptx.data()), "nvrtcGetPTX");
  nvrtcDestroyProgram(&program);
  return ptx;
}

}  // namespace

CudaContext::CudaContext() {
  cuda_check(cuInit(0), "cuInit");
  int count = 0;
  cuda_check(cuDeviceGetCount(&count), "cuDeviceGetCount");
  if (count < 1) throw std::runtime_error("no CUDA GPU detected");
  cuda_check(cuDeviceGet(&device_, 0), "cuDeviceGet");
  cuda_check(cuDevicePrimaryCtxRetain(&context_, device_), "cuDevicePrimaryCtxRetain");
  try {
    cuda_check(cuCtxSetCurrent(context_), "cuCtxSetCurrent");
    std::array<char, 256> name{};
    cuda_check(cuDeviceGetName(name.data(), static_cast<int>(name.size()), device_),
               "cuDeviceGetName");
    info_.name = name.data();
    cuda_check(cuDeviceGetAttribute(&info_.major,
                                    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device_),
               "cuDeviceGetAttribute(compute major)");
    cuda_check(cuDeviceGetAttribute(&info_.minor,
                                    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device_),
               "cuDeviceGetAttribute(compute minor)");
    cuda_check(cuDeviceGetAttribute(&info_.sm_count,
                                    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device_),
               "cuDeviceGetAttribute");
    cuda_check(cuDriverGetVersion(&info_.cuda_driver_api_version), "cuDriverGetVersion");
  } catch (...) {
    cuDevicePrimaryCtxRelease(device_);
    context_ = nullptr;
    throw;
  }
}

CudaContext::~CudaContext() {
  if (context_ != nullptr) {
    cuCtxSetCurrent(nullptr);
    cuDevicePrimaryCtxRelease(device_);
  }
}

GpuRunner::GpuRunner(const CudaContext& context, const std::string& kernel_path,
                     int points_per_thread, int blocks,
                     const std::vector<std::uint64_t>& x,
                     const std::vector<std::uint64_t>& y)
    : points_per_thread_(points_per_thread), blocks_(blocks) {
  if (points_per_thread != 8 && points_per_thread != 16) {
    throw std::invalid_argument("points per thread must be 8 or 16");
  }
  chains_ = static_cast<std::size_t>(blocks_) * kThreadsPerBlock *
            static_cast<std::size_t>(points_per_thread_);
  if (x.size() != chains_ * 4U || y.size() != chains_ * 4U) {
    throw std::invalid_argument("start-point array size does not match CUDA launch geometry");
  }

  const auto cleanup = [&] {
    if (count_device_ != 0) cuMemFree(count_device_);
    if (matches_device_ != 0) cuMemFree(matches_device_);
    if (pattern_device_ != 0) cuMemFree(pattern_device_);
    if (y_device_ != 0) cuMemFree(y_device_);
    if (x_device_ != 0) cuMemFree(x_device_);
    if (module_ != nullptr) cuModuleUnload(module_);
    count_device_ = matches_device_ = pattern_device_ = y_device_ = x_device_ = 0;
    module_ = nullptr;
  };

  try {
    const std::string source = read_file(kernel_path);
    const std::string ptx = compile_ptx(source, kernel_path, context.info().major,
                                        context.info().minor, points_per_thread_);
    cuda_check(cuModuleLoadData(&module_, ptx.data()), "cuModuleLoadData");
    cuda_check(cuModuleGetFunction(&kernel_, module_, "vanity_kernel"),
               "cuModuleGetFunction");
    const std::size_t coordinate_bytes = x.size() * sizeof(std::uint64_t);
    cuda_check(cuMemAlloc(&x_device_, coordinate_bytes), "cuMemAlloc(x)");
    cuda_check(cuMemAlloc(&y_device_, coordinate_bytes), "cuMemAlloc(y)");
    cuda_check(cuMemAlloc(&pattern_device_, sizeof(PatternParams)), "cuMemAlloc(pattern)");
    cuda_check(cuMemAlloc(&matches_device_, sizeof(MatchRecord) * kMaxMatches),
               "cuMemAlloc(matches)");
    cuda_check(cuMemAlloc(&count_device_, sizeof(std::uint32_t)), "cuMemAlloc(count)");
    cuda_check(cuMemcpyHtoD(x_device_, x.data(), coordinate_bytes), "cuMemcpyHtoD(x)");
    cuda_check(cuMemcpyHtoD(y_device_, y.data(), coordinate_bytes), "cuMemcpyHtoD(y)");
  } catch (...) {
    cleanup();
    throw;
  }
}

GpuRunner::~GpuRunner() {
  if (count_device_ != 0) cuMemFree(count_device_);
  if (matches_device_ != 0) cuMemFree(matches_device_);
  if (pattern_device_ != 0) cuMemFree(pattern_device_);
  if (y_device_ != 0) cuMemFree(y_device_);
  if (x_device_ != 0) cuMemFree(x_device_);
  if (module_ != nullptr) cuModuleUnload(module_);
}

void GpuRunner::set_pattern(const PatternParams& pattern) {
  cuda_check(cuMemcpyHtoD(pattern_device_, &pattern, sizeof(pattern)),
             "cuMemcpyHtoD(pattern)");
}

LaunchResult GpuRunner::launch(int steps) {
  if (steps <= 0) throw std::invalid_argument("steps must be positive");
  const std::uint32_t zero = 0;
  cuda_check(cuMemcpyHtoD(count_device_, &zero, sizeof(zero)), "cuMemcpyHtoD(count)");
  CUdeviceptr x_argument = x_device_;
  CUdeviceptr y_argument = y_device_;
  CUdeviceptr pattern_argument = pattern_device_;
  CUdeviceptr matches_argument = matches_device_;
  CUdeviceptr count_argument = count_device_;
  std::uint64_t launch_offset = step_offset_;
  std::uint32_t maximum = kMaxMatches;
  void* arguments[] = {&x_argument,       &y_argument,       &steps,
                       &launch_offset, &pattern_argument,
                       &matches_argument, &count_argument, &maximum};
  const auto started = std::chrono::steady_clock::now();
  cuda_check(cuLaunchKernel(kernel_, static_cast<unsigned>(blocks_), 1, 1,
                            kThreadsPerBlock, 1, 1, 0, nullptr, arguments, nullptr),
             "cuLaunchKernel");
  cuda_check(cuCtxSynchronize(), "cuCtxSynchronize");
  const auto finished = std::chrono::steady_clock::now();
  step_offset_ += static_cast<std::uint64_t>(steps);

  LaunchResult result;
  result.elapsed_seconds = std::chrono::duration<double>(finished - started).count();
  cuda_check(cuMemcpyDtoH(&result.count, count_device_, sizeof(result.count)),
             "cuMemcpyDtoH(count)");
  const std::uint32_t stored = std::min(result.count, kMaxMatches);
  result.records.resize(stored);
  if (stored > 0) {
    cuda_check(cuMemcpyDtoH(result.records.data(), matches_device_,
                            sizeof(MatchRecord) * stored),
               "cuMemcpyDtoH(matches)");
  }
  return result;
}

double tune_points_per_thread(const CudaContext& context,
                              const std::string& kernel_path,
                              const PatternParams& pattern,
                              int points_per_thread) {
  constexpr int tuning_blocks = 4;
  const std::size_t chains = static_cast<std::size_t>(tuning_blocks) * kThreadsPerBlock *
                             static_cast<std::size_t>(points_per_thread);
  auto points = create_start_points(chains, false, 4);
  GpuRunner runner(context, kernel_path, points_per_thread, tuning_blocks, points.x, points.y);
  runner.set_pattern(pattern);
  runner.launch(32);
  const auto measured = runner.launch(32);
  points.wipe_private_keys();
  return static_cast<double>(chains * 32U) / measured.elapsed_seconds;
}

}  // namespace tron
