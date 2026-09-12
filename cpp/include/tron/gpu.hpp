#pragma once

#include "tron/pattern.hpp"

#include <cuda.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tron {

struct MatchRecord {
  std::uint32_t thread_id{};
  std::uint32_t pad{};
  std::uint64_t step{};
  char address[34]{};
  char pad2[6]{};
};

static_assert(sizeof(MatchRecord) == 56, "CUDA match ABI changed");

struct GpuInfo {
  std::string name;
  int major{};
  int minor{};
  int sm_count{};
  int cuda_driver_api_version{};
};

struct LaunchResult {
  std::uint32_t count{};
  std::vector<MatchRecord> records;
  double elapsed_seconds{};
};

class CudaContext {
 public:
  CudaContext();
  ~CudaContext();
  CudaContext(const CudaContext&) = delete;
  CudaContext& operator=(const CudaContext&) = delete;

  const GpuInfo& info() const { return info_; }
  CUdevice device() const { return device_; }

 private:
  CUdevice device_{};
  CUcontext context_{};
  GpuInfo info_;
};

class GpuRunner {
 public:
  GpuRunner(const CudaContext& context, const std::string& kernel_path,
            int points_per_thread, int blocks, const std::vector<std::uint64_t>& x,
            const std::vector<std::uint64_t>& y);
  ~GpuRunner();
  GpuRunner(const GpuRunner&) = delete;
  GpuRunner& operator=(const GpuRunner&) = delete;

  void set_pattern(const PatternParams& pattern);
  LaunchResult launch(int steps);
  std::size_t chains() const { return chains_; }
  int points_per_thread() const { return points_per_thread_; }
  int blocks() const { return blocks_; }
  std::uint64_t step_offset() const { return step_offset_; }

 private:
  CUmodule module_{};
  CUfunction kernel_{};
  CUdeviceptr x_device_{};
  CUdeviceptr y_device_{};
  CUdeviceptr pattern_device_{};
  CUdeviceptr matches_device_{};
  CUdeviceptr count_device_{};
  int points_per_thread_{};
  int blocks_{};
  std::size_t chains_{};
  std::uint64_t step_offset_{};
};

double tune_points_per_thread(const CudaContext& context,
                              const std::string& kernel_path,
                              const PatternParams& pattern,
                              int points_per_thread);

}  // namespace tron
