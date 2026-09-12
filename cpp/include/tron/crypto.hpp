#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tron {

using PrivateKey = std::array<unsigned char, 32>;

struct StartPoints {
  std::vector<std::uint64_t> x;
  std::vector<std::uint64_t> y;
  std::vector<PrivateKey> private_keys;

  StartPoints() = default;
  ~StartPoints();
  StartPoints(const StartPoints&) = delete;
  StartPoints& operator=(const StartPoints&) = delete;
  StartPoints(StartPoints&&) noexcept = default;
  StartPoints& operator=(StartPoints&&) noexcept = default;

  void wipe_private_keys();
};

StartPoints create_start_points(std::size_t count, bool random_keys,
                                unsigned worker_count = 0);
std::string address_for_offset(const PrivateKey& base_key, std::uint64_t offset);

}  // namespace tron
