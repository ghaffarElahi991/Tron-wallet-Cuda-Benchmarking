#include "tron/crypto.hpp"

#include "tron/pattern.hpp"

#include <openssl/bn.h>
#include <openssl/crypto.h>
#include <openssl/ec.h>
#include <openssl/obj_mac.h>
#include <openssl/rand.h>
#include <openssl/sha.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace tron {
namespace {

template <typename T, void (*FreeFunction)(T*)>
using OpenSslPtr = std::unique_ptr<T, decltype(FreeFunction)>;

using BnPtr = OpenSslPtr<BIGNUM, BN_free>;
using ClearBnPtr = OpenSslPtr<BIGNUM, BN_clear_free>;
using BnContextPtr = OpenSslPtr<BN_CTX, BN_CTX_free>;
using EcGroupPtr = OpenSslPtr<EC_GROUP, EC_GROUP_free>;
using EcPointPtr = OpenSslPtr<EC_POINT, EC_POINT_free>;

constexpr std::array<std::uint64_t, 24> kKeccakRoundConstants = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808AULL,
    0x8000000080008000ULL, 0x000000000000808BULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008AULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800AULL, 0x800000008000000AULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL};

std::uint64_t rotate_left(std::uint64_t value, unsigned amount) {
  return (value << amount) | (value >> (64U - amount));
}

void keccak_round(std::array<std::uint64_t, 25>& state, std::uint64_t constant) {
  const std::uint64_t bc0 = state[0] ^ state[5] ^ state[10] ^ state[15] ^ state[20];
  const std::uint64_t bc1 = state[1] ^ state[6] ^ state[11] ^ state[16] ^ state[21];
  const std::uint64_t bc2 = state[2] ^ state[7] ^ state[12] ^ state[17] ^ state[22];
  const std::uint64_t bc3 = state[3] ^ state[8] ^ state[13] ^ state[18] ^ state[23];
  const std::uint64_t bc4 = state[4] ^ state[9] ^ state[14] ^ state[19] ^ state[24];
  const std::uint64_t t0 = bc4 ^ rotate_left(bc1, 1);
  const std::uint64_t t1 = bc0 ^ rotate_left(bc2, 1);
  const std::uint64_t t2 = bc1 ^ rotate_left(bc3, 1);
  const std::uint64_t t3 = bc2 ^ rotate_left(bc4, 1);
  const std::uint64_t t4 = bc3 ^ rotate_left(bc0, 1);
  for (int row = 0; row < 25; row += 5) {
    state[static_cast<std::size_t>(row)] ^= t0;
    state[static_cast<std::size_t>(row + 1)] ^= t1;
    state[static_cast<std::size_t>(row + 2)] ^= t2;
    state[static_cast<std::size_t>(row + 3)] ^= t3;
    state[static_cast<std::size_t>(row + 4)] ^= t4;
  }

  std::uint64_t temporary = state[1];
  auto move = [&](int destination, unsigned rotation) {
    const std::uint64_t next = state[static_cast<std::size_t>(destination)];
    state[static_cast<std::size_t>(destination)] = rotate_left(temporary, rotation);
    temporary = next;
  };
  move(10, 1); move(7, 3); move(11, 6); move(17, 10);
  move(18, 15); move(3, 21); move(5, 28); move(16, 36);
  move(8, 45); move(21, 55); move(24, 2); move(4, 14);
  move(15, 27); move(23, 41); move(19, 56); move(13, 8);
  move(12, 25); move(2, 43); move(20, 62); move(14, 18);
  move(22, 39); move(9, 61); move(6, 20);
  state[1] = rotate_left(temporary, 44);

  for (int row = 0; row < 25; row += 5) {
    const auto offset = static_cast<std::size_t>(row);
    const std::uint64_t a0 = state[offset];
    const std::uint64_t a1 = state[offset + 1U];
    const std::uint64_t a2 = state[offset + 2U];
    const std::uint64_t a3 = state[offset + 3U];
    const std::uint64_t a4 = state[offset + 4U];
    state[offset] = a0 ^ ((~a1) & a2);
    state[offset + 1U] = a1 ^ ((~a2) & a3);
    state[offset + 2U] = a2 ^ ((~a3) & a4);
    state[offset + 3U] = a3 ^ ((~a4) & a0);
    state[offset + 4U] = a4 ^ ((~a0) & a1);
  }
  state[0] ^= constant;
}

std::array<unsigned char, 32> keccak256_64(const unsigned char* input) {
  std::array<std::uint64_t, 25> state{};
  for (std::size_t index = 0; index < 8U; ++index) {
    for (std::size_t byte = 0; byte < 8U; ++byte) {
      state[index] |= static_cast<std::uint64_t>(input[index * 8U + byte]) << (byte * 8U);
    }
  }
  state[8] ^= 1U;
  state[16] ^= 0x8000000000000000ULL;
  for (const auto constant : kKeccakRoundConstants) keccak_round(state, constant);

  std::array<unsigned char, 32> output{};
  for (std::size_t index = 0; index < 4U; ++index) {
    for (std::size_t byte = 0; byte < 8U; ++byte) {
      output[index * 8U + byte] =
          static_cast<unsigned char>(state[index] >> (byte * 8U));
    }
  }
  return output;
}

std::string base58_encode_25(const std::array<unsigned char, 25>& input) {
  auto buffer = input;
  std::string reversed;
  reversed.reserve(40);
  std::size_t start = 0;
  while (start < buffer.size() && buffer[start] == 0) {
    reversed.push_back(kBase58[0]);
    ++start;
  }
  std::size_t top = start;
  while (top < buffer.size()) {
    unsigned remainder = 0;
    for (std::size_t index = top; index < buffer.size(); ++index) {
      const unsigned value = (remainder << 8U) | buffer[index];
      buffer[index] = static_cast<unsigned char>(value / 58U);
      remainder = value % 58U;
    }
    reversed.push_back(kBase58[remainder]);
    while (top < buffer.size() && buffer[top] == 0) ++top;
  }
  std::reverse(reversed.begin(), reversed.end());
  return reversed;
}

std::uint64_t big_endian_limb(const unsigned char* bytes, std::size_t limb) {
  const std::size_t offset = (3U - limb) * 8U;
  std::uint64_t result = 0;
  for (std::size_t index = 0; index < 8U; ++index) {
    result = (result << 8U) | bytes[offset + index];
  }
  return result;
}

void check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

}  // namespace

StartPoints::~StartPoints() { wipe_private_keys(); }

void StartPoints::wipe_private_keys() {
  for (auto& key : private_keys) OPENSSL_cleanse(key.data(), key.size());
  private_keys.clear();
  private_keys.shrink_to_fit();
}

StartPoints create_start_points(std::size_t count, bool random_keys, unsigned worker_count) {
  StartPoints result;
  result.x.resize(count * 4U);
  result.y.resize(count * 4U);
  result.private_keys.resize(count);
  if (count == 0U) return result;

  if (worker_count == 0U) {
    worker_count = std::min(8U, std::max(1U, std::thread::hardware_concurrency()));
  }
  worker_count = std::min<unsigned>(worker_count, static_cast<unsigned>(count));
  std::atomic<std::size_t> next{0};
  std::exception_ptr failure;
  std::mutex failure_mutex;
  std::vector<std::thread> workers;
  workers.reserve(worker_count);

  for (unsigned worker = 0; worker < worker_count; ++worker) {
    workers.emplace_back([&] {
      try {
        EcGroupPtr group(EC_GROUP_new_by_curve_name(NID_secp256k1), EC_GROUP_free);
        BnContextPtr context(BN_CTX_new(), BN_CTX_free);
        EcPointPtr point(group ? EC_POINT_new(group.get()) : nullptr, EC_POINT_free);
        ClearBnPtr private_bn(BN_new(), BN_clear_free);
        BnPtr order(BN_new(), BN_free);
        BnPtr x(BN_new(), BN_free);
        BnPtr y(BN_new(), BN_free);
        check(group && context && point && private_bn && order && x && y,
              "OpenSSL secp256k1 allocation failed");
        check(EC_GROUP_get_order(group.get(), order.get(), context.get()) == 1,
              "cannot read secp256k1 order");

        std::array<unsigned char, 32> x_bytes{};
        std::array<unsigned char, 32> y_bytes{};
        while (true) {
          const std::size_t index = next.fetch_add(1);
          if (index >= count) break;
          if (random_keys) {
            do {
              check(RAND_bytes(result.private_keys[index].data(), 32) == 1,
                    "OpenSSL random generator failed");
              check(BN_bin2bn(result.private_keys[index].data(), 32, private_bn.get()) != nullptr,
                    "private-key conversion failed");
            } while (BN_is_zero(private_bn.get()) != 0 ||
                     BN_cmp(private_bn.get(), order.get()) >= 0);
          } else {
            check(BN_set_word(private_bn.get(), static_cast<BN_ULONG>(index + 1U)) == 1,
                  "deterministic private-key setup failed");
            check(BN_bn2binpad(private_bn.get(), result.private_keys[index].data(), 32) == 32,
                  "private-key serialization failed");
          }
          check(EC_POINT_mul(group.get(), point.get(), private_bn.get(), nullptr, nullptr,
                             context.get()) == 1,
                "secp256k1 point multiplication failed");
          check(EC_POINT_get_affine_coordinates(group.get(), point.get(), x.get(), y.get(),
                                                context.get()) == 1,
                "secp256k1 affine conversion failed");
          check(BN_bn2binpad(x.get(), x_bytes.data(), 32) == 32 &&
                    BN_bn2binpad(y.get(), y_bytes.data(), 32) == 32,
                "public point serialization failed");
          for (std::size_t limb = 0; limb < 4U; ++limb) {
            result.x[index * 4U + limb] = big_endian_limb(x_bytes.data(), limb);
            result.y[index * 4U + limb] = big_endian_limb(y_bytes.data(), limb);
          }
        }
      } catch (...) {
        std::lock_guard lock(failure_mutex);
        if (!failure) failure = std::current_exception();
      }
    });
  }
  for (auto& worker : workers) worker.join();
  if (failure) {
    result.wipe_private_keys();
    std::rethrow_exception(failure);
  }
  return result;
}

std::string address_for_offset(const PrivateKey& base_key, std::uint64_t offset) {
  EcGroupPtr group(EC_GROUP_new_by_curve_name(NID_secp256k1), EC_GROUP_free);
  BnContextPtr context(BN_CTX_new(), BN_CTX_free);
  EcPointPtr point(group ? EC_POINT_new(group.get()) : nullptr, EC_POINT_free);
  ClearBnPtr private_bn(BN_bin2bn(base_key.data(), 32, nullptr), BN_clear_free);
  BnPtr offset_bn(BN_new(), BN_free);
  BnPtr order(BN_new(), BN_free);
  check(group && context && point && private_bn && offset_bn && order,
        "OpenSSL verification allocation failed");
  check(BN_set_word(offset_bn.get(), static_cast<BN_ULONG>(offset)) == 1 &&
            EC_GROUP_get_order(group.get(), order.get(), context.get()) == 1 &&
            BN_mod_add(private_bn.get(), private_bn.get(), offset_bn.get(), order.get(),
                       context.get()) == 1,
        "private-key offset calculation failed");
  check(BN_is_zero(private_bn.get()) == 0, "derived private key is zero");
  check(EC_POINT_mul(group.get(), point.get(), private_bn.get(), nullptr, nullptr,
                     context.get()) == 1,
        "CPU secp256k1 verification failed");

  std::array<unsigned char, 65> public_key{};
  check(EC_POINT_point2oct(group.get(), point.get(), POINT_CONVERSION_UNCOMPRESSED,
                           public_key.data(), public_key.size(), context.get()) ==
            public_key.size(),
        "public-key serialization failed");
  const auto digest = keccak256_64(public_key.data() + 1U);
  std::array<unsigned char, 21> payload{};
  payload[0] = 0x41;
  std::copy(digest.end() - 20, digest.end(), payload.begin() + 1);

  std::array<unsigned char, SHA256_DIGEST_LENGTH> first_hash{};
  std::array<unsigned char, SHA256_DIGEST_LENGTH> second_hash{};
  SHA256(payload.data(), payload.size(), first_hash.data());
  SHA256(first_hash.data(), first_hash.size(), second_hash.data());

  std::array<unsigned char, 25> raw{};
  std::copy(payload.begin(), payload.end(), raw.begin());
  std::copy_n(second_hash.begin(), 4, raw.begin() + 21);
  OPENSSL_cleanse(public_key.data(), public_key.size());
  return base58_encode_25(raw);
}

}  // namespace tron
