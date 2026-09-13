#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace tron {

inline constexpr std::string_view kBase58 =
    "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

class PatternError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

using Token = std::uint64_t;

struct PatternParams {
  std::int32_t prefix_len{};
  std::int32_t suffix_len{};
  std::int32_t repeat_n{};
  std::int32_t reserved{};
  std::array<Token, 5> prefix_masks{};
  std::array<Token, 6> suffix_masks{};
};

static_assert(sizeof(PatternParams) == 104, "CUDA pattern ABI changed");

struct Probability {
  long double probability{};
  long double expected_attempts{};
};

std::vector<Token> parse_fixed_pattern(std::string_view pattern, bool ignore_case);
PatternParams compile_pattern(std::string_view prefix, std::string_view suffix,
                              bool ignore_case, bool require_4x6 = true);
std::string representative_literals(std::string_view pattern, bool ignore_case,
                                    std::size_t count, bool first_after_t = false);
bool matches(std::string_view address, const std::vector<Token>& prefix,
             const std::vector<Token>& suffix);
Probability match_probability(std::string_view prefix, std::string_view suffix,
                              bool ignore_case);
// Client-specified approximation: the character after T is uniform over the
// 25 possible symbols; every later position is uniform over Base58's 58
// symbols. Case-folded masks are counted instead of assuming every letter has
// two valid forms.
Probability match_probability_simplified(std::string_view prefix,
                                         std::string_view suffix,
                                         bool ignore_case);
int base58_index(char value);

}  // namespace tron
