#include "tron/pattern.hpp"

#include <boost/multiprecision/cpp_int.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <functional>
#include <limits>

namespace tron {
namespace {

using boost::multiprecision::cpp_int;
constexpr Token kAllBase58 = (Token{1} << 58U) - 1U;

Token literal_mask(char value, bool ignore_case) {
  Token result = 0;
  const auto add = [&result](char candidate) {
    const int index = base58_index(candidate);
    if (index >= 0) result |= Token{1} << static_cast<unsigned>(index);
  };
  add(value);
  if (ignore_case && std::isalpha(static_cast<unsigned char>(value)) != 0) {
    add(static_cast<char>(std::tolower(static_cast<unsigned char>(value))));
    add(static_cast<char>(std::toupper(static_cast<unsigned char>(value))));
  }
  return result;
}

std::array<int, 34> base58_digits(cpp_int number) {
  std::array<int, 34> result{};
  for (auto it = result.rbegin(); it != result.rend(); ++it) {
    const cpp_int remainder = number % 58;
    *it = remainder.convert_to<int>();
    number /= 58;
  }
  if (number != 0) throw PatternError("Base58 interval exceeds 34 digits");
  return result;
}

cpp_int count_at_most(const cpp_int& bound, const std::array<Token, 34>& allowed) {
  if (bound < 0) return 0;
  const auto digits = base58_digits(bound);
  std::array<std::array<cpp_int, 2>, 35> memo{};
  std::array<std::array<bool, 2>, 35> ready{};

  std::function<cpp_int(std::size_t, bool)> visit =
      [&](std::size_t position, bool tight) -> cpp_int {
    if (position == allowed.size()) return 1;
    const std::size_t tight_index = tight ? 1U : 0U;
    if (ready[position][tight_index]) return memo[position][tight_index];
    const int maximum = tight ? digits[position] : 57;
    cpp_int total = 0;
    for (int digit = 0; digit <= maximum; ++digit) {
      if ((allowed[position] & (Token{1} << static_cast<unsigned>(digit))) != 0) {
        total += visit(position + 1U, tight && digit == maximum);
      }
    }
    ready[position][tight_index] = true;
    memo[position][tight_index] = total;
    return total;
  };
  return visit(0, true);
}

}  // namespace

int base58_index(char value) {
  const auto found = kBase58.find(value);
  return found == std::string_view::npos ? -1 : static_cast<int>(found);
}

std::vector<Token> parse_fixed_pattern(std::string_view pattern, bool ignore_case) {
  std::vector<Token> tokens;
  for (std::size_t index = 0; index < pattern.size();) {
    Token accepted = 0;
    const char value = pattern[index];
    if (value == '?') {
      accepted = kAllBase58;
      ++index;
    } else if (value == '[') {
      const auto close = pattern.find(']', index + 1U);
      if (close == std::string_view::npos) throw PatternError("unclosed '[' character class");
      if (close == index + 1U) throw PatternError("empty '[]' character class");
      for (std::size_t item = index + 1U; item < close; ++item) {
        if (pattern[item] == '[' || pattern[item] == '?' || pattern[item] == ']') {
          throw PatternError("character classes contain Base58 literals only");
        }
        accepted |= literal_mask(pattern[item], ignore_case);
      }
      index = close + 1U;
    } else {
      constexpr std::string_view unsupported = "]*+{}()|.^$\\";
      if (unsupported.find(value) != std::string_view::npos) {
        throw PatternError(std::string("unsupported regex operator '") + value +
                           "'; use literals, '?', or '[abc]'");
      }
      accepted = literal_mask(value, ignore_case);
      ++index;
    }
    if (accepted == 0) {
      throw PatternError(std::string("character '") + value + "' is not in Base58");
    }
    tokens.push_back(accepted);
  }
  return tokens;
}

PatternParams compile_pattern(std::string_view prefix, std::string_view suffix,
                              bool ignore_case, bool require_4x6) {
  const auto prefix_tokens = parse_fixed_pattern(prefix, ignore_case);
  const auto suffix_tokens = parse_fixed_pattern(suffix, ignore_case);
  if (prefix_tokens.size() > 4U || suffix_tokens.size() > 6U ||
      (require_4x6 && (prefix_tokens.size() != 4U || suffix_tokens.size() != 6U))) {
    throw PatternError(require_4x6
                           ? "this milestone requires exactly four prefix tokens after T and six suffix tokens"
                           : "validation pattern exceeds the 4x6 matcher capacity");
  }
  PatternParams result;
  result.prefix_len = static_cast<std::int32_t>(prefix_tokens.size());
  result.suffix_len = static_cast<std::int32_t>(suffix_tokens.size());
  std::copy(prefix_tokens.begin(), prefix_tokens.end(), result.prefix_masks.begin());
  std::copy(suffix_tokens.begin(), suffix_tokens.end(), result.suffix_masks.begin());
  return result;
}

std::string representative_literals(std::string_view pattern, bool ignore_case,
                                    std::size_t count, bool first_after_t) {
  const auto tokens = parse_fixed_pattern(pattern, ignore_case);
  count = std::min(count, tokens.size());
  std::string result;
  result.reserve(count);
  constexpr std::string_view possible_second = "9ABCDEFGHJKLMNPQRSTUVWXYZ";
  for (std::size_t index = 0; index < count; ++index) {
    bool found = false;
    for (std::size_t digit = 0; digit < kBase58.size(); ++digit) {
      const char candidate = kBase58[digit];
      if (first_after_t && index == 0U &&
          possible_second.find(candidate) == std::string_view::npos) {
        continue;
      }
      if ((tokens[index] & (Token{1} << digit)) != 0) {
        result.push_back(candidate);
        found = true;
        break;
      }
    }
    if (!found) throw PatternError("first prefix token cannot occur immediately after T");
  }
  return result;
}

bool matches(std::string_view address, const std::vector<Token>& prefix,
             const std::vector<Token>& suffix) {
  if (address.size() != 34U || address.front() != 'T' ||
      1U + prefix.size() + suffix.size() > address.size()) {
    return false;
  }
  for (std::size_t index = 0; index < prefix.size(); ++index) {
    const int digit = base58_index(address[index + 1U]);
    if (digit < 0 || (prefix[index] & (Token{1} << static_cast<unsigned>(digit))) == 0) {
      return false;
    }
  }
  const std::size_t start = address.size() - suffix.size();
  for (std::size_t index = 0; index < suffix.size(); ++index) {
    const int digit = base58_index(address[start + index]);
    if (digit < 0 || (suffix[index] & (Token{1} << static_cast<unsigned>(digit))) == 0) {
      return false;
    }
  }
  return true;
}

Probability match_probability(std::string_view prefix, std::string_view suffix,
                              bool ignore_case) {
  const auto prefix_tokens = parse_fixed_pattern(prefix, ignore_case);
  const auto suffix_tokens = parse_fixed_pattern(suffix, ignore_case);
  if (1U + prefix_tokens.size() + suffix_tokens.size() > 34U) {
    throw PatternError("fixed T + prefix + suffix exceeds the 34-character address");
  }

  std::array<Token, 34> allowed{};
  allowed.fill(kAllBase58);
  allowed[0] = literal_mask('T', false);
  for (std::size_t index = 0; index < prefix_tokens.size(); ++index) {
    allowed[index + 1U] &= prefix_tokens[index];
  }
  const std::size_t suffix_start = 34U - suffix_tokens.size();
  for (std::size_t index = 0; index < suffix_tokens.size(); ++index) {
    allowed[suffix_start + index] &= suffix_tokens[index];
  }
  if (std::any_of(allowed.begin(), allowed.end(), [](Token mask) { return mask == 0; })) {
    throw PatternError("pattern cannot occur in a TRON Base58Check address");
  }

  const cpp_int low = cpp_int(0x41) << 192;
  const cpp_int high = (cpp_int(0x42) << 192) - 1;
  const cpp_int total = cpp_int(1) << 192;
  const cpp_int favorable = count_at_most(high, allowed) - count_at_most(low - 1, allowed);
  if (favorable == 0) throw PatternError("pattern cannot occur in a TRON Base58Check address");
  const long double favorable_ld = favorable.convert_to<long double>();
  const long double total_ld = total.convert_to<long double>();
  return Probability{favorable_ld / total_ld, total_ld / favorable_ld};
}

Probability match_probability_simplified(std::string_view prefix,
                                         std::string_view suffix,
                                         bool ignore_case) {
  const auto prefix_tokens = parse_fixed_pattern(prefix, ignore_case);
  const auto suffix_tokens = parse_fixed_pattern(suffix, ignore_case);
  if (prefix_tokens.empty()) {
    throw PatternError("the simplified model requires a pattern for the character after T");
  }
  if (1U + prefix_tokens.size() + suffix_tokens.size() > 34U) {
    throw PatternError("fixed T + prefix + suffix exceeds the 34-character address");
  }

  Token second_character_mask = 0;
  constexpr std::string_view possible_second = "9ABCDEFGHJKLMNPQRSTUVWXYZ";
  for (const char value : possible_second) {
    second_character_mask |= Token{1} << static_cast<unsigned>(base58_index(value));
  }

  cpp_int favorable = std::popcount(prefix_tokens.front() & second_character_mask);
  cpp_int total = 25;
  if (favorable == 0) {
    throw PatternError("first prefix token cannot occur immediately after T");
  }

  const auto include_base58_position = [&](Token accepted) {
    const unsigned count = std::popcount(accepted & kAllBase58);
    if (count == 0U) throw PatternError("pattern position cannot match Base58");
    favorable *= count;
    total *= 58;
  };
  for (std::size_t index = 1; index < prefix_tokens.size(); ++index) {
    include_base58_position(prefix_tokens[index]);
  }
  for (const Token accepted : suffix_tokens) include_base58_position(accepted);

  const long double favorable_ld = favorable.convert_to<long double>();
  const long double total_ld = total.convert_to<long double>();
  return Probability{favorable_ld / total_ld, total_ld / favorable_ld};
}

}  // namespace tron
