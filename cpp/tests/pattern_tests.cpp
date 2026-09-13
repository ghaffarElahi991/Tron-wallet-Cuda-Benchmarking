#include "tron/pattern.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
void expect(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
}  // namespace

int main() {
  using tron::base58_index;
  const auto strict = tron::compile_pattern("Abcd", "WxyzAB", true);
  expect(strict.prefix_len == 4 && strict.suffix_len == 6, "4x6 lengths");
  expect((strict.prefix_masks[0] & (std::uint64_t{1} << base58_index('A'))) != 0,
         "uppercase folding");
  expect((strict.prefix_masks[0] & (std::uint64_t{1} << base58_index('a'))) != 0,
         "lowercase folding");

  const auto loose = tron::compile_pattern("Ab??", "Wxy?A?", true);
  expect(loose.prefix_masks[2] == ((std::uint64_t{1} << 58U) - 1U), "prefix wildcard");
  expect(loose.suffix_masks[3] == ((std::uint64_t{1} << 58U) - 1U), "suffix wildcard 1");
  expect(loose.suffix_masks[5] == ((std::uint64_t{1} << 58U) - 1U), "suffix wildcard 2");

  const auto three_by_four = tron::compile_pattern("New", "adre", true, false);
  expect(three_by_four.prefix_len == 3 && three_by_four.suffix_len == 4,
         "3x4 requested pattern");
  const auto four_by_three = tron::compile_pattern("Neww", "adr", true, false);
  expect(four_by_three.prefix_len == 4 && four_by_three.suffix_len == 3,
         "4x3 requested pattern");
  const auto two_by_five = tron::compile_pattern("Ne", "adreS", true, false);
  expect(two_by_five.prefix_len == 2 && two_by_five.suffix_len == 5,
         "2x5 requested pattern");
  const auto five_by_two = tron::compile_pattern("NewWa", "ad", true, false);
  expect(five_by_two.prefix_len == 5 && five_by_two.suffix_len == 2,
         "5x2 requested pattern");

  const auto parsed_prefix = tron::parse_fixed_pattern("Ab??", true);
  const auto parsed_suffix = tron::parse_fixed_pattern("Wxy?A?", true);
  expect(tron::matches("TAb1111111111111111111111111Wxy2A3", parsed_prefix, parsed_suffix),
         "loose positive match");
  expect(!tron::matches("TAc1111111111111111111111111Wxy2A3", parsed_prefix, parsed_suffix),
         "loose negative match");

  const auto probability = tron::match_probability("Ab??", "Wxy?A?", true);
  expect(probability.probability > 0.0L, "nonzero probability");
  expect(probability.expected_attempts > 1.0L, "nontrivial expected attempts");
  const long double expected = 478688906.5402532L;
  expect(std::abs(probability.expected_attempts - expected) / expected < 1e-12L,
         "exact interval probability differs from Python reference");

  const auto target = tron::match_probability_simplified("NewW", "adreSS", true);
  const long double target_attempts = 362678649396725.0L;
  expect(std::abs(target.expected_attempts - target_attempts) < 0.5L,
         "client-specified TNewW...adreSS expected attempts");
  expect(std::abs(target.probability - 1.0L / target_attempts) < 1e-30L,
         "client-specified TNewW...adreSS probability");

  const auto one_case = tron::match_probability_simplified("NL??", "??????", true);
  expect(std::abs(one_case.expected_attempts - 1450.0L) < 1e-12L,
         "letters with one valid Base58 case must not receive a 1/29 factor");

  bool rejected_lowercase_second = false;
  try {
    static_cast<void>(tron::match_probability_simplified("n???", "??????", false));
  } catch (const tron::PatternError&) {
    rejected_lowercase_second = true;
  }
  expect(rejected_lowercase_second, "lowercase n cannot immediately follow T");
  std::cout << "pattern tests passed\n";
}
