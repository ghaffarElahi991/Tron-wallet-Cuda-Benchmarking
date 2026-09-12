#include "tron/crypto.hpp"

#include <iostream>
#include <stdexcept>

int main() {
  tron::PrivateKey key{};
  key.back() = 1;
  if (tron::address_for_offset(key, 0) != "TMVQGm1qAQYVdetCeGRRkTWYYrLXuHK2HC") {
    throw std::runtime_error("private key 1 produced the wrong TRON address");
  }
  if (tron::address_for_offset(key, 1) != "TDvSsdrNM5eeXNL3czpa6AxLDHZA9nwe9K") {
    throw std::runtime_error("private key 1 + offset 1 produced the wrong TRON address");
  }
  std::cout << "crypto tests passed\n";
}
