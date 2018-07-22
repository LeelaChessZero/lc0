#include "version.h"
#include <cassert>

std::uint32_t GetVersionInt(int major, int minor, int patch) {
  return major * 1000000 + minor * 1000 + patch;
}

std::string GetVersionStr(int major, int minor, int patch,
                          const std::string& postfix) {
  auto v = std::to_string(major) + "." + std::to_string(minor);
  if (postfix.empty()) {
    return v + "." + std::to_string(patch);
  } else {
    assert(patch == 0);
    return v + "-" + postfix;
  }
}
