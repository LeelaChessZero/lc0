#include "version.h"

std::uint32_t GetVersionInt(int major, int minor, int patch, const std::string& postfix) {
  std::uint32_t version_int = major * 1000000 + minor * 1000 + patch;
  if (postfix.empty()) return version_int;
  return version_int + 1;
}

std::string GetVersionStr(int major, int minor, int patch, const std::string& postfix) {
  auto v = std::to_string(major) + "." + std::to_string(minor) + "." +
           std::to_string(patch);
  if (postfix.empty()) return v;
  return v + "-" + postfix;
}
