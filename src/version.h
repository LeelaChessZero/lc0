#pragma once

#define LC0_VERSION_MAJOR 0
#define LC0_VERSION_MINOR 15
#define LC0_VERSION_PATCH 1

#include <string>

std::uint32_t GetVersionInt(int major = LC0_VERSION_MAJOR,
                            int minor = LC0_VERSION_MINOR,
                            int patch = LC0_VERSION_PATCH);

std::string GetVersionStr(int major = LC0_VERSION_MAJOR,
                          int minor = LC0_VERSION_MINOR,
                          int patch = LC0_VERSION_PATCH);
