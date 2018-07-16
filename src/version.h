#pragma once

#include <string>
#include "version.inc"

std::uint32_t GetVersionInt(int major = LC0_VERSION_MAJOR,
                            int minor = LC0_VERSION_MINOR,
                            int patch = LC0_VERSION_PATCH);

std::string GetVersionStr(int major = LC0_VERSION_MAJOR,
                          int minor = LC0_VERSION_MINOR,
                          int patch = LC0_VERSION_PATCH);
