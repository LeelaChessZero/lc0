#pragma once

// Versioning is performed according to the standard at <https://semver.org/>
// Our postfix consists of the 8 character git-sha1 id iff the commit is not
// tagged. Version bumping should be performed using scripts/bumpversion.py. An
// example versioning flow looks like this:
//
//   0.13.0 (released build)
//   0.13.1-sha1 (development of next release)
//   0.13.1-sha2
//   0.13.1 (released build)
//   0.13.2-sha3 (development of next release)
//   0.13.2-sha4
//   0.14.0 (released build)
//   ...
//   ...
//   1.0.0 (released build)
//   ...

#include <string>
#include "version.inc"

std::uint32_t GetVersionInt(int major = LC0_VERSION_MAJOR,
                            int minor = LC0_VERSION_MINOR,
                            int patch = LC0_VERSION_PATCH);

std::string GetVersionStr(int major = LC0_VERSION_MAJOR,
                          int minor = LC0_VERSION_MINOR,
                          int patch = LC0_VERSION_PATCH,
                          const std::string& postfix = LC0_VERSION_POSTFIX);
