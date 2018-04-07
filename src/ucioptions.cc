/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "ucioptions.h"

#include <iomanip>
#include <iostream>
#include <sstream>

namespace lczero {

std::vector<std::string> UciOptions::ListOptionsUci() const {
  std::vector<std::string> result;
  for (const auto& iter : options_) {
    result.emplace_back("option name " + iter->GetName() + " " +
                        iter->GetOptionString());
  }
  return result;
}

void UciOptions::SetOption(const std::string& name, const std::string& value) {
  auto option = FindOptionByName(name);
  if (option) {
    option->SetValue(value);
  }
}

void UciOptions::SendAllOptions() {
  for (const auto& x : options_) {
    x->SendValue();
  }
}

void UciOptions::SendOption(const std::string& name) {
  auto option = FindOptionByName(name);
  if (option) {
    option->SendValue();
  }
}

void UciOptions::Add(std::unique_ptr<Option> option) {
  options_.emplace_back(std::move(option));
}

UciOptions::Option* UciOptions::FindOptionByName(
    const std::string& name) const {
  for (const auto& val : options_) {
    if (val->GetName() == name) return val.get();
  }
  return nullptr;
}

bool UciOptions::ProcessAllFlags() {
  for (const char **argv = argv_ + 1, **end = argv_ + argc_; argv != end;
       ++argv) {
    std::string param = *argv;
    if (param == "-h" || param == "--help") {
      ShowHelp();
      return false;
    }

    if (param.substr(0, 2) == "--") {
      param = param.substr(2);
      std::string value;
      auto pos = param.find('=');
      if (pos != std::string::npos) {
        value = param.substr(pos + 1);
        param = param.substr(0, pos);
      }
      bool processed = false;
      for (auto& option : options_) {
        if (option->ProcessLongFlag(param, value)) {
          processed = true;
          break;
        }
      }
      if (!processed) {
        std::cerr << "Unknown command line flag: " << *argv << ".\n";
        std::cerr << "For help run:\n  " << argv_[0] << " --help" << std::endl;
        return false;
      }
      continue;
    }
    if (param.size() == 2 && param[0] == '-') {
      std::string value;
      bool processed = false;
      if (argv + 1 != end) {
        value = *(argv + 1);
      }
      for (auto& option : options_) {
        if (option->ProcessShortFlag(param[1])) {
          processed = true;
          break;
        } else if (option->ProcessShortFlagWithValue(param[1], value)) {
          if (!value.empty()) ++argv;
          processed = true;
          break;
        }
      }
      if (!processed) {
        std::cerr << "Unknown command line flag: " << *argv << ".\n";
        std::cerr << "For help run:\n  " << argv_[0] << " --help" << std::endl;
        return false;
      }
      continue;
    }

    std::cerr << "Unknown command line argument: " << *argv << ".\n";
    std::cerr << "For help run:\n  " << argv_[0] << " --help" << std::endl;
    return false;
  }
  return true;
}

namespace {
std ::string FormatFlag(char short_flag, const std::string& long_flag,
                        const std::string& help, const std::string& def = {}) {
  std::ostringstream oss;
  oss << "  ";
  if (short_flag) {
    oss << '-' << short_flag;
  } else {
    oss << "  ";
  }
  if (short_flag && !long_flag.empty()) {
    oss << ",  ";
  } else {
    oss << "   ";
  }
  oss << std::setw(20) << std::left;
  if (!short_flag && long_flag.empty()) {
    oss << "(uci parameter)";
  } else {
    oss << (long_flag.empty() ? "" : "--" + long_flag);
  }
  oss << ' ' << help << ".\n";
  if (!def.empty()) {
    oss << std::string(28, ' ') << "(default: " << def << ")\n";
  }
  return oss.str();
}

}  // namespace

bool StringOption::ProcessLongFlag(const std::string& flag,
                                   const std::string& value) {
  if (flag == GetLongFlag()) {
    value_ = value;
    return true;
  }
  return false;
}

bool StringOption::ProcessShortFlagWithValue(char flag,
                                             const std::string& value) {
  if (flag == GetShortFlag()) {
    value_ = value;
    return true;
  }
  return false;
}

std::string StringOption::GetHelp() const {
  return FormatFlag(GetShortFlag(), GetLongFlag() + "=STRING", GetName(),
                    value_);
}

bool SpinOption::ProcessLongFlag(const std::string& flag,
                                 const std::string& value) {
  if (flag == GetLongFlag()) {
    value_ = std::stoi(value);
    return true;
  }
  return false;
}

bool SpinOption::ProcessShortFlagWithValue(char flag,
                                           const std::string& value) {
  if (flag == GetShortFlag()) {
    value_ = std::stoi(value);
    return true;
  }
  return false;
}

std::string SpinOption::GetHelp() const {
  std::string long_flag = GetLongFlag();
  if (!long_flag.empty()) {
    long_flag += "=" + std::to_string(min_) + ".." + std::to_string(max_);
  }
  return FormatFlag(GetShortFlag(), long_flag, GetName(),
                    std::to_string(value_) + "  min: " + std::to_string(min_) +
                        "  max: " + std::to_string(max_));
}

bool CheckOption::ProcessLongFlag(const std::string& flag,
                                  const std::string& value) {
  if (flag == GetLongFlag()) {
    value_ = (value.empty() || value == "on" || value == "true");
    return true;
  }
  if (flag == "no-" + GetLongFlag()) {
    value_ = false;
    return true;
  }
  return false;
}

bool CheckOption::ProcessShortFlag(char flag) {
  if (flag == GetShortFlag()) {
    value_ = !value_;
    return true;
  }
  return false;
}

std::string CheckOption::GetHelp() const {
  std::string long_flag = GetLongFlag();
  if (!long_flag.empty()) {
    long_flag = "[no-]" + long_flag;
  }
  return FormatFlag(GetShortFlag(), long_flag, GetName(),
                    value_ ? "true" : "false");
}

void UciOptions::ShowHelp() const {
  std::cerr << "Usage: " << argv_[0] << " [ flags... ]" << std::endl;
  std::cerr << "\nAllowed command line flags:\n";
  std::cerr << FormatFlag('h', "help", "Show help and exit");
  for (const auto& option : options_) std::cerr << option->GetHelp();
}

}  // namespace lczero