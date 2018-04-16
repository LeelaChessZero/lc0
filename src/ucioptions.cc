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

UciOptions::Option::Option(const std::string& name,
                           const std::string& long_flag, char short_flag)
    : name_(name), long_flag_(long_flag), short_flag_(short_flag) {}

int UciOptions::Option::GetIntValue() const {
  throw Exception("Unsupported command line value type.");
}

std::string UciOptions::Option::GetStringValue() const {
  throw Exception("Unsupported command line value type.");
}

// If has boolean value, return, otherwise throw exception.
bool UciOptions::Option::GetBoolValue() const {
  throw Exception("Unsupported command line value type.");
}

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

const UciOptions::Option* UciOptions::GetOption(const std::string& name) const {
  auto x = FindOptionByName(name);
  if (!x) throw Exception("Unknown option: " + name);
  return x;
}

int UciOptions::GetIntValue(const std::string& name) const {
  return GetOption(name)->GetIntValue();
}

bool UciOptions::GetBoolValue(const std::string& name) const {
  return GetOption(name)->GetBoolValue();
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
  oss << std::setw(30) << std::left;
  if (!short_flag && long_flag.empty()) {
    oss << "(uci parameter)";
  } else {
    oss << (long_flag.empty() ? "" : "--" + long_flag);
  }
  oss << ' ' << help << ".\n";
  if (!def.empty()) {
    oss << std::string(38, ' ') << "(default: " << def << ")\n";
  }
  return oss.str();
}
}  // namespace

void UciOptions::ShowHelp() const {
  std::cerr << "Usage: " << argv_[0] << " [ flags... ]" << std::endl;
  std::cerr << "\nAllowed command line flags:\n";
  std::cerr << FormatFlag('h', "help", "Show help and exit");
  for (const auto& option : options_) std::cerr << option->GetHelp();
}

/////////////////////////////////////////////////////////////////
// StringOption
/////////////////////////////////////////////////////////////////

StringOption::StringOption(const std::string& name, const std::string& def,
                           std::function<void(const std::string&)> setter,
                           const std::string& long_flag, char short_flag)
    : Option(name, long_flag, short_flag), value_(def), setter_(setter) {}

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

std::string StringOption::GetOptionString() const {
  return "type string default " + value_;
}

void StringOption::SendValue() const {
  if (setter_) setter_(value_);
}

/////////////////////////////////////////////////////////////////
// SpinOption
/////////////////////////////////////////////////////////////////

SpinOption::SpinOption(const std::string& name, int def, int min, int max,
                       std::function<void(int)> setter,
                       const std::string& long_flag, char short_flag)
    : Option(name, long_flag, short_flag),
      value_(def),
      min_(min),
      max_(max),
      setter_(setter) {}

void SpinOption::SetValue(const std::string& value) {
  value_ = std::stoi(value);
}

int SpinOption::GetIntValue() const { return value_; }

std::string SpinOption::GetOptionString() const {
  return "type spin default " + std::to_string(value_) + " min " +
         std::to_string(min_) + " max " + std::to_string(max_);
}

void SpinOption::SendValue() const {
  if (setter_) setter_(value_);
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

/////////////////////////////////////////////////////////////////
// SpinOption
/////////////////////////////////////////////////////////////////

CheckOption::CheckOption(const std::string& name, bool def,
                         std::function<void(bool)> setter,
                         const std::string& long_flag, char short_flag)
    : Option(name, long_flag, short_flag), value_(def), setter_(setter) {}

void CheckOption::SetValue(const std::string& value) {
  value_ = (value == "true");
}

std::string CheckOption::GetOptionString() const {
  return "type check default " + std::string(value_ ? "true" : "false");
}

void CheckOption::SendValue() const {
  if (setter_) setter_(value_);
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

}  // namespace lczero