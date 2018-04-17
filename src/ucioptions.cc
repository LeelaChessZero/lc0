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
#include "utils/commandline.h"

namespace lczero {

UciOptions::OptionParser::OptionParser(const std::string& name,
                                       const std::string& long_flag,
                                       char short_flag)
    : name_(name), long_flag_(long_flag), short_flag_(short_flag) {}

UciOptions::UciOptions() { contexts_.emplace("", &defaults_); }

std::vector<std::string> UciOptions::ListOptionsUci() const {
  std::vector<std::string> result;
  for (const auto& iter : options_) {
    result.emplace_back("option name " + iter->GetName() + " " +
                        iter->GetOptionString(defaults_));
  }
  return result;
}

void UciOptions::SetOption(const std::string& name, const std::string& value) {
  auto option = FindOptionByName(name);
  if (option) {
    option->SetValue(value, GetMutableOptions());
  }
}

void UciOptions::SendOption(const std::string& name) {
  auto option = FindOptionByName(name);
  if (option) {
    option->SendValue(GetOptionsDict());
  }
}

void UciOptions::SendAllOptions() {
  for (const auto& x : options_) {
    x->SendValue(GetOptionsDict());
  }
}

UciOptions::OptionParser* UciOptions::FindOptionByName(
    const std::string& name) const {
  for (const auto& val : options_) {
    if (val->GetName() == name) return val.get();
  }
  return nullptr;
}

OptionsDict* UciOptions::GetMutableOptions(const std::string& context) {
  auto iter = contexts_.find(context);
  if (iter == contexts_.end()) throw Exception("Unknown context: " + context);
  return &iter->second;
}

const OptionsDict& UciOptions::GetOptionsDict(const std::string& context) {
  return *GetMutableOptions(context);
}

bool UciOptions::ProcessAllFlags() {
  for (auto iter = CommandLine::Arguments().begin(),
            end = CommandLine::Arguments().end();
       iter != end; ++iter) {
    std::string param = *iter;
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
        if (option->ProcessLongFlag(param, value, GetMutableOptions())) {
          processed = true;
          break;
        }
      }
      if (!processed) {
        std::cerr << "Unknown command line flag: " << *iter << ".\n";
        std::cerr << "For help run:\n  " << CommandLine::BinaryName()
                  << " --help" << std::endl;
        return false;
      }
      continue;
    }
    if (param.size() == 2 && param[0] == '-') {
      std::string value;
      bool processed = false;
      if (iter + 1 != end) {
        value = *(iter + 1);
      }
      for (auto& option : options_) {
        if (option->ProcessShortFlag(param[1], GetMutableOptions())) {
          processed = true;
          break;
        } else if (option->ProcessShortFlagWithValue(param[1], value,
                                                     GetMutableOptions())) {
          if (!value.empty()) ++iter;
          processed = true;
          break;
        }
      }
      if (!processed) {
        std::cerr << "Unknown command line flag: " << *iter << ".\n";
        std::cerr << "For help run:\n  " << CommandLine::BinaryName()
                  << " --help" << std::endl;
        return false;
      }
      continue;
    }

    std::cerr << "Unknown command line argument: " << *iter << ".\n";
    std::cerr << "For help run:\n  " << CommandLine::BinaryName() << " --help"
              << std::endl;
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
  std::cerr << "Usage: " << CommandLine::BinaryName() << " [ flags... ]"
            << std::endl;
  std::cerr << "\nAllowed command line flags:\n";
  std::cerr << FormatFlag('h', "help", "Show help and exit");
  for (const auto& option : options_) std::cerr << option->GetHelp(defaults_);
}

/////////////////////////////////////////////////////////////////
// StringOption
/////////////////////////////////////////////////////////////////

StringOption::StringOption(const std::string& name,
                           const std::string& long_flag, char short_flag,
                           std::function<void(const std::string&)> setter)
    : OptionParser(name, long_flag, short_flag), setter_(setter) {}

void StringOption::SetValue(const std::string& value, OptionsDict* dict) {
  SetVal(dict, value);
}

bool StringOption::ProcessLongFlag(const std::string& flag,
                                   const std::string& value,
                                   OptionsDict* dict) {
  if (flag == GetLongFlag()) {
    SetVal(dict, value);
    return true;
  }
  return false;
}

bool StringOption::ProcessShortFlagWithValue(char flag,
                                             const std::string& value,
                                             OptionsDict* dict) {
  if (flag == GetShortFlag()) {
    SetVal(dict, value);
    return true;
  }
  return false;
}

std::string StringOption::GetHelp(const OptionsDict& dict) const {
  return FormatFlag(GetShortFlag(), GetLongFlag() + "=STRING", GetName(),
                    GetVal(dict));
}

std::string StringOption::GetOptionString(const OptionsDict& dict) const {
  return "type string default " + GetVal(dict);
}

void StringOption::SendValue(const OptionsDict& dict) const {
  if (setter_) setter_(GetVal(dict));
}

std::string StringOption::GetVal(const OptionsDict& dict) const {
  return dict.Get<ValueType>(GetName());
}

void StringOption::SetVal(OptionsDict* dict, const ValueType& val) const {
  dict->Set<ValueType>(GetName(), val);
}

/////////////////////////////////////////////////////////////////
// SpinOption
/////////////////////////////////////////////////////////////////

SpinOption::SpinOption(const std::string& name, int min, int max,
                       const std::string& long_flag, char short_flag,
                       std::function<void(int)> setter)
    : OptionParser(name, long_flag, short_flag),
      min_(min),
      max_(max),
      setter_(setter) {}

void SpinOption::SetValue(const std::string& value, OptionsDict* dict) {
  SetVal(dict, std::stoi(value));
}

bool SpinOption::ProcessLongFlag(const std::string& flag,
                                 const std::string& value, OptionsDict* dict) {
  if (flag == GetLongFlag()) {
    SetVal(dict, std::stoi(value));
    return true;
  }
  return false;
}

bool SpinOption::ProcessShortFlagWithValue(char flag, const std::string& value,
                                           OptionsDict* dict) {
  if (flag == GetShortFlag()) {
    SetVal(dict, std::stoi(value));
    return true;
  }
  return false;
}

std::string SpinOption::GetHelp(const OptionsDict& dict) const {
  std::string long_flag = GetLongFlag();
  if (!long_flag.empty()) {
    long_flag += "=" + std::to_string(min_) + ".." + std::to_string(max_);
  }
  return FormatFlag(GetShortFlag(), long_flag, GetName(),
                    std::to_string(GetVal(dict)) +
                        "  min: " + std::to_string(min_) +
                        "  max: " + std::to_string(max_));
}

std::string SpinOption::GetOptionString(const OptionsDict& dict) const {
  return "type spin default " + std::to_string(GetVal(dict)) + " min " +
         std::to_string(min_) + " max " + std::to_string(max_);
}

void SpinOption::SendValue(const OptionsDict& dict) const {
  if (setter_) setter_(GetVal(dict));
}

SpinOption::ValueType SpinOption::GetVal(const OptionsDict& dict) const {
  return dict.Get<ValueType>(GetName());
}

void SpinOption::SetVal(OptionsDict* dict, const ValueType& val) const {
  dict->Set<ValueType>(GetName(), val);
}

/////////////////////////////////////////////////////////////////
// CheckOption
/////////////////////////////////////////////////////////////////

CheckOption::CheckOption(const std::string& name, const std::string& long_flag,
                         char short_flag, std::function<void(bool)> setter)
    : OptionParser(name, long_flag, short_flag), setter_(setter) {}

void CheckOption::SetValue(const std::string& value, OptionsDict* dict) {
  SetVal(dict, value == "true");
}

bool CheckOption::ProcessLongFlag(const std::string& flag,
                                  const std::string& value, OptionsDict* dict) {
  if (flag == GetLongFlag()) {
    SetVal(dict, value.empty() || (value != "off" && value != "false"));
    return true;
  }
  if (flag == "no-" + GetLongFlag()) {
    SetVal(dict, false);
    return true;
  }
  return false;
}

bool CheckOption::ProcessShortFlagWithValue(char flag, const std::string& value,
                                            OptionsDict* dict) {
  if (flag == GetShortFlag()) {
    SetVal(dict, !GetVal(*dict));
    return true;
  }
  return false;
}

std::string CheckOption::GetHelp(const OptionsDict& dict) const {
  std::string long_flag = GetLongFlag();
  if (!long_flag.empty()) {
    long_flag = "[no-]" + long_flag;
  }
  return FormatFlag(GetShortFlag(), long_flag, GetName(),
                    GetVal(dict) ? "true" : "false");
}

std::string CheckOption::GetOptionString(const OptionsDict& dict) const {
  return "type check default " + std::string(GetVal(dict) ? "true" : "false");
}

void CheckOption::SendValue(const OptionsDict& dict) const {
  if (setter_) setter_(GetVal(dict));
}

CheckOption::ValueType CheckOption::GetVal(const OptionsDict& dict) const {
  return dict.Get<ValueType>(GetName());
}

void CheckOption::SetVal(OptionsDict* dict, const ValueType& val) const {
  dict->Set<ValueType>(GetName(), val);
}

}  // namespace lczero
