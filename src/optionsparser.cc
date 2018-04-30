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

#include "optionsparser.h"

#include <iomanip>
#include <iostream>
#include <sstream>
#include "utils/commandline.h"

namespace lczero {

OptionsParser::Option::Option(const std::string& name,
                              const std::string& long_flag, char short_flag)
    : name_(name), long_flag_(long_flag), short_flag_(short_flag) {}

OptionsParser::OptionsParser() : values_(*defaults_.AddSubdict("values")) {}

std::vector<std::string> OptionsParser::ListOptionsUci() const {
  std::vector<std::string> result;
  for (const auto& iter : options_) {
    result.emplace_back("option name " + iter->GetName() + " " +
                        iter->GetOptionString(defaults_));
  }
  return result;
}

void OptionsParser::SetOption(const std::string& name, const std::string& value,
                              const std::string& context) {
  auto option = FindOptionByName(name);
  if (option) {
    option->SetValue(value, GetMutableOptions(context));
  }
}

void OptionsParser::SendOption(const std::string& name) {
  auto option = FindOptionByName(name);
  if (option) {
    option->SendValue(GetOptionsDict());
  }
}

void OptionsParser::SendAllOptions() {
  for (const auto& x : options_) {
    x->SendValue(GetOptionsDict());
  }
}

OptionsParser::Option* OptionsParser::FindOptionByName(
    const std::string& name) const {
  for (const auto& val : options_) {
    if (val->GetName() == name) return val.get();
  }
  return nullptr;
}

OptionsDict* OptionsParser::GetMutableOptions(const std::string& context) {
  if (context == "") return &values_;
  return values_.GetMutableSubdict(context);
}

const OptionsDict& OptionsParser::GetOptionsDict(const std::string& context) {
  if (context == "") return values_;
  return values_.GetSubdict(context);
}

bool OptionsParser::ProcessAllFlags() {
  std::string context;
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
        if (option->ProcessLongFlag(param, value, GetMutableOptions(context))) {
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
        if (option->ProcessShortFlag(param[1], GetMutableOptions(context))) {
          processed = true;
          break;
        } else if (option->ProcessShortFlagWithValue(
                       param[1], value, GetMutableOptions(context))) {
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

    if (!param.empty() && param[param.size() - 1] == ':') {
      context = param.substr(0, param.size() - 1);
      continue;
    }

    std::cerr << "Unknown command line argument: " << *iter << ".\n";
    std::cerr << "For help run:\n  " << CommandLine::BinaryName() << " --help"
              << std::endl;
    return false;
  }
  return true;
}

void OptionsParser::AddContext(const std::string& context) {
  values_.AddSubdict(context);
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

void OptionsParser::ShowHelp() const {
  std::cerr << "Usage: " << CommandLine::BinaryName() << " [<mode>] [flags...]";
  for (const auto& context : values_.ListSubdicts()) {
    std::cerr << " [" << context << ": flags...]";
  }

  std::cerr << std::endl;
  std::cerr << "\nAvailable modes. A help for a mode: "
            << CommandLine::BinaryName() << " <mode> --help\n";
  for (const auto& mode : CommandLine::GetModes()) {
    std::cerr << "  " << std::setw(10) << std::left << mode.first << " "
              << mode.second << std::endl;
  }
  std::cerr << "\nAllowed command line flags for current mode:\n";
  std::cerr << FormatFlag('h', "help", "Show help and exit");
  for (const auto& option : options_) std::cerr << option->GetHelp(defaults_);
}

/////////////////////////////////////////////////////////////////
// StringOption
/////////////////////////////////////////////////////////////////

StringOption::StringOption(const std::string& name,
                           const std::string& long_flag, char short_flag,
                           std::function<void(const std::string&)> setter)
    : Option(name, long_flag, short_flag), setter_(setter) {}

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
// IntOption
/////////////////////////////////////////////////////////////////

IntOption::IntOption(const std::string& name, int min, int max,
                     const std::string& long_flag, char short_flag,
                     std::function<void(int)> setter)
    : Option(name, long_flag, short_flag),
      min_(min),
      max_(max),
      setter_(setter) {}

void IntOption::SetValue(const std::string& value, OptionsDict* dict) {
  SetVal(dict, std::stoi(value));
}

bool IntOption::ProcessLongFlag(const std::string& flag,
                                const std::string& value, OptionsDict* dict) {
  if (flag == GetLongFlag()) {
    SetVal(dict, std::stoi(value));
    return true;
  }
  return false;
}

bool IntOption::ProcessShortFlagWithValue(char flag, const std::string& value,
                                          OptionsDict* dict) {
  if (flag == GetShortFlag()) {
    SetVal(dict, std::stoi(value));
    return true;
  }
  return false;
}

std::string IntOption::GetHelp(const OptionsDict& dict) const {
  std::string long_flag = GetLongFlag();
  if (!long_flag.empty()) {
    long_flag += "=" + std::to_string(min_) + ".." + std::to_string(max_);
  }
  return FormatFlag(GetShortFlag(), long_flag, GetName(),
                    std::to_string(GetVal(dict)) +
                        "  min: " + std::to_string(min_) +
                        "  max: " + std::to_string(max_));
}

std::string IntOption::GetOptionString(const OptionsDict& dict) const {
  return "type spin default " + std::to_string(GetVal(dict)) + " min " +
         std::to_string(min_) + " max " + std::to_string(max_);
}

void IntOption::SendValue(const OptionsDict& dict) const {
  if (setter_) setter_(GetVal(dict));
}

IntOption::ValueType IntOption::GetVal(const OptionsDict& dict) const {
  return dict.Get<ValueType>(GetName());
}

void IntOption::SetVal(OptionsDict* dict, const ValueType& val) const {
  dict->Set<ValueType>(GetName(), val);
}

/////////////////////////////////////////////////////////////////
// FloatOption
/////////////////////////////////////////////////////////////////

FloatOption::FloatOption(const std::string& name, float min, float max,
                         const std::string& long_flag, char short_flag,
                         std::function<void(float)> setter)
    : Option(name, long_flag, short_flag),
      min_(min),
      max_(max),
      setter_(setter) {}

void FloatOption::SetValue(const std::string& value, OptionsDict* dict) {
  SetVal(dict, std::stof(value));
}

bool FloatOption::ProcessLongFlag(const std::string& flag,
                                  const std::string& value, OptionsDict* dict) {
  if (flag == GetLongFlag()) {
    SetVal(dict, std::stof(value));
    return true;
  }
  return false;
}

bool FloatOption::ProcessShortFlagWithValue(char flag, const std::string& value,
                                            OptionsDict* dict) {
  if (flag == GetShortFlag()) {
    SetVal(dict, std::stof(value));
    return true;
  }
  return false;
}

std::string FloatOption::GetHelp(const OptionsDict& dict) const {
  std::string long_flag = GetLongFlag();
  if (!long_flag.empty()) {
    long_flag += "=" + std::to_string(min_) + ".." + std::to_string(max_);
  }
  return FormatFlag(GetShortFlag(), long_flag, GetName(),
                    std::to_string(GetVal(dict)) +
                        "  min: " + std::to_string(min_) +
                        "  max: " + std::to_string(max_));
}

std::string FloatOption::GetOptionString(const OptionsDict& dict) const {
  return "type string default " + std::to_string(GetVal(dict));
}

void FloatOption::SendValue(const OptionsDict& dict) const {
  if (setter_) setter_(GetVal(dict));
}

FloatOption::ValueType FloatOption::GetVal(const OptionsDict& dict) const {
  return dict.Get<ValueType>(GetName());
}

void FloatOption::SetVal(OptionsDict* dict, const ValueType& val) const {
  dict->Set<ValueType>(GetName(), val);
}

/////////////////////////////////////////////////////////////////
// BoolOption
/////////////////////////////////////////////////////////////////

BoolOption::BoolOption(const std::string& name, const std::string& long_flag,
                       char short_flag, std::function<void(bool)> setter)
    : Option(name, long_flag, short_flag), setter_(setter) {}

void BoolOption::SetValue(const std::string& value, OptionsDict* dict) {
  SetVal(dict, value == "true");
}

bool BoolOption::ProcessLongFlag(const std::string& flag,
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

bool BoolOption::ProcessShortFlagWithValue(char flag, const std::string& value,
                                           OptionsDict* dict) {
  if (flag == GetShortFlag()) {
    SetVal(dict, !GetVal(*dict));
    return true;
  }
  return false;
}

std::string BoolOption::GetHelp(const OptionsDict& dict) const {
  std::string long_flag = GetLongFlag();
  if (!long_flag.empty()) {
    long_flag = "[no-]" + long_flag;
  }
  return FormatFlag(GetShortFlag(), long_flag, GetName(),
                    GetVal(dict) ? "true" : "false");
}

std::string BoolOption::GetOptionString(const OptionsDict& dict) const {
  return "type check default " + std::string(GetVal(dict) ? "true" : "false");
}

void BoolOption::SendValue(const OptionsDict& dict) const {
  if (setter_) setter_(GetVal(dict));
}

BoolOption::ValueType BoolOption::GetVal(const OptionsDict& dict) const {
  return dict.Get<ValueType>(GetName());
}

void BoolOption::SetVal(OptionsDict* dict, const ValueType& val) const {
  dict->Set<ValueType>(GetName(), val);
}

/////////////////////////////////////////////////////////////////
// ChoiceOption
/////////////////////////////////////////////////////////////////

ChoiceOption::ChoiceOption(const std::string& name,
                           const std::vector<std::string>& choices,
                           const std::string& long_flag, char short_flag,
                           std::function<void(const std::string&)> setter)
    : Option(name, long_flag, short_flag), setter_(setter), choices_(choices) {}

void ChoiceOption::SetValue(const std::string& value, OptionsDict* dict) {
  SetVal(dict, value);
}

bool ChoiceOption::ProcessLongFlag(const std::string& flag,
                                   const std::string& value,
                                   OptionsDict* dict) {
  if (flag == GetLongFlag()) {
    SetVal(dict, value);
    return true;
  }
  return false;
}

bool ChoiceOption::ProcessShortFlagWithValue(char flag,
                                             const std::string& value,
                                             OptionsDict* dict) {
  if (flag == GetShortFlag()) {
    SetVal(dict, value);
    return true;
  }
  return false;
}

std::string ChoiceOption::GetHelp(const OptionsDict& dict) const {
  std::string values;
  for (const auto& choice : choices_) {
    if (!values.empty()) values += ',';
    values += choice;
  }
  return FormatFlag(GetShortFlag(), GetLongFlag() + "=CHOICE", GetName(),
                    GetVal(dict) + "  values: " + values);
}

std::string ChoiceOption::GetOptionString(const OptionsDict& dict) const {
  std::string res = "type combo default " + GetVal(dict);
  for (const auto& choice : choices_) {
    res += " var " + choice;
  }
  return res;
}

void ChoiceOption::SendValue(const OptionsDict& dict) const {
  if (setter_) setter_(GetVal(dict));
}

std::string ChoiceOption::GetVal(const OptionsDict& dict) const {
  return dict.Get<ValueType>(GetName());
}

void ChoiceOption::SetVal(OptionsDict* dict, const ValueType& val) const {
  dict->Set<ValueType>(GetName(), val);
}

}  // namespace lczero
