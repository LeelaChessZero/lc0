/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2019 The LCZero Authors

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

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#include "optionsparser.h"

#include <charconv>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "utils/commandline.h"
#include "utils/configfile.h"
#include "utils/logging.h"
#include "utils/string.h"

namespace lczero {
namespace {
const int kHelpIndent = 15;
const int kUciLineIndent = 15;
const int kHelpWidth = 80;
}  // namespace

OptionsParser::Option::Option(const OptionId& id) : id_(id) {}

OptionsParser::OptionsParser() : values_(*defaults_.AddSubdict("values")) {}

std::vector<std::string> OptionsParser::ListOptionsUci() const {
  std::vector<std::string> result;
  for (const auto& iter : options_) {
    if (!iter->GetUciOption().empty() &&
        (iter->GetId().visibility_mask() & visibility_mode_)) {
      result.emplace_back("option name " + iter->GetUciOption() + " " +
                          iter->GetOptionString(values_));
    }
  }
  return result;
}

void OptionsParser::SetUciOption(const std::string& name,
                                 const std::string& value,
                                 const std::string& context) {
  auto option = FindOptionByUciName(name);
  if (option) {
    option->SetValue(value, GetMutableOptions(context));
    return;
  }
  throw Exception("Unknown option: " + name);
}

OptionsParser::Option* OptionsParser::FindOptionByLongFlag(
    const std::string& flag) const {
  for (const auto& val : options_) {
    auto longflg = val->GetLongFlag();
    if (flag == longflg || flag == ("no-" + longflg)) return val.get();
  }
  return nullptr;
}

OptionsParser::Option* OptionsParser::FindOptionByUciName(
    const std::string& name) const {
  for (const auto& val : options_) {
    if (StringsEqualIgnoreCase(val->GetUciOption(), name)) return val.get();
  }
  return nullptr;
}

OptionsParser::Option* OptionsParser::FindOptionById(const OptionId& id) const {
  for (const auto& val : options_) {
    if (id == val->GetId()) return val.get();
  }
  return nullptr;
}

OptionsDict* OptionsParser::GetMutableOptions(const std::string& context) {
  if (context == "") return &values_;
  auto* result = &values_;
  for (const auto& x : StrSplit(context, ".")) {
    result = result->GetMutableSubdict(x);
  }
  return result;
}

const OptionsDict& OptionsParser::GetOptionsDict(const std::string& context) {
  if (context == "") return values_;
  const auto* result = &values_;
  for (const auto& x : StrSplit(context, ".")) {
    result = &result->GetSubdict(x);
  }
  return *result;
}

bool OptionsParser::ProcessAllFlags() {
  return ProcessFlags(ConfigFile::Arguments()) &&
         ProcessFlags(CommandLine::Arguments());
}

bool OptionsParser::ProcessFlags(const std::vector<std::string>& args) {
  auto show_help = false;
  if (CommandLine::BinaryName().find("simple") != std::string::npos) {
    visibility_mode_ = OptionId::kSimpleMode;
  } else if (CommandLine::BinaryName().find("pro") != std::string::npos) {
    visibility_mode_ = OptionId::kProMode;
  }
  for (auto iter = args.begin(), end = args.end(); iter != end; ++iter) {
    std::string param = *iter;
    if (param == "--show-hidden") {
      visibility_mode_ = OptionId::kProMode;
      continue;
    }
    if (param == "-h" || param == "--help") {
      // Set a flag so that --show-hidden after --help works.
      show_help = true;
      continue;
    }

    if (param.substr(0, 2) == "--") {
      std::string context;
      param = param.substr(2);
      std::string value;
      auto pos = param.find('=');
      if (pos != std::string::npos) {
        value = param.substr(pos + 1);
        param = param.substr(0, pos);
      }
      pos = param.rfind('.');
      if (pos != std::string::npos) {
        context = param.substr(0, pos);
        param = param.substr(pos + 1);
      }
      bool processed = false;
      Option* option = FindOptionByLongFlag(param);
      if (option &&
          option->ProcessLongFlag(param, value, GetMutableOptions(context))) {
        processed = true;
      }
      if (!processed) {
        CERR << "Unknown command line flag: " << *iter << ".";
        CERR << "For help run:\n  " << CommandLine::BinaryName() << " --help";
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
        CERR << "Unknown command line flag: " << *iter << ".";
        CERR << "For help run:\n  " << CommandLine::BinaryName() << " --help";
        return false;
      }
      continue;
    }

    CERR << "Unknown command line argument: " << *iter << ".\n";
    CERR << "For help run:\n  " << CommandLine::BinaryName() << " --help";
    return false;
  }
  if (show_help) {
    ShowHelp();
    return false;
  }
  return true;
}

void OptionsParser::AddContext(const std::string& context) {
  values_.AddSubdict(context);
}

namespace {
std ::string FormatFlag(char short_flag, const std::string& long_flag,
                        const std::string& help,
                        const std::string& uci_option = {},
                        const std::string& def = {}) {
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
  std::string long_flag_str = "";
  if (!short_flag && long_flag.empty()) {
    long_flag_str = "(uci parameter)";
  } else {
    long_flag_str = long_flag.empty() ? "" : "--" + long_flag;
  }
  oss << long_flag_str;
  auto help_lines = FlowText(help, kHelpWidth);
  bool is_first_line = true;
  for (const auto& line : help_lines) {
    if (is_first_line) {
      is_first_line = false;
      if (long_flag_str.size() < kHelpIndent - 7) {
        oss << std::string(kHelpIndent - 7 - long_flag_str.size(), ' ') << line
            << "\n";
        continue;
      }
      oss << "\n";
    }
    oss << std::string(kHelpIndent, ' ') << line << "\n";
  }
  if (!def.empty() || !uci_option.empty()) {
    oss << std::string(kUciLineIndent, ' ') << '[';
    if (!uci_option.empty()) oss << "UCI: " << uci_option;
    if (!uci_option.empty() && !def.empty()) oss << "  ";
    if (!def.empty()) oss << "DEFAULT: " << def;
    oss << "]\n";
  }
  oss << '\n';
  return oss.str();
}
}  // namespace

void OptionsParser::ShowHelp() const {
  std::cout << "Usage: " << CommandLine::BinaryName() << " [<mode>] [flags...]"
            << std::endl;

  std::cout << "\nAvailable modes. A help for a mode: "
            << CommandLine::BinaryName() << " <mode> --help\n";
  for (const auto& mode : CommandLine::GetModes()) {
    std::cout << "  " << std::setw(10) << std::left << mode.first << " "
              << mode.second << std::endl;
  }

  std::cout << "\nAllowed command line flags for current mode:\n";
  std::cout << FormatFlag('h', "help", "Show help and exit.");
  std::cout << FormatFlag('\0', "show-hidden",
                          "Show hidden options. Use with --help.");
  for (const auto& option : options_) {
    if ((option->GetId().visibility_mask() & visibility_mode_)) {
      std::cout << option->GetHelp(values_);
    }
  }

  auto contexts = values_.ListSubdicts();
  if (!contexts.empty()) {
    std::cout << "\nFlags can be defined per context (one of: "
              << StrJoin(contexts, ", ") << "), for example:\n";
    std::cout << "       --" << contexts[0] << '.'
              << options_.back()->GetLongFlag() << "=(value)\n";
  }
}

/////////////////////////////////////////////////////////////////
// StringOption
/////////////////////////////////////////////////////////////////

StringOption::StringOption(const OptionId& id) : Option(id) {}

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
  std::string long_flag = GetLongFlag();
  if (!long_flag.empty()) {
    long_flag += "=STRING";
  }
  return FormatFlag(GetShortFlag(), long_flag, GetHelpText(), GetUciOption(),
                    GetVal(dict));
}

std::string StringOption::GetOptionString(const OptionsDict& dict) const {
  return "type string default " + GetVal(dict);
}

std::string StringOption::GetVal(const OptionsDict& dict) const {
  return dict.Get<ValueType>(GetId());
}

void StringOption::SetVal(OptionsDict* dict, const ValueType& val) const {
  dict->Set<ValueType>(GetId(), val);
}

/////////////////////////////////////////////////////////////////
// IntOption
/////////////////////////////////////////////////////////////////

IntOption::IntOption(const OptionId& id, int min, int max)
    : Option(id), min_(min), max_(max) {}

void IntOption::SetValue(const std::string& value, OptionsDict* dict) {
  SetVal(dict, ValidateIntString(value));
}

bool IntOption::ProcessLongFlag(const std::string& flag,
                                const std::string& value, OptionsDict* dict) {
  if (flag == GetLongFlag()) {
    SetVal(dict, ValidateIntString(value));
    return true;
  }
  return false;
}

bool IntOption::ProcessShortFlagWithValue(char flag, const std::string& value,
                                          OptionsDict* dict) {
  if (flag == GetShortFlag()) {
    SetVal(dict, ValidateIntString(value));
    return true;
  }
  return false;
}

std::string IntOption::GetHelp(const OptionsDict& dict) const {
  std::string long_flag = GetLongFlag();
  if (!long_flag.empty()) {
    long_flag += "=" + std::to_string(min_) + ".." + std::to_string(max_);
  }
  return FormatFlag(GetShortFlag(), long_flag, GetHelpText(), GetUciOption(),
                    std::to_string(GetVal(dict)) +
                        "  MIN: " + std::to_string(min_) +
                        "  MAX: " + std::to_string(max_));
}

std::string IntOption::GetOptionString(const OptionsDict& dict) const {
  return "type spin default " + std::to_string(GetVal(dict)) + " min " +
         std::to_string(min_) + " max " + std::to_string(max_);
}

IntOption::ValueType IntOption::GetVal(const OptionsDict& dict) const {
  return dict.Get<ValueType>(GetId());
}

void IntOption::SetVal(OptionsDict* dict, const ValueType& val) const {
  if (val < min_ || val > max_) {
    std::ostringstream buf;
    buf << "Flag '--" << GetLongFlag() << "' must be between " << min_
        << " and " << max_ << ".";
    throw Exception(buf.str());
  }
  dict->Set<ValueType>(GetId(), val);
}

int IntOption::ValidateIntString(const std::string& val) const {
  int result;
  const auto end = val.data() + val.size();
  auto [ptr, err] = std::from_chars(val.data(), end, result);
  if (err == std::errc::invalid_argument) {
    throw Exception("Flag '--" + GetLongFlag() + "' has an invalid format.");
  } else if (err == std::errc::result_out_of_range) {
    throw Exception("Flag '--" + GetLongFlag() + "' is out of range.");
  } else if (ptr != end) {
    throw Exception("Flag '--" + GetLongFlag() + "' has trailing characters.");
  } else {
    return result;
  }
}

/////////////////////////////////////////////////////////////////
// FloatOption
/////////////////////////////////////////////////////////////////

FloatOption::FloatOption(const OptionId& id, float min, float max)
    : Option(id), min_(min), max_(max) {}

void FloatOption::SetValue(const std::string& value, OptionsDict* dict) {
  try {
    SetVal(dict, std::stof(value));
  } catch (std::invalid_argument&) {
    throw Exception("invalid value " + value);
  } catch (const std::out_of_range&) {
    throw Exception("out of range value " + value);
  }
}

bool FloatOption::ProcessLongFlag(const std::string& flag,
                                  const std::string& value, OptionsDict* dict) {
  if (flag == GetLongFlag()) {
    try {
      SetVal(dict, std::stof(value));
    } catch (std::invalid_argument&) {
      throw Exception("invalid value " + value);
    } catch (const std::out_of_range&) {
      throw Exception("out of range value " + value);
    }
    return true;
  }
  return false;
}

bool FloatOption::ProcessShortFlagWithValue(char flag, const std::string& value,
                                            OptionsDict* dict) {
  if (flag == GetShortFlag()) {
    try {
      SetVal(dict, std::stof(value));
    } catch (std::invalid_argument&) {
      throw Exception("invalid value " + value);
    } catch (const std::out_of_range&) {
      throw Exception("out of range value " + value);
    }
    return true;
  }
  return false;
}

std::string FloatOption::GetHelp(const OptionsDict& dict) const {
  std::string long_flag = GetLongFlag();
  if (!long_flag.empty()) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << min_ << ".." << max_;
    long_flag += "=" + oss.str();
  }
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2) << GetVal(dict) << "  MIN: " << min_
      << "  MAX: " << max_;
  return FormatFlag(GetShortFlag(), long_flag, GetHelpText(), GetUciOption(),
                    oss.str());
}

std::string FloatOption::GetOptionString(const OptionsDict& dict) const {
  return "type string default " + std::to_string(GetVal(dict));
}

FloatOption::ValueType FloatOption::GetVal(const OptionsDict& dict) const {
  return dict.Get<ValueType>(GetId());
}

void FloatOption::SetVal(OptionsDict* dict, const ValueType& val) const {
  if (val < min_ || val > max_) {
    std::ostringstream buf;
    buf << "Flag '--" << GetLongFlag() << "' must be between " << min_
        << " and " << max_ << ".";
    throw Exception(buf.str());
  }
  dict->Set<ValueType>(GetId(), val);
}

/////////////////////////////////////////////////////////////////
// BoolOption
/////////////////////////////////////////////////////////////////

BoolOption::BoolOption(const OptionId& id) : Option(id) {}

void BoolOption::SetValue(const std::string& value, OptionsDict* dict) {
  ValidateBoolString(value);
  SetVal(dict, value == "true");
}

bool BoolOption::ProcessLongFlag(const std::string& flag,
                                 const std::string& value, OptionsDict* dict) {
  if (flag == "no-" + GetLongFlag()) {
    SetVal(dict, false);
    return true;
  }
  if (flag == GetLongFlag() && value.empty()) {
    SetVal(dict, true);
    return true;
  }

  ValidateBoolString(value);

  if (flag == GetLongFlag()) {
    SetVal(dict, value.empty() || (value != "false"));
    return true;
  }
  return false;
}

bool BoolOption::ProcessShortFlag(char flag, OptionsDict* dict) {
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
  return FormatFlag(GetShortFlag(), long_flag, GetHelpText(), GetUciOption(),
                    GetVal(dict) ? "true" : "false");
}

std::string BoolOption::GetOptionString(const OptionsDict& dict) const {
  return "type check default " + std::string(GetVal(dict) ? "true" : "false");
}

BoolOption::ValueType BoolOption::GetVal(const OptionsDict& dict) const {
  return dict.Get<ValueType>(GetId());
}

void BoolOption::SetVal(OptionsDict* dict, const ValueType& val) const {
  dict->Set<ValueType>(GetId(), val);
}

void BoolOption::ValidateBoolString(const std::string& val) {
  if (val != "true" && val != "false") {
    std::ostringstream buf;
    buf << "Flag '--" << GetLongFlag() << "' must be either "
        << "'true' or 'false'.";
    throw Exception(buf.str());
  }
}

/////////////////////////////////////////////////////////////////
// ButtonOption
/////////////////////////////////////////////////////////////////

ButtonOption::ButtonOption(const OptionId& id) : Option(id) {}

void ButtonOption::SetValue(const std::string& /*value*/, OptionsDict* dict) {
  dict->Set<ValueType>(GetId(), true);
}

bool ButtonOption::ProcessLongFlag(const std::string& /*flag*/,
                                   const std::string& /*value*/,
                                   OptionsDict* /*dict*/) {
  return false;
}

bool ButtonOption::ProcessShortFlag(char /*flag*/, OptionsDict* /*dict*/) {
  return false;
}

std::string ButtonOption::GetHelp(const OptionsDict& /*dict*/) const {
  return "";
}

std::string ButtonOption::GetOptionString(const OptionsDict& /*dict*/) const {
  return "type button";
}

/////////////////////////////////////////////////////////////////
// ChoiceOption
/////////////////////////////////////////////////////////////////

ChoiceOption::ChoiceOption(const OptionId& id,
                           const std::vector<std::string>& choices)
    : Option(id), choices_(choices) {}

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
  return FormatFlag(GetShortFlag(), GetLongFlag() + "=CHOICE", GetHelpText(),
                    GetUciOption(), GetVal(dict) + "  VALUES: " + values);
}

std::string ChoiceOption::GetOptionString(const OptionsDict& dict) const {
  std::string res = "type combo default " + GetVal(dict);
  for (const auto& choice : choices_) {
    res += " var " + choice;
  }
  return res;
}

std::string ChoiceOption::GetVal(const OptionsDict& dict) const {
  return dict.Get<ValueType>(GetId());
}

void ChoiceOption::SetVal(OptionsDict* dict, const ValueType& val) const {
  bool valid = false;
  std::string choice_string;
  for (const auto& choice : choices_) {
    choice_string += " " + choice;
    if (val == choice) {
      valid = true;
      break;
    }
  }
  if (!valid) {
    std::ostringstream buf;
    buf << "Flag '--" << GetLongFlag() << "' must be one of the "
        << "following values:" << choice_string << ".";
    throw Exception(buf.str());
  }
  dict->Set<ValueType>(GetId(), val);
}

}  // namespace lczero
