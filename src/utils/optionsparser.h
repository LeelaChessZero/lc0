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

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "utils/exception.h"
#include "utils/optionsdict.h"

namespace lczero {

struct OptionId {
  OptionId(const char* long_flag = "", const char* uci_option = "",
           const char* help_text = "", const char short_flag = '\0')
      : long_flag(long_flag),
        uci_option(uci_option),
        help_text(help_text),
        short_flag(short_flag) {}

  OptionId(const OptionId& other) = delete;

  const char* const long_flag;
  const char* const uci_option;
  const char* const help_text;
  const char short_flag;

  // Returns Option's own address as string.
  // TODO(crem) Generalize OptionsDict to have a version which has OptionId*
  //            as keys instead of std::string.
  std::string GetId() const {
    return std::to_string(reinterpret_cast<intptr_t>(this));
  }
};

class OptionsParser {
 public:
  OptionsParser();

  class Option {
   public:
    Option(const OptionId& id);
    virtual ~Option(){};
    // Set value from string.
    virtual void SetValue(const std::string& value, OptionsDict* dict) = 0;

   protected:
    std::string GetId() const { return id_.GetId(); }
    std::string GetUciOption() const { return id_.uci_option; }
    std::string GetHelpText() const { return id_.help_text; }
    std::string GetLongFlag() const { return id_.long_flag; }
    char GetShortFlag() const { return id_.short_flag; }

   private:
    virtual std::string GetOptionString(const OptionsDict& dict) const = 0;
    virtual bool ProcessLongFlag(const std::string& /*flag*/,
                                 const std::string& /*value*/,
                                 OptionsDict* /*dict*/) {
      return false;
    }
    virtual bool ProcessShortFlag(char /*flag*/, OptionsDict* /*dict*/) {
      return false;
    }
    virtual bool ProcessShortFlagWithValue(char /*flag*/,
                                           const std::string& /*value*/,
                                           OptionsDict* /*dict*/) {
      return false;
    }
    virtual std::string GetHelp(const OptionsDict& dict) const = 0;

    const OptionId& id_;
    bool hidden_ = false;
    friend class OptionsParser;
  };

  // Add an option to the list of available options (from command line flags
  // or UCI params)
  // Usage:
  // options->Add<StringOption>(name, func, long_flag, short_flag) = def_val;
  template <typename Option, typename... Args>
  typename Option::ValueType& Add(Args&&... args) {
    options_.emplace_back(
        std::make_unique<Option>(std::forward<Args>(args)...));
    return defaults_.GetRef<typename Option::ValueType>(
        options_.back()->GetId());
  }

  // Returns list of options in UCI format.
  std::vector<std::string> ListOptionsUci() const;

  // Set the UCI option from string value.
  void SetUciOption(const std::string& name, const std::string& value,
                    const std::string& context = "");
  // Hide this option from help and UCI.
  void HideOption(const OptionId& id);
  // Processes all flags from the command line and an optional
  // configuration file. Returns false if there is an invalid flag.
  bool ProcessAllFlags();
  // Processes either the command line or configuration file flags.
  bool ProcessFlags(const std::vector<std::string>& args);

  // Get the options dict for given context.
  const OptionsDict& GetOptionsDict(const std::string& context = {});
  // Gets the dictionary for given context which caller can modify.
  OptionsDict* GetMutableOptions(const std::string& context = {});
  // Gets the mutable list of default options.
  OptionsDict* GetMutableDefaultsOptions() { return &defaults_; }
  // Adds a subdictionary for a given context.
  void AddContext(const std::string&);

 private:
  // Prints help to std::cout.
  void ShowHelp() const;
  // Prints markdown formatted help to std::cout.
  void ShowHelpMd() const;
  // Make all hidden options visible.
  void ShowHidden() const;
  // Returns an option based on the long flag.
  Option* FindOptionByLongFlag(const std::string& flag) const;
  // Returns an option based by its uci name.
  Option* FindOptionByUciName(const std::string& name) const;
  // Returns an option based by its id.
  Option* FindOptionById(const std::string& name) const;

  std::vector<std::unique_ptr<Option>> options_;
  OptionsDict defaults_;
  OptionsDict& values_;
};

class StringOption : public OptionsParser::Option {
 public:
  using ValueType = std::string;
  StringOption(const OptionId& id);

  void SetValue(const std::string& value, OptionsDict* dict) override;

 private:
  std::string GetOptionString(const OptionsDict& dict) const override;
  bool ProcessLongFlag(const std::string& flag, const std::string& value,
                       OptionsDict* dict) override;
  std::string GetHelp(const OptionsDict& dict) const override;
  bool ProcessShortFlagWithValue(char flag, const std::string& value,
                                 OptionsDict* dict) override;

  ValueType GetVal(const OptionsDict&) const;
  void SetVal(OptionsDict* dict, const ValueType& val) const;
};

class IntOption : public OptionsParser::Option {
 public:
  using ValueType = int;
  IntOption(const OptionId& id, int min, int max);

  void SetValue(const std::string& value, OptionsDict* dict) override;

 private:
  std::string GetOptionString(const OptionsDict& dict) const override;
  bool ProcessLongFlag(const std::string& flag, const std::string& value,
                       OptionsDict* dict) override;
  std::string GetHelp(const OptionsDict& dict) const override;
  bool ProcessShortFlagWithValue(char flag, const std::string& value,
                                 OptionsDict* dict) override;

  ValueType GetVal(const OptionsDict&) const;
  void SetVal(OptionsDict* dict, const ValueType& val) const;

  int min_;
  int max_;
};

class FloatOption : public OptionsParser::Option {
 public:
  using ValueType = float;
  FloatOption(const OptionId& id, float min, float max);

  void SetValue(const std::string& value, OptionsDict* dict) override;

 private:
  std::string GetOptionString(const OptionsDict& dict) const override;
  bool ProcessLongFlag(const std::string& flag, const std::string& value,
                       OptionsDict* dict) override;
  std::string GetHelp(const OptionsDict& dict) const override;
  bool ProcessShortFlagWithValue(char flag, const std::string& value,
                                 OptionsDict* dict) override;

  ValueType GetVal(const OptionsDict&) const;
  void SetVal(OptionsDict* dict, const ValueType& val) const;

  float min_;
  float max_;
};

class BoolOption : public OptionsParser::Option {
 public:
  using ValueType = bool;
  BoolOption(const OptionId& id);

  void SetValue(const std::string& value, OptionsDict* dict) override;

 private:
  std::string GetOptionString(const OptionsDict& dict) const override;
  bool ProcessLongFlag(const std::string& flag, const std::string& value,
                       OptionsDict* dict) override;
  std::string GetHelp(const OptionsDict& dict) const override;
  bool ProcessShortFlag(char flag, OptionsDict* dict) override;

  ValueType GetVal(const OptionsDict&) const;
  void SetVal(OptionsDict* dict, const ValueType& val) const;
  void ValidateBoolString(const std::string& val);
};

class ChoiceOption : public OptionsParser::Option {
 public:
  using ValueType = std::string;
  ChoiceOption(const OptionId& id, const std::vector<std::string>& choices);

  void SetValue(const std::string& value, OptionsDict* dict) override;

 private:
  std::string GetOptionString(const OptionsDict& dict) const override;
  bool ProcessLongFlag(const std::string& flag, const std::string& value,
                       OptionsDict* dict) override;
  std::string GetHelp(const OptionsDict& dict) const override;
  bool ProcessShortFlagWithValue(char flag, const std::string& value,
                                 OptionsDict* dict) override;

  ValueType GetVal(const OptionsDict&) const;
  void SetVal(OptionsDict* dict, const ValueType& val) const;

  std::vector<std::string> choices_;
};

}  // namespace lczero
