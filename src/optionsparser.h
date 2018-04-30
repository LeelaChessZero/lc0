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

#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "utils/exception.h"
#include "utils/optionsdict.h"

namespace lczero {

class OptionsParser {
 public:
  OptionsParser();

  class Option {
   public:
    Option(const std::string& name, const std::string& long_flag,
           char short_flag);
    virtual ~Option(){};
    // Set value from string.
    virtual void SetValue(const std::string& value, OptionsDict* dict) = 0;

   protected:
    virtual const std::string& GetName() const { return name_; }
    const std::string& GetLongFlag() const { return long_flag_; }
    char GetShortFlag() const { return short_flag_; }

   private:
    virtual std::string GetOptionString(const OptionsDict& dict) const = 0;
    virtual void SendValue(const OptionsDict& dict) const = 0;
    virtual bool ProcessLongFlag(const std::string& flag,
                                 const std::string& value, OptionsDict* dict) {
      return false;
    }
    virtual bool ProcessShortFlag(char flag, OptionsDict* dict) {
      return false;
    }
    virtual bool ProcessShortFlagWithValue(char flag, const std::string& value,
                                           OptionsDict* dict) {
      return false;
    }
    virtual std::string GetHelp(const OptionsDict& dict) const = 0;

    std::string name_;
    std::string long_flag_;
    char short_flag_;
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
        options_.back()->GetName());
  }

  // Returns list of options in UCI format.
  std::vector<std::string> ListOptionsUci() const;

  // Set the option from string value.
  void SetOption(const std::string& name, const std::string& value,
                 const std::string& context = "");
  // Call option setter for this option.
  void SendOption(const std::string& name);
  // Call option setter all options.
  void SendAllOptions();
  // Processes all flags. Returns false if should exit.
  bool ProcessAllFlags();

  // Get the options dict for given context.
  const OptionsDict& GetOptionsDict(const std::string& context = {});
  // Gets the dictionary for given context which caller can modify.
  OptionsDict* GetMutableOptions(const std::string& context = {});
  // Adds a subdictionary for a given context.
  void AddContext(const std::string&);

 private:
  void ShowHelp() const;

  Option* FindOptionByName(const std::string& name) const;
  std::vector<std::unique_ptr<Option>> options_;

  OptionsDict defaults_;
  OptionsDict& values_;
};

class StringOption : public OptionsParser::Option {
 public:
  using ValueType = std::string;
  StringOption(const std::string& name, const std::string& long_flag = {},
               char short_flag = '\0',
               std::function<void(const std::string&)> setter = {});

  void SetValue(const std::string& value, OptionsDict* dict) override;

 private:
  std::string GetOptionString(const OptionsDict& dict) const override;
  void SendValue(const OptionsDict& dict) const override;
  bool ProcessLongFlag(const std::string& flag, const std::string& value,
                       OptionsDict* dict) override;
  std::string GetHelp(const OptionsDict& dict) const override;
  bool ProcessShortFlagWithValue(char flag, const std::string& value,
                                 OptionsDict* dict) override;

  ValueType GetVal(const OptionsDict&) const;
  void SetVal(OptionsDict* dict, const ValueType& val) const;

  std::function<void(const std::string&)> setter_;
};

class IntOption : public OptionsParser::Option {
 public:
  using ValueType = int;
  IntOption(const std::string& name, int min, int max,
            const std::string& long_flag = {}, char short_flag = '\0',
            std::function<void(int)> setter = {});

  void SetValue(const std::string& value, OptionsDict* dict) override;

 private:
  std::string GetOptionString(const OptionsDict& dict) const override;
  void SendValue(const OptionsDict& dict) const override;
  bool ProcessLongFlag(const std::string& flag, const std::string& value,
                       OptionsDict* dict) override;
  std::string GetHelp(const OptionsDict& dict) const override;
  bool ProcessShortFlagWithValue(char flag, const std::string& value,
                                 OptionsDict* dict) override;

  ValueType GetVal(const OptionsDict&) const;
  void SetVal(OptionsDict* dict, const ValueType& val) const;

  int min_;
  int max_;
  std::function<void(int)> setter_;
};

class FloatOption : public OptionsParser::Option {
 public:
  using ValueType = float;
  FloatOption(const std::string& name, float min, float max,
              const std::string& long_flag = {}, char short_flag = '\0',
              std::function<void(float)> setter = {});

  void SetValue(const std::string& value, OptionsDict* dict) override;

 private:
  std::string GetOptionString(const OptionsDict& dict) const override;
  void SendValue(const OptionsDict& dict) const override;
  bool ProcessLongFlag(const std::string& flag, const std::string& value,
                       OptionsDict* dict) override;
  std::string GetHelp(const OptionsDict& dict) const override;
  bool ProcessShortFlagWithValue(char flag, const std::string& value,
                                 OptionsDict* dict) override;

  ValueType GetVal(const OptionsDict&) const;
  void SetVal(OptionsDict* dict, const ValueType& val) const;

  float min_;
  float max_;
  std::function<void(int)> setter_;
};

class BoolOption : public OptionsParser::Option {
 public:
  using ValueType = bool;
  BoolOption(const std::string& name, const std::string& long_flag = {},
             char short_flag = '\0', std::function<void(bool)> setter = {});

  void SetValue(const std::string& value, OptionsDict* dict) override;

 private:
  std::string GetOptionString(const OptionsDict& dict) const override;
  void SendValue(const OptionsDict& dict) const override;
  bool ProcessLongFlag(const std::string& flag, const std::string& value,
                       OptionsDict* dict) override;
  std::string GetHelp(const OptionsDict& dict) const override;
  bool ProcessShortFlagWithValue(char flag, const std::string& value,
                                 OptionsDict* dict) override;

  ValueType GetVal(const OptionsDict&) const;
  void SetVal(OptionsDict* dict, const ValueType& val) const;

  std::function<void(bool)> setter_;
};

class ChoiceOption : public OptionsParser::Option {
 public:
  using ValueType = std::string;
  ChoiceOption(const std::string& name, const std::vector<std::string>& choices,
               const std::string& long_flag = {}, char short_flag = '\0',
               std::function<void(const std::string&)> setter = {});

  void SetValue(const std::string& value, OptionsDict* dict) override;

 private:
  std::string GetOptionString(const OptionsDict& dict) const override;
  void SendValue(const OptionsDict& dict) const override;
  bool ProcessLongFlag(const std::string& flag, const std::string& value,
                       OptionsDict* dict) override;
  std::string GetHelp(const OptionsDict& dict) const override;
  bool ProcessShortFlagWithValue(char flag, const std::string& value,
                                 OptionsDict* dict) override;

  ValueType GetVal(const OptionsDict&) const;
  void SetVal(OptionsDict* dict, const ValueType& val) const;

  std::function<void(const std::string&)> setter_;
  std::vector<std::string> choices_;
};

}  // namespace lczero