/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2020 The LCZero Authors

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

#include "utils/cascading_dict.h"
#include "utils/exception.h"

namespace lczero {

class OptionId {
 public:
  enum VisibilityMode {
    kSimpleMode = 1 << 0,    // Simple mode.
    kNormalMode = 1 << 1,    // Normal mode.
    kProMode = 1 << 2,       // Pro mode.
    kSelfplayMode = 1 << 3,  // Self-play mode.
  };

  enum VisibilityMask {
    kSimpleOnly = kSimpleMode,
    kDefaultVisibility = kNormalMode | kProMode | kSelfplayMode,
    kProModeMask = kProMode | kSelfplayMode,
    kAlwaysVisible = kSimpleMode | kNormalMode | kProMode | kSelfplayMode,
  };

  struct OptionsParams {
    const char* long_flag = nullptr;
    const char* uci_option = nullptr;
    const char* help_text = nullptr;
    char short_flag = '\0';
    VisibilityMask visibility_mask = kDefaultVisibility;
  };

  OptionId(const OptionsParams& params)
      : long_flag_(params.long_flag),
        uci_option_(params.uci_option),
        help_text_(params.help_text),
        short_flag_(params.short_flag),
        visibility_mask_(params.visibility_mask) {}

  OptionId(const char* long_flag, const char* uci_option, const char* help_text,
           const char short_flag = '\0')
      : long_flag_(long_flag),
        uci_option_(uci_option),
        help_text_(help_text),
        short_flag_(short_flag) {}

  OptionId(const OptionId& other) = delete;
  bool operator==(const OptionId& other) const { return this == &other; }

  const char* long_flag() const { return long_flag_; }
  const char* uci_option() const { return uci_option_; }
  const char* help_text() const { return help_text_; }
  char short_flag() const { return short_flag_; }
  uint64_t visibility_mask() const { return visibility_mask_; }

  friend std::ostream& operator<<(std::ostream& os, const OptionId& id) {
    os << "OptionId [";
    if (id.long_flag_ && *id.long_flag_) os << "--" << id.long_flag_;
    if (id.short_flag_) os << " -" << id.short_flag_;
    if (id.uci_option_ && *id.uci_option_) os << " uci:" << id.uci_option_;
    os << "]";
    return os;
  }

 private:
  const char* const long_flag_;
  const char* const uci_option_;
  const char* const help_text_;
  const char short_flag_;
  uint64_t visibility_mask_ = kDefaultVisibility;
};

class Button {
 public:
  Button() { val = std::make_shared<bool>(false); }
  Button(bool x) { val = std::make_shared<bool>(x); }
  bool TestAndReset() {
    bool r = *val;
    *val = false;
    return r;
  }

 private:
  std::shared_ptr<bool> val;
};

template <typename T>
struct Ref : public std::reference_wrapper<T> {
  using std::reference_wrapper<T>::reference_wrapper;
  template <typename H>
  friend H AbslHashValue(H h, const Ref& c) {
    return H::combine(std::move(h), &c.get());
  }
  bool operator==(const Ref& other) const {
    return &this->get() == &other.get();
  }
};

using ProgramOptions =
    CascadingDict<Ref<const OptionId>, bool, Button, int, std::string, float>;

class ProgramOptionsManager {
 public:
  ProgramOptionsManager();

  class Option {
   public:
    Option(const OptionId& id);
    virtual ~Option() {};
    // Set value from string.
    virtual void SetValue(const std::string& value, ProgramOptions* dict) = 0;

   protected:
    const OptionId& GetId() const { return id_; }
    std::string GetUciOption() const {
      return id_.uci_option() ? id_.uci_option() : "";
    }
    std::string GetHelpText() const { return id_.help_text(); }
    std::string GetLongFlag() const {
      return id_.long_flag() ? id_.long_flag() : "";
    }
    char GetShortFlag() const { return id_.short_flag(); }

   private:
    virtual std::string GetOptionString(const ProgramOptions& dict) const = 0;
    virtual bool ProcessLongFlag(const std::string& /*flag*/,
                                 const std::string& /*value*/,
                                 ProgramOptions* /*dict*/) {
      return false;
    }
    virtual bool ProcessShortFlag(char /*flag*/, ProgramOptions* /*dict*/) {
      return false;
    }
    virtual bool ProcessShortFlagWithValue(char /*flag*/,
                                           const std::string& /*value*/,
                                           ProgramOptions* /*dict*/) {
      return false;
    }
    virtual std::string GetHelp(const ProgramOptions& dict) const = 0;

    const OptionId& id_;
    friend class ProgramOptionsManager;
  };

  // Add an option to the list of available options (from command line flags
  // or UCI params)
  // Usage:
  // options->Add<StringOption>(name, func, long_flag, short_flag) = def_val;
  template <typename Option, typename... Args>
  typename Option::ValueType& Add(Args&&... args) {
    options_.emplace_back(
        std::make_unique<Option>(std::forward<Args>(args)...));
    return defaults_.GetOwnRef<typename Option::ValueType>(
        options_.back()->GetId());
  }

  // Returns list of options in UCI format.
  std::vector<std::string> ListOptionsUci() const;

  // Set the UCI option from string value.
  void SetUciOption(const std::string& name, const std::string& value,
                    const std::string& context = "");
  // Processes all flags from the command line and an optional
  // configuration file. Returns false if there is an invalid flag.
  bool ProcessAllFlags();
  // Processes either the command line or configuration file flags.
  bool ProcessFlags(const std::vector<std::string>& args);

  // Get the options dict for given context.
  const ProgramOptions& GetOptionsDict(const std::string& context = {});
  // Gets the dictionary for given context which caller can modify.
  ProgramOptions* GetMutableOptions(const std::string& context = {});
  // Gets the mutable list of default options.
  ProgramOptions* GetMutableDefaultsOptions() { return &defaults_; }
  // Adds a subdictionary for a given context.
  void AddContext(const std::string&);
  // Prints help to std::cout.
  void ShowHelp() const;

 private:
  // Returns an option based on the long flag.
  Option* FindOptionByLongFlag(const std::string& flag) const;
  // Returns an option based by its uci name.
  Option* FindOptionByUciName(const std::string& name) const;
  // Returns an option based by its id.
  Option* FindOptionById(const OptionId& id) const;

  std::vector<std::unique_ptr<Option>> options_;
  ProgramOptions defaults_;
  ProgramOptions& values_;
  OptionId::VisibilityMode visibility_mode_ = OptionId::kNormalMode;
};

class StringOption : public ProgramOptionsManager::Option {
 public:
  using ValueType = std::string;
  StringOption(const OptionId& id);

  void SetValue(const std::string& value, ProgramOptions* dict) override;

 private:
  std::string GetOptionString(const ProgramOptions& dict) const override;
  bool ProcessLongFlag(const std::string& flag, const std::string& value,
                       ProgramOptions* dict) override;
  std::string GetHelp(const ProgramOptions& dict) const override;
  bool ProcessShortFlagWithValue(char flag, const std::string& value,
                                 ProgramOptions* dict) override;

  ValueType GetVal(const ProgramOptions&) const;
  void SetVal(ProgramOptions* dict, const ValueType& val) const;
};

class IntOption : public ProgramOptionsManager::Option {
 public:
  using ValueType = int;
  IntOption(const OptionId& id, int min, int max);

  void SetValue(const std::string& value, ProgramOptions* dict) override;

 private:
  std::string GetOptionString(const ProgramOptions& dict) const override;
  bool ProcessLongFlag(const std::string& flag, const std::string& value,
                       ProgramOptions* dict) override;
  std::string GetHelp(const ProgramOptions& dict) const override;
  bool ProcessShortFlagWithValue(char flag, const std::string& value,
                                 ProgramOptions* dict) override;

  ValueType GetVal(const ProgramOptions&) const;
  void SetVal(ProgramOptions* dict, const ValueType& val) const;
  int ValidateIntString(const std::string& val) const;

  int min_;
  int max_;
};

class FloatOption : public ProgramOptionsManager::Option {
 public:
  using ValueType = float;
  FloatOption(const OptionId& id, float min, float max);

  void SetValue(const std::string& value, ProgramOptions* dict) override;

 private:
  std::string GetOptionString(const ProgramOptions& dict) const override;
  bool ProcessLongFlag(const std::string& flag, const std::string& value,
                       ProgramOptions* dict) override;
  std::string GetHelp(const ProgramOptions& dict) const override;
  bool ProcessShortFlagWithValue(char flag, const std::string& value,
                                 ProgramOptions* dict) override;

  ValueType GetVal(const ProgramOptions&) const;
  void SetVal(ProgramOptions* dict, const ValueType& val) const;

  float min_;
  float max_;
};

class BoolOption : public ProgramOptionsManager::Option {
 public:
  using ValueType = bool;
  BoolOption(const OptionId& id);

  void SetValue(const std::string& value, ProgramOptions* dict) override;

 private:
  std::string GetOptionString(const ProgramOptions& dict) const override;
  bool ProcessLongFlag(const std::string& flag, const std::string& value,
                       ProgramOptions* dict) override;
  std::string GetHelp(const ProgramOptions& dict) const override;
  bool ProcessShortFlag(char flag, ProgramOptions* dict) override;

  ValueType GetVal(const ProgramOptions&) const;
  void SetVal(ProgramOptions* dict, const ValueType& val) const;
  void ValidateBoolString(const std::string& val);
};

class ButtonOption : public ProgramOptionsManager::Option {
 public:
  using ValueType = Button;
  ButtonOption(const OptionId& id);

  void SetValue(const std::string& value, ProgramOptions* dict) override;

 private:
  std::string GetOptionString(const ProgramOptions& dict) const override;
  bool ProcessLongFlag(const std::string& flag, const std::string& value,
                       ProgramOptions* dict) override;
  std::string GetHelp(const ProgramOptions& dict) const override;
  bool ProcessShortFlag(char flag, ProgramOptions* dict) override;

  ValueType GetVal(ProgramOptions*) const;
  void SetVal(ProgramOptions* dict, const ValueType& val) const;
};

class ChoiceOption : public ProgramOptionsManager::Option {
 public:
  using ValueType = std::string;
  ChoiceOption(const OptionId& id, const std::vector<std::string>& choices);

  void SetValue(const std::string& value, ProgramOptions* dict) override;

 private:
  std::string GetOptionString(const ProgramOptions& dict) const override;
  bool ProcessLongFlag(const std::string& flag, const std::string& value,
                       ProgramOptions* dict) override;
  std::string GetHelp(const ProgramOptions& dict) const override;
  bool ProcessShortFlagWithValue(char flag, const std::string& value,
                                 ProgramOptions* dict) override;

  ValueType GetVal(const ProgramOptions&) const;
  void SetVal(ProgramOptions* dict, const ValueType& val) const;

  std::vector<std::string> choices_;
};

}  // namespace lczero
