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
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "utils/cascading_dict.h"

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

using OptionsDict =
    CascadingDict<Ref<const OptionId>, bool, Button, int, std::string, float>;
using InlineConfig = CascadingDict<std::string, bool, int, std::string, float>;

void ParseInlineConfig(const std::string& str, InlineConfig* options_dict);

}  // namespace lczero
