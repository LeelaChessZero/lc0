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

#include "utils/optionsdict.h"
#include <cassert>
#include <cctype>
#include <sstream>
#include <string>
#include "utils/exception.h"

namespace lczero {

const OptionsDict& OptionsDict::GetSubdict(const std::string& name) const {
  auto iter = subdicts_.find(name);
  if (iter == subdicts_.end())
    throw Exception("Subdictionary not found: " + name);
  return iter->second;
}

// Returns subdictionary. Throws exception if doesn't exist.
OptionsDict* OptionsDict::GetMutableSubdict(const std::string& name) {
  auto iter = subdicts_.find(name);
  if (iter == subdicts_.end())
    throw Exception("Subdictionary not found: " + name);
  return &iter->second;
}

// Creates subdictionary. Throws exception if already exists.
OptionsDict* OptionsDict::AddSubdict(const std::string& name) {
  auto iter = subdicts_.find(name);
  if (iter != subdicts_.end())
    throw Exception("Subdictionary already exists: " + name);
  auto x = &subdicts_.emplace(name, this).first->second;
  return x;
}

// Returns list of subdictionaries.
std::vector<std::string> OptionsDict::ListSubdicts() const {
  std::vector<std::string> result;
  for (const auto& subdict : subdicts_) {
    result.emplace_back(subdict.first);
  }
  return result;
}

bool OptionsDict::HasSubdict(const std::string& name) const {
  return subdicts_.find(name) != subdicts_.end();
}

namespace {

class Lexer {
 public:
  enum TokenType {
    L_INTEGER,
    L_FLOAT,
    L_STRING,
    L_IDENTIFIER,
    L_LEFT_PARENTHESIS,
    L_RIGHT_PARENTHESIS,
    L_COMMA,
    L_EQUAL,
    L_EOF
  };

  Lexer(const std::string& str) : str_(str) { Next(); }

  void Next() {
    // Skip whitespace:
    while (idx_ < str_.size() && std::isspace(str_[idx_])) ++idx_;
    last_offset_ = idx_;

    // If end of line, report end of line.
    if (idx_ == str_.size()) {
      type_ = L_EOF;
      return;
    }

    // Single characters.
    static const std::pair<char, TokenType> kCharacters[] = {
        {',', L_COMMA},
        {'(', L_LEFT_PARENTHESIS},
        {')', L_RIGHT_PARENTHESIS},
        {'=', L_EQUAL}};
    for (const auto& ch : kCharacters) {
      if (str_[idx_] == ch.first) {
        ++idx_;
        type_ = ch.second;
        return;
      }
    }

    // Numbers (integer of float).
    static const std::string kNumberChars = "0123456789-.";
    if (kNumberChars.find(str_[idx_]) != std::string::npos) {
      ReadNumber();
      return;
    }

    // Strings (single or double quoted)
    if (str_[idx_] == '\'' || str_[idx_] == '\"') {
      ReadString();
      return;
    }

    // Identifier
    if (std::isalnum(str_[idx_])) {
      ReadIdentifier();
      return;
    }

    RaiseError("Unable to parse token");
  }

  void RaiseError(const std::string& message) {
    throw Exception("Unable to parse config at offset " +
                    std::to_string(last_offset_) + ": " + str_ + " (" +
                    message + ")");
  }

  TokenType GetToken() const { return type_; }
  const std::string& GetStringVal() const { return string_val_; }
  int GetIntVal() const { return int_val_; }
  float GetFloatVal() const { return float_val_; }

 private:
  void ReadString() {
    last_offset_ = idx_;
    const char quote = str_[idx_++];

    for (; idx_ < str_.size(); ++idx_) {
      if (str_[idx_] == quote) {
        type_ = L_STRING;
        string_val_ = str_.substr(last_offset_ + 1, idx_ - last_offset_ - 1);
        ++idx_;
        return;
      }
    }

    last_offset_ = idx_;
    RaiseError("String is not closed at end of line");
  }

  void ReadIdentifier() {
    string_val_ = "";
    type_ = L_IDENTIFIER;
    for (; idx_ < str_.size(); ++idx_) {
      if (!std::isalnum(str_[idx_]) && str_[idx_] != '_') break;
      string_val_ += str_[idx_];
    }
  }

  void ReadNumber() {
    last_offset_ = idx_;
    bool is_float = false;
    static const std::string kFloatChars = ".eE";
    static const std::string kAllowedChars = "+-1234567890.eExX";
    for (; idx_ < str_.size(); ++idx_) {
      if (kAllowedChars.find(str_[idx_]) == std::string::npos) break;
      if (kFloatChars.find(str_[idx_]) != std::string::npos) is_float = true;
    }

    try {
      if (is_float) {
        type_ = L_FLOAT;
        float_val_ = stof(str_.substr(last_offset_, idx_ - last_offset_));

      } else {
        type_ = L_INTEGER;
        int_val_ = stoi(str_.substr(last_offset_, idx_ - last_offset_));
      }

    } catch (...) {
      RaiseError("Unable to parse number");
    }
  }

  float float_val_;
  int int_val_;
  std::string string_val_;
  TokenType type_;
  const std::string str_;
  size_t idx_ = 0;
  int last_offset_ = 0;
};

class Parser {
 public:
  Parser(const std::string& str) : lexer_(str) {}

  void ParseMain(OptionsDict* dict) {
    ParseList(dict);            // Parse list of options
    EnsureToken(Lexer::L_EOF);  // Check that everything is read.
  }

 private:
  // Returns first non-existing subdict with name like "[0]", "[24]", etc.
  static std::string GetFreeSubdictName(OptionsDict* dict) {
    for (int idx = 0;; ++idx) {
      std::string id = "[" + std::to_string(idx) + "]";
      if (!dict->HasSubdict(id)) return id;
    }
    assert(false);
    return "";
  }

  // Parses comma separated list of either:
  // * key=value, or
  // * subdict(comma separated list)
  // Note that in subdict all parts are optional:
  // * (comma separated list) -- name will be synthesized (e.g. "[1]")
  // * subdict() -- empty list
  // * subdict -- the same.
  void ParseList(OptionsDict* dict) {
    while (true) {
      std::string identifier;
      if (lexer_.GetToken() == Lexer::L_LEFT_PARENTHESIS) {
        // List entry starts with "(", that's a special case of subdict without
        // name, we have to come up with the name ourselves.
        identifier = GetFreeSubdictName(dict);
      } else if (lexer_.GetToken() == Lexer::L_IDENTIFIER) {
        // Read identifier.
        identifier = lexer_.GetStringVal();
        lexer_.Next();
      } else {
        // Unexpected token, exiting parsing list.
        return;
      }
      // If there is "=" after identifier, that's key=value entry, read value.
      if (lexer_.GetToken() == Lexer::L_EQUAL) {
        lexer_.Next();
        ReadVal(dict, identifier);
      } else {
        // Otherwise it's subdict.
        ReadSubDict(dict, identifier);
      }
      // If next val is not comma, end of the list.
      if (lexer_.GetToken() != Lexer::L_COMMA) return;
      lexer_.Next();
    }
  }

  void EnsureToken(Lexer::TokenType type) {
    if (lexer_.GetToken() != type)
      lexer_.RaiseError("Expected token #" + std::to_string(type));
  }

  void ReadVal(OptionsDict* dict, const std::string& id) {
    if (lexer_.GetToken() == Lexer::L_FLOAT) {
      dict->Set<float>(id, lexer_.GetFloatVal());
    } else if (lexer_.GetToken() == Lexer::L_INTEGER) {
      dict->Set<int>(id, lexer_.GetIntVal());
    } else if (lexer_.GetToken() == Lexer::L_STRING) {
      // Strings may be:
      // * Single quoted: 'asdf'
      // * Double quoted: "asdf"
      // * Without quotes, if only alphanumeric and not "true" or "false".
      dict->Set<std::string>(id, lexer_.GetStringVal());
    } else if (lexer_.GetToken() == Lexer::L_IDENTIFIER) {
      if (lexer_.GetStringVal() == "true") {
        dict->Set<bool>(id, true);
      } else if (lexer_.GetStringVal() == "false") {
        dict->Set<bool>(id, false);
      } else {
        dict->Set<std::string>(id, lexer_.GetStringVal());
      }
    } else {
      lexer_.RaiseError("Expected value");
    }
    lexer_.Next();
  }

  void ReadSubDict(OptionsDict* dict, const std::string& identifier) {
    OptionsDict* new_dict = dict->AddSubdict(identifier);
    // If opening parentheses, read list of a subdict, otherwise list is empty,
    // so return immediately.
    if (lexer_.GetToken() == Lexer::L_LEFT_PARENTHESIS) {
      lexer_.Next();
      ParseList(new_dict);
      EnsureToken(Lexer::L_RIGHT_PARENTHESIS);
      lexer_.Next();
    }
  }

 private:
  Lexer lexer_;
};

}  // namespace

OptionsDict OptionsDict::FromString(const std::string& str,
                                    const OptionsDict* parent) {
  OptionsDict dict(parent);
  Parser parser(str);
  parser.ParseMain(&dict);
  return dict;
}

}  // namespace lczero
