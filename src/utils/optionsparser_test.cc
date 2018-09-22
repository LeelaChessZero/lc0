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

#include <gtest/gtest.h>
#include <iostream>
#include "utils/optionsparser.h"

namespace lczero {

TEST(OptionsParser, CheckInvalidOption) {
  OptionsParser options;
  options.Add<StringOption>("this-is-a-valid-option", "this-is-a-valid-option", 'a') = "";

  EXPECT_NO_THROW(options.SetOption("this-is-a-valid-option", "valid-value"));
  EXPECT_THROW(options.SetOption("this-is-an-invalid-option", "0"), Exception);
}

TEST(OptionsParser, IntOptionCheckValueConstraints) {
  OptionsParser options;
  options.Add<IntOption>("int-test-a", 25, 75, "int-test-a", 'a') = 50;

  EXPECT_NO_THROW(options.SetOption("int-test-a", "25"));
  EXPECT_NO_THROW(options.SetOption("int-test-a", "50"));
  EXPECT_NO_THROW(options.SetOption("int-test-a", "75"));
  EXPECT_THROW(options.SetOption("int-test-a", "0"), Exception);
  EXPECT_THROW(options.SetOption("int-test-a", "100"), Exception);
}

TEST(OptionsParser, FloatOptionCheckValueConstraints) {
  OptionsParser options;
  options.Add<FloatOption>("float-test-a", 25.0f, 75.0f, "float-test-a", 'a') = 50.0f;

  EXPECT_NO_THROW(options.SetOption("float-test-a", "25.0"));
  EXPECT_NO_THROW(options.SetOption("float-test-a", "50.0"));
  EXPECT_NO_THROW(options.SetOption("float-test-a", "75.0"));
  EXPECT_THROW(options.SetOption("float-test-a", "0.0"), Exception);
  EXPECT_THROW(options.SetOption("float-test-a", "100.0"), Exception);
}

TEST(OptionsParser, BoolOptionsCheckValueConstraints) {
  OptionsParser options;
  options.Add<BoolOption>("bool-test-a", "bool-test-a", 'a') = false;

  EXPECT_NO_THROW(options.SetOption("bool-test-a", "true"));
  EXPECT_NO_THROW(options.SetOption("bool-test-a", "false"));
  EXPECT_THROW(options.SetOption("bool-test-a", "leela"), Exception);
}

TEST(OptionsParser, ChoiceOptionCheckValueConstraints) {
  OptionsParser options;
  std::vector<std::string> choices;
  choices.push_back("choice-a");
  choices.push_back("choice-b");
  choices.push_back("choice-c");
  options.Add<ChoiceOption>("choice-test-a", choices, "choice-test-a", 'a') = "choice-a";

  EXPECT_NO_THROW(options.SetOption("choice-test-a", "choice-a"));
  EXPECT_NO_THROW(options.SetOption("choice-test-a", "choice-b"));
  EXPECT_NO_THROW(options.SetOption("choice-test-a", "choice-c"));
  EXPECT_THROW(options.SetOption("choice-test-a", "choice-d"), Exception);
}

}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

