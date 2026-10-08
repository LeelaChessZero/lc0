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

#include "utils/optionsparser.h"
#include <gtest/gtest.h>
#include <iostream>

namespace lczero {

TEST(OptionsParser, CheckInvalidOption) {
  OptionsParser options;
  const OptionId id{"this-is-a-valid-option", "this-is-a-valid-option", "help",
                    'a'};
  options.Add<StringOption>(id) = "";
  EXPECT_NO_THROW(
      options.SetUciOption("this-is-a-valid-option", "valid-value"));
  EXPECT_THROW(options.SetUciOption("this-is-an-invalid-option", "0"),
               Exception);
}

TEST(OptionsParser, IntOptionCheckValueConstraints) {
  OptionsParser options;
  const OptionId id{"int-test-a", "int-test-a", "help", 'a'};
  options.Add<IntOption>(id, 25, 75) = 50;

  EXPECT_NO_THROW(options.SetUciOption("int-test-a", "25"));
  EXPECT_NO_THROW(options.SetUciOption("int-test-a", "50"));
  EXPECT_NO_THROW(options.SetUciOption("int-test-a", "75"));
  EXPECT_THROW(options.SetUciOption("int-test-a", "0"), Exception);
  EXPECT_THROW(options.SetUciOption("int-test-a", "100"), Exception);
}

TEST(OptionsParser, FloatOptionCheckValueConstraints) {
  OptionsParser options;
  const OptionId id{"float-test-a", "float-test-a", "help", 'a'};
  options.Add<FloatOption>(id, 25.0f, 75.0f) = 50.0f;

  EXPECT_NO_THROW(options.SetUciOption("float-test-a", "25.0"));
  EXPECT_NO_THROW(options.SetUciOption("float-test-a", "50.0"));
  EXPECT_NO_THROW(options.SetUciOption("float-test-a", "75.0"));
  EXPECT_THROW(options.SetUciOption("float-test-a", "0.0"), Exception);
  EXPECT_THROW(options.SetUciOption("float-test-a", "100.0"), Exception);
}

TEST(OptionsParser, BoolOptionsCheckValueConstraints) {
  OptionsParser options;
  const OptionId id{"bool-test-a", "bool-test-a", "help", 'a'};
  options.Add<BoolOption>(id) = false;

  EXPECT_NO_THROW(options.SetUciOption("bool-test-a", "true"));
  EXPECT_NO_THROW(options.SetUciOption("bool-test-a", "false"));
  EXPECT_THROW(options.SetUciOption("bool-test-a", "leela"), Exception);
}

TEST(OptionsParser, ChoiceOptionCheckValueConstraints) {
  OptionsParser options;
  const OptionId id{"choice-test-a", "choice-test-a", "help", 'a'};
  std::vector<std::string> choices;
  choices.push_back("choice-a");
  choices.push_back("choice-b");
  choices.push_back("choice-c");
  options.Add<ChoiceOption>(id, choices) = "choice-a";

  EXPECT_NO_THROW(options.SetUciOption("choice-test-a", "choice-a"));
  EXPECT_NO_THROW(options.SetUciOption("choice-test-a", "choice-b"));
  EXPECT_NO_THROW(options.SetUciOption("choice-test-a", "choice-c"));
  EXPECT_THROW(options.SetUciOption("choice-test-a", "choice-d"), Exception);
}

TEST(OptionsDict, FlattenSubdictToStringInheritanceAndOverride) {
  OptionsDict dict;
  dict.AddSubdictFromString("a=b, c=d, x(c=e, f=g, y(h=i))");

  // Flatten subdict x
  std::string s = dict.FlattenSubdictToString("x");

  // Parse back to verify
  OptionsDict parsed;
  parsed.AddSubdictFromString(s);

  EXPECT_EQ(parsed.Get<std::string>("a"), "b");
  EXPECT_EQ(parsed.Get<std::string>("c"), "e");
  EXPECT_EQ(parsed.Get<std::string>("f"), "g");
  EXPECT_TRUE(parsed.HasSubdict("y"));
  EXPECT_EQ(parsed.GetSubdict("y").Get<std::string>("h"), "i");
  EXPECT_FALSE(parsed.HasSubdict("x"));
}

TEST(OptionsDict, FlattenSubdictToStringTypesAndIgnoreKeys) {
  OptionsDict dict;
  dict.AddSubdictFromString(
      "backend=demux, flag=true, count=42, rate=1.5, path=/tmp/foo, (gpu=0), (gpu=1)");

  // Flatten anonymous subdict [0], ignore "backend"
  std::string s0 = dict.FlattenSubdictToString("[0]", {"backend"});
  OptionsDict parsed0;
  parsed0.AddSubdictFromString(s0);

  EXPECT_FALSE(parsed0.Exists<std::string>("backend"));
  EXPECT_EQ(parsed0.Get<bool>("flag"), true);
  EXPECT_EQ(parsed0.Get<int>("count"), 42);
  EXPECT_FLOAT_EQ(parsed0.Get<float>("rate"), 1.5f);
  EXPECT_EQ(parsed0.Get<std::string>("path"), "/tmp/foo");
  EXPECT_EQ(parsed0.Get<int>("gpu"), 0);

  // Flatten anonymous subdict [1], ignore "backend"
  std::string s1 = dict.FlattenSubdictToString("[1]", {"backend"});
  OptionsDict parsed1;
  parsed1.AddSubdictFromString(s1);

  EXPECT_FALSE(parsed1.Exists<std::string>("backend"));
  EXPECT_EQ(parsed1.Get<int>("gpu"), 1);
}

TEST(OptionsDict, FlattenSubdictToStringFloatExactness) {
  OptionsDict dict;
  dict.AddSubdictFromString("val=1.0, zero=0.0");
  std::string s = dict.FlattenSubdictToString();

  OptionsDict parsed;
  parsed.AddSubdictFromString(s);

  EXPECT_TRUE(parsed.Exists<float>("val"));
  EXPECT_FLOAT_EQ(parsed.Get<float>("val"), 1.0f);
  EXPECT_TRUE(parsed.Exists<float>("zero"));
  EXPECT_FLOAT_EQ(parsed.Get<float>("zero"), 0.0f);
}

}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
