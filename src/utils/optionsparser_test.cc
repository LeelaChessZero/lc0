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
      "backend=demux, flag=true, count=42, rate=1.5, path=/tmp/foo, (gpu=0), "
      "(gpu=1)");

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
  for (float value : {0.0f, 1.0f, 1.2345678f, -1.2345678f, 1e-20f, 1e20f}) {
    SCOPED_TRACE(value);
    OptionsDict dict;
    dict.Set<float>("val", value);
    OptionsDict parsed;
    parsed.AddSubdictFromString(dict.FlattenSubdictToString());
    EXPECT_EQ(parsed.Get<float>("val"), value);
  }
}

TEST(OptionsDict, FlattenSubdictToStringQuotesValuesAndNames) {
  for (const std::string value :
       {"", "true", "false", "42", "_foo", "-foo", ".hidden", "123abc", "12-34",
        "a b", "a,b", "a(b)", "a=b", "single'quote", "double\"quote"}) {
    SCOPED_TRACE(value);
    OptionsDict dict;
    dict.Set<std::string>(value, value);
    dict.AddSubdict(value)->Set<int>("id", 7);
    OptionsDict parsed;
    parsed.AddSubdictFromString(dict.FlattenSubdictToString());
    EXPECT_EQ(parsed.Get<std::string>(value), value);
    EXPECT_EQ(parsed.GetSubdict(value).Get<int>("id"), 7);
  }
}

TEST(OptionsDict, FlattenSubdictToStringFiltersOnlyTopLevel) {
  OptionsDict dict;
  dict.AddSubdictFromString(
      "backend=demux, threads=2, omitted(x=1), "
      "child(backend=demux, threads=3, omitted(x=2), "
      "nested(backend=cuda, threads=4, omitted(x=5)))");
  OptionsDict parsed;
  parsed.AddSubdictFromString(
      dict.FlattenSubdictToString("child", {"backend", "omitted"}));
  EXPECT_FALSE(parsed.Exists<std::string>("backend"));
  EXPECT_EQ(parsed.Get<int>("threads"), 3);
  EXPECT_FALSE(parsed.HasSubdict("omitted"));
  const auto& nested = parsed.GetSubdict("nested");
  EXPECT_EQ(nested.Get<std::string>("backend"), "cuda");
  EXPECT_EQ(nested.Get<int>("threads"), 4);
  EXPECT_EQ(nested.GetSubdict("omitted").Get<int>("x"), 5);
}

TEST(OptionsDict, FlattenSubdictToStringFiltersRootBatchStep) {
  OptionsDict dict;
  dict.AddSubdictFromString("batch_step=3, child(batch_step=5), empty()");
  for (const std::string name : {"", "empty", "child"}) {
    SCOPED_TRACE(name);
    OptionsDict parsed;
    parsed.AddSubdictFromString(
        dict.FlattenSubdictToString(name, {}, {"batch_step"}));
    if (name == "child") {
      EXPECT_EQ(parsed.Get<int>("batch_step"), 5);
    } else {
      EXPECT_FALSE(parsed.Exists<int>("batch_step"));
    }
  }
}

TEST(OptionsDict, FlattenSubdictToStringPreservesAnonymousNames) {
  OptionsDict dict;
  for (int i = 0; i < 12; ++i)
    dict.AddSubdictFromString("(id=" + std::to_string(i) + ")");
  OptionsDict parsed;
  parsed.AddSubdictFromString(dict.FlattenSubdictToString("", {"[1]"}));
  EXPECT_FALSE(parsed.HasSubdict("[1]"));
  for (int i = 0; i < 12; ++i) {
    if (i == 1) continue;
    EXPECT_EQ(parsed.GetSubdict("[" + std::to_string(i) + "]").Get<int>("id"),
              i);
  }
}

}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
