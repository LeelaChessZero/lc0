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

TEST(OptionsDict, CloneScalarsDetachesLocalValues) {
  OptionsDict parent;
  parent.Set<int>("inherited", 1);
  OptionsDict source(&parent);
  source.AddSubdictFromString(
      "enabled=true, threads=2, rate=1.5, backend=cuda");
  source.Set<Button>("button", Button(true));
  source.AddSubdict("child");

  auto clone = source.CloneScalars();
  EXPECT_TRUE(clone->Get<bool>("enabled"));
  EXPECT_EQ(clone->Get<int>("threads"), 2);
  EXPECT_EQ(clone->Get<float>("rate"), 1.5f);
  EXPECT_EQ(clone->Get<std::string>("backend"), "cuda");
  EXPECT_FALSE(clone->Exists<int>("inherited"));
  EXPECT_FALSE(clone->Exists<Button>("button"));
  EXPECT_TRUE(clone->ListSubdicts().empty());
  source.Set<int>("threads", 4);
  EXPECT_EQ(clone->Get<int>("threads"), 2);
}

TEST(OptionsDict, MergeAndCopySubdictsReplaceTrees) {
  OptionsDict source;
  source.AddSubdictFromString("threads=2, backend=cuda, child(id=7, nested())");
  OptionsDict dict;
  dict.AddSubdictFromString(
      "threads=4, batch=32, stale(), child(old=1, nested(old=2))");

  dict.MergeFrom(source);
  EXPECT_EQ(dict.Get<int>("threads"), 2);
  EXPECT_EQ(dict.Get<int>("batch"), 32);
  EXPECT_EQ(dict.Get<std::string>("backend"), "cuda");
  EXPECT_TRUE(dict.HasSubdict("stale"));
  const auto& child = dict.GetSubdict("child");
  EXPECT_EQ(child.Get<int>("id"), 7);
  EXPECT_EQ(child.Get<int>("batch"), 32);
  EXPECT_FALSE(child.Exists<int>("old"));
  EXPECT_FALSE(child.GetSubdict("nested").Exists<int>("old"));

  dict.Set<int>("threads", 8);
  dict.GetMutableSubdict("child")->Set<int>("old", 3);
  dict.GetMutableSubdict("child")->GetMutableSubdict("nested")->Set<int>("old",
                                                                         4);
  dict.CopySubdictsFrom(source);
  EXPECT_EQ(dict.Get<int>("threads"), 8);
  EXPECT_EQ(dict.GetSubdict("child").Get<int>("threads"), 8);
  EXPECT_EQ(dict.GetSubdict("child").Get<int>("id"), 7);
  EXPECT_FALSE(dict.GetSubdict("child").Exists<int>("old"));
  EXPECT_FALSE(
      dict.GetSubdict("child").GetSubdict("nested").Exists<int>("old"));
  source.GetMutableSubdict("child")->Set<int>("id", 9);
  EXPECT_EQ(dict.GetSubdict("child").Get<int>("id"), 7);
}

TEST(OptionsDict, RemoveLeavesParentAndUnrelatedValues) {
  OptionsDict parent;
  parent.Set<int>("threads", 2);
  OptionsDict dict(&parent);
  dict.AddSubdictFromString("threads=4, batch=32, threads(), child(threads=8)");

  dict.Remove("threads");
  EXPECT_FALSE(dict.OwnExists<int>("threads"));
  EXPECT_FALSE(dict.HasSubdict("threads"));
  EXPECT_EQ(dict.Get<int>("threads"), 2);
  EXPECT_EQ(parent.Get<int>("threads"), 2);
  EXPECT_EQ(dict.Get<int>("batch"), 32);
  EXPECT_EQ(dict.GetSubdict("child").Get<int>("threads"), 8);
}

TEST(OptionsDict, SerializeRoundTripsScalarsAndSubdict) {
  OptionsDict dict;
  dict.AddSubdictFromString("enabled=true, threads=2, child(device=0)");
  dict.Set<std::string>("display name", "cuda backend");
  dict.Set<float>("rate", 1.2345678f);
  dict.Set<float>("offset", -1.0f);
  OptionsDict parsed;
  parsed.AddSubdictFromString(dict.Serialize());

  EXPECT_TRUE(parsed.Get<bool>("enabled"));
  EXPECT_EQ(parsed.Get<int>("threads"), 2);
  EXPECT_EQ(parsed.Get<std::string>("display name"), "cuda backend");
  EXPECT_EQ(parsed.Get<float>("rate"), 1.2345678f);
  EXPECT_EQ(parsed.Get<float>("offset"), -1.0f);
  EXPECT_EQ(parsed.GetSubdict("child").Get<int>("device"), 0);
}
}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
