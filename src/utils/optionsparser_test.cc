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
#include <string_view>

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

TEST(OptionsDict, CloneScalarsPreservesTypesAndDetaches) {
  OptionsDict parent;
  parent.Set<int>("parent", 1);
  OptionsDict alias;
  alias.Set<int>("alias", 2);
  OptionsDict dict(&parent);
  dict.AddAliasDict(&alias);
  dict.Set<bool>("value", true);
  dict.Set<int>("value", 42);
  dict.Set<float>("value", 1.5f);
  dict.Set<std::string>("value", "text");
  dict.Set<Button>("button", Button(true));
  dict.AddSubdict("child");

  auto clone = dict.CloneScalars();
  EXPECT_TRUE(clone->Get<bool>("value"));
  EXPECT_EQ(clone->Get<int>("value"), 42);
  EXPECT_EQ(clone->Get<float>("value"), 1.5f);
  EXPECT_EQ(clone->Get<std::string>("value"), "text");
  EXPECT_FALSE(clone->Exists<int>("parent"));
  EXPECT_FALSE(clone->Exists<int>("alias"));
  EXPECT_FALSE(clone->Exists<Button>("button"));
  EXPECT_TRUE(clone->ListSubdicts().empty());
  dict.Set<int>("value", 99);
  EXPECT_EQ(clone->Get<int>("value"), 42);
}

TEST(OptionsDict, MergeFromOverridesByNameAndReplacesTree) {
  OptionsDict dict;
  dict.AddSubdictFromString("a=b, c=d, stale(), y(old=1, nested(old=2))");
  dict.Set<bool>("c", true);
  dict.Set<int>("c", 42);
  dict.Set<float>("c", 1.5f);
  dict.Set<Button>("c", Button(true));
  dict.Set<float>("multi", 9.5f);
  dict.Set<Button>("multi", Button(true));
  OptionsDict source;
  source.AddSubdictFromString("c=e, f=g, y(h=i, nested())");
  source.Set<bool>("multi", true);
  source.Set<int>("multi", 7);
  source.Set<float>("multi", 1.5f);
  source.Set<std::string>("multi", "text");

  dict.MergeFrom(source);
  EXPECT_EQ(dict.Get<std::string>("a"), "b");
  EXPECT_EQ(dict.Get<std::string>("c"), "e");
  EXPECT_FALSE(dict.Exists<bool>("c"));
  EXPECT_FALSE(dict.Exists<int>("c"));
  EXPECT_FALSE(dict.Exists<float>("c"));
  EXPECT_FALSE(dict.Exists<Button>("c"));
  EXPECT_EQ(dict.Get<std::string>("f"), "g");
  EXPECT_TRUE(dict.Get<bool>("multi"));
  EXPECT_EQ(dict.Get<int>("multi"), 7);
  EXPECT_EQ(dict.Get<float>("multi"), 1.5f);
  EXPECT_EQ(dict.Get<std::string>("multi"), "text");
  EXPECT_FALSE(dict.Exists<Button>("multi"));
  EXPECT_TRUE(dict.HasSubdict("stale"));
  const auto& child = dict.GetSubdict("y");
  const auto& nested = child.GetSubdict("nested");
  EXPECT_FALSE(child.Exists<int>("old"));
  EXPECT_FALSE(nested.Exists<int>("old"));
  EXPECT_NE(&child, &source.GetSubdict("y"));
  EXPECT_NE(&nested, &source.GetSubdict("y").GetSubdict("nested"));
  EXPECT_EQ(child.Get<std::string>("h"), "i");
  dict.Set<std::string>("a", "updated");
  EXPECT_EQ(child.Get<std::string>("a"), "updated");
  EXPECT_EQ(nested.Get<std::string>("a"), "updated");
  source.GetMutableSubdict("y")->Set<std::string>("h", "changed");
  EXPECT_EQ(child.Get<std::string>("h"), "i");
  dict.MergeFrom(dict);
  EXPECT_EQ(&dict.GetSubdict("y"), &child);
}

TEST(OptionsDict, CopySubdictsFromPreservesScalarsAndClonesTree) {
  OptionsDict parent;
  parent.Set<int>("parent", 1);
  OptionsDict alias;
  alias.Set<int>("alias", 2);
  OptionsDict source(&parent);
  source.AddAliasDict(&alias);
  source.AddSubdictFromString("threads=2, child(id=7, nested())");
  source.GetMutableSubdict("child")->Set<Button>("button", Button(true));
  OptionsDict dict;
  dict.AddSubdictFromString(
      "threads=3, stale(), child(old=1, removed(), nested(old=2))");

  dict.CopySubdictsFrom(source);
  EXPECT_EQ(dict.Get<int>("threads"), 3);
  EXPECT_TRUE(dict.HasSubdict("stale"));
  const auto& child = dict.GetSubdict("child");
  const auto& nested = child.GetSubdict("nested");
  EXPECT_FALSE(child.Exists<int>("old"));
  EXPECT_FALSE(child.HasSubdict("removed"));
  EXPECT_FALSE(nested.Exists<int>("old"));
  EXPECT_NE(&child, &source.GetSubdict("child"));
  EXPECT_NE(&nested, &source.GetSubdict("child").GetSubdict("nested"));
  EXPECT_EQ(child.Get<int>("id"), 7);
  EXPECT_FALSE(child.Exists<int>("parent"));
  EXPECT_FALSE(child.Exists<int>("alias"));
  EXPECT_FALSE(child.Exists<Button>("button"));
  dict.Set<int>("threads", 4);
  EXPECT_EQ(child.Get<int>("threads"), 4);
  EXPECT_EQ(nested.Get<int>("threads"), 4);
  source.GetMutableSubdict("child")->Set<int>("id", 9);
  source.GetMutableSubdict("child")->GetMutableSubdict("nested")->Set<int>(
      "new", 10);
  EXPECT_EQ(child.Get<int>("id"), 7);
  EXPECT_FALSE(nested.Exists<int>("new"));
  dict.CopySubdictsFrom(dict);
  EXPECT_EQ(&dict.GetSubdict("child"), &child);
}

TEST(OptionsDict, RemoveAllLocalTypesAndSubdictOnly) {
  OptionsDict parent;
  parent.Set<int>("key", 7);
  OptionsDict dict(&parent);
  dict.Set<bool>("key", true);
  dict.Set<int>("key", 42);
  dict.Set<float>("key", 1.5f);
  dict.Set<std::string>("key", "text");
  dict.Set<Button>("key", Button(true));
  dict.AddSubdict("key");
  dict.AddSubdict("nested")->Set<std::string>("key", "retained");
  const std::string name = "prefix-key-suffix";

  dict.Remove(std::string_view(name).substr(7, 3));
  EXPECT_FALSE(dict.OwnExists<bool>("key"));
  EXPECT_FALSE(dict.OwnExists<int>("key"));
  EXPECT_FALSE(dict.OwnExists<float>("key"));
  EXPECT_FALSE(dict.OwnExists<std::string>("key"));
  EXPECT_FALSE(dict.OwnExists<Button>("key"));
  EXPECT_FALSE(dict.HasSubdict("key"));
  EXPECT_EQ(dict.Get<int>("key"), 7);
  EXPECT_EQ(parent.Get<int>("key"), 7);
  EXPECT_EQ(dict.GetSubdict("nested").Get<std::string>("key"), "retained");
}

TEST(OptionsDict, SerializeScalarTypesAndLocalValues) {
  OptionsDict parent;
  parent.Set<int>("parent", 1);
  OptionsDict alias;
  alias.Set<int>("alias", 2);
  OptionsDict dict(&parent);
  dict.AddAliasDict(&alias);
  dict.AddSubdictFromString(
      "flag=true, count=42, rate=1.5, path=/tmp/foo, (gpu=0), (gpu=1)");
  dict.Set<Button>("button", Button(true));
  OptionsDict parsed;
  parsed.AddSubdictFromString(dict.Serialize());

  EXPECT_TRUE(parsed.Get<bool>("flag"));
  EXPECT_EQ(parsed.Get<int>("count"), 42);
  EXPECT_EQ(parsed.Get<float>("rate"), 1.5f);
  EXPECT_EQ(parsed.Get<std::string>("path"), "/tmp/foo");
  EXPECT_EQ(parsed.GetSubdict("[0]").Get<int>("gpu"), 0);
  EXPECT_EQ(parsed.GetSubdict("[1]").Get<int>("gpu"), 1);
  EXPECT_FALSE(parsed.Exists<int>("parent"));
  EXPECT_FALSE(parsed.Exists<int>("alias"));
  EXPECT_FALSE(parsed.Exists<Button>("button"));
  EXPECT_FALSE(parsed.Exists<std::string>("button"));
}

TEST(OptionsDict, SerializeFloatExactness) {
  for (float value :
       {0.0f, 1.0f, -1.0f, 1.2345678f, -1.2345678f, 1e-20f, 1e20f}) {
    SCOPED_TRACE(value);
    OptionsDict dict;
    dict.Set<float>("val", value);
    OptionsDict parsed;
    parsed.AddSubdictFromString(dict.Serialize());
    EXPECT_EQ(parsed.Get<float>("val"), value);
  }
}

TEST(OptionsDict, SerializeQuotesValuesAndNames) {
  for (const std::string value :
       {"", "true", "false", "42", "_foo", "-foo", ".hidden", "123abc", "12-34",
        "a b", "a,b", "a(b)", "a=b", "single'quote", "double\"quote"}) {
    SCOPED_TRACE(value);
    OptionsDict dict;
    dict.Set<std::string>(value, value);
    dict.AddSubdict(value)->Set<int>("id", 7);
    OptionsDict parsed;
    parsed.AddSubdictFromString(dict.Serialize());
    EXPECT_EQ(parsed.Get<std::string>(value), value);
    EXPECT_EQ(parsed.GetSubdict(value).Get<int>("id"), 7);
  }
}

TEST(OptionsDict, SerializePreservesAnonymousNames) {
  OptionsDict dict;
  for (int i = 0; i < 12; ++i)
    dict.AddSubdictFromString("(id=" + std::to_string(i) + ")");
  dict.Remove("[1]");
  OptionsDict parsed;
  parsed.AddSubdictFromString(dict.Serialize());
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
