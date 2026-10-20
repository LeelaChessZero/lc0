/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2026 The LCZero Authors

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

#include <gtest/gtest.h>

#include "utils/exception.h"

namespace lczero {

TEST(OptionsDict, CheckAllOptionsReadAcceptsReadOptions) {
  OptionsDict dict;
  dict.AddSubdictFromString("legacy(steepness=4.0)");
  const OptionsDict& legacy = dict.GetSubdict("legacy");
  EXPECT_FLOAT_EQ(legacy.GetOrDefault<float>("steepness", 7.0f), 4.0f);
  EXPECT_NO_THROW(dict.CheckAllOptionsRead(""));
}

TEST(OptionsDict, CheckAllOptionsReadRejectsUnreadOption) {
  OptionsDict dict;
  dict.AddSubdictFromString("legacy(steepnes=4.0)");
  const OptionsDict& legacy = dict.GetSubdict("legacy");
  legacy.GetOrDefault<float>("steepness", 7.0f);
  EXPECT_THROW(dict.CheckAllOptionsRead(""), Exception);
}

TEST(OptionsDict, CheckAllOptionsReadRejectsUnreadSubdict) {
  // A missing "=" makes "steepness4.0" a subdictionary instead of an option.
  OptionsDict dict;
  dict.AddSubdictFromString("legacy(steepness4.0)");
  const OptionsDict& legacy = dict.GetSubdict("legacy");
  legacy.GetOrDefault<float>("steepness", 7.0f);
  EXPECT_THROW(dict.CheckAllOptionsRead(""), Exception);
}

TEST(OptionsDict, CheckAllOptionsReadAcceptsSubdictsWithoutParentheses) {
  OptionsDict dict;
  dict.AddSubdictFromString("one,two");
  dict.GetSubdict("one");
  dict.GetSubdict("two");
  EXPECT_NO_THROW(dict.CheckAllOptionsRead(""));
}

}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
