/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2025 The LCZero Authors

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

#include "engine.h"

#include "gtest/gtest.h"
#include "neural/mock_backend.h"
#include "neural/register.h"
#include "search/mock_search.h"
#include "search/search.h"

namespace lczero {
namespace {

class EngineTest : public ::testing::Test {
 protected:
  EngineTest() {
    auto backend_factory = std::make_unique<MockBackendFactory>();
    backend_factory_ = backend_factory.get();
    BackendManager::Get()->AddBackend(std::move(backend_factory));

    auto search = std::make_unique<MockSearch>();
    search_ = search.get();
    EXPECT_CALL(search_factory_, CreateSearch(testing::_, testing::_))
        .WillOnce(testing::Return(std::move(search)));
  }

  ~EngineTest() { BackendManager::Get()->RemoveBackend(backend_factory_); }

  OptionsDict options_;
  BackendFactory* backend_factory_ = nullptr;
  MockSearchFactory search_factory_;
  MockSearch* search_ = nullptr;
  std::unique_ptr<Engine> engine_;
};

TEST_F(EngineTest, ExampleTest) {
  Engine engine(search_factory_, options_);

  // Add test logic here.
  EXPECT_TRUE(true);
}

}  // namespace
}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
