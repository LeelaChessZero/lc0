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

#pragma once

#include <gmock/gmock.h>

#include "search/search.h"

namespace lczero {

class MockSearch : public SearchBase {
 public:
  using SearchBase::SearchBase;
  UciResponder* GetUciResponder() const { return uci_responder_; }
  MOCK_METHOD(void, SetBackend, (Backend * backend), (override));
  MOCK_METHOD(void, SetSyzygyTablebase, (SyzygyTablebase * tb), (override));
  MOCK_METHOD(void, NewGame, (), (override));
  MOCK_METHOD(void, SetPosition, (const GameState&), (override));
  MOCK_METHOD(void, StartSearch, (const GoParams&), (override));
  MOCK_METHOD(void, StartClock, (), (override));
  MOCK_METHOD(void, WaitSearch, (), (override));
  MOCK_METHOD(void, StopSearch, (), (override));
  MOCK_METHOD(void, AbortSearch, (), (override));
  MOCK_METHOD(SearchArtifacts, GetArtifacts, (), (const, override));
};

class MockSearchFactory : public SearchFactory {
 public:
  MOCK_METHOD(std::string_view, GetName, (), (const, override));
  MOCK_METHOD(void, PopulateParams, (OptionsParser*), (const, override));
  MOCK_METHOD(std::unique_ptr<SearchBase>, CreateSearch,
              (UciResponder*, const OptionsDict*), (const, override));
};

}  // namespace lczero
