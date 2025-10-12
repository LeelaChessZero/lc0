/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2024 The LCZero Authors

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

#include <memory>
#include <span>

#include "search/artifacts.h"
#include "utils/exception.h"

namespace lczero {

class Backend;
struct GameState;
struct GoParams;
class OptionsDict;
class OptionsParser;
class UciResponder;
class SyzygyTablebase;

class SearchBase {
 public:
  SearchBase(UciResponder* responder) : uci_responder_(responder) {}
  virtual ~SearchBase() = default;

  // Sets objects needed by the search.
  // They are guarnteed to be set before any other function is called, and after
  // that, only can be changed while the search is stopped.
  virtual void SetBackend(Backend* backend) { backend_ = backend; }
  virtual void SetSyzygyTablebase(SyzygyTablebase* tb) { syzygy_tb_ = tb; }

  // Resets search tree, and whatever else is needed to start a new game.
  virtual void NewGame() {}
  // Sets the position to search from in the future searches.
  virtual void SetPosition(const GameState&) = 0;
  // Start the search. Must not block, should return immediately.
  virtual void StartSearch(const GoParams&) = 0;
  // Starts the timer for the search. Must not block, should return immediately.
  // It can be called either after or befor StartSearch(), particularly:
  // - In the "strict timing" mode, it's called before SetPosition().
  // - In normal mode, it's called before StartSearch().
  // - In Ponder mode, it may potentially be called at `ponderhit` (although
  // actually we'll stop the search, change the position and start again).
  virtual void StartClock() = 0;
  // Wait for the search to finish. This is blocking.
  virtual void WaitSearch() = 0;
  // Stops the search as soon as possible and responds with bestmove. Doesn't
  // block.
  virtual void StopSearch() = 0;
  // Same as Stop(), but doesn't respond with bestmove. Doesn't block.
  virtual void AbortSearch() = 0;
  // Return the data needed to build a training data frame from the last search.
  virtual SearchArtifacts GetArtifacts() const {
    throw Exception(
        "Training data generation is not supported for this search algorithm.");
  }

 protected:
  UciResponder* uci_responder_ = nullptr;
  Backend* backend_ = nullptr;
  SyzygyTablebase* syzygy_tb_ = nullptr;
};

// Creates an environment for a given search algorithm. One instance of the
// factory per algorithm is created at the start of the program, and registered
// in the SearchManager.
class SearchFactory {
 public:
  virtual ~SearchFactory() = default;
  // Name of the algorithm (used in UCI options or command line).
  virtual std::string_view GetName() const = 0;
  // Populates the parameters of the algorithm.
  virtual void PopulateParams(OptionsParser*) const {}
  // Creates a new environment for the algorithm.
  virtual std::unique_ptr<SearchBase> CreateSearch(
      UciResponder*, const OptionsDict*) const = 0;
};

}  // namespace lczero