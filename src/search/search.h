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

// Collection of pointers to objects that search needs but doesn't own. Just a
// convenience struct in order to avoid passing all of them separately in
// constructors.
struct SearchContext {
  UciResponder* uci_responder = nullptr;
  const OptionsDict* search_options = nullptr;
  SyzygyTablebase* syzygy_tb = nullptr;
  Backend* backend = nullptr;
};

// Base class for search runs. A separate instance is created for each search
// (i.e. for each move).
class SearchBase {
 public:
  explicit SearchBase(const SearchContext& context) : context_(context) {}
  virtual ~SearchBase() = default;

  // Start the search. Must not block, should return immediately.
  virtual void Start(const GoParams&) = 0;
  // Wait for the search to finish. This is blocking.
  virtual void Wait() = 0;
  // Stops the search as soon as possible and responds with bestmove. Doesn't
  // block.
  virtual void Stop() = 0;
  // Same as Stop(), but doesn't respond with bestmove. Doesn't block.
  virtual void Abort() = 0;
  // Return the data needed to build a training data frame (after the search is
  // done).
  virtual SearchArtifacts GetArtifacts() const {
    throw Exception(
        "Training data generation is not supported for this search algorithm.");
  }

 protected:
  UciResponder* uci_responder() const { return context_.uci_responder; }
  SyzygyTablebase* syzygy_tb() const { return context_.syzygy_tb; }
  Backend* backend() const { return context_.backend; }
  const OptionsDict& search_options() const { return *context_.search_options; }

 private:
  SearchContext context_;
};

// Search environment keeps the data that has to be shared between searches,
// for example the tree of the game, statistics, or whatever time manager wants
// to keep.
class SearchEnvironment {
 public:
  explicit SearchEnvironment(UciResponder* uci, const OptionsDict* dict)
      : context_{uci, dict} {}
  virtual ~SearchEnvironment() = default;

  // Resets search tree, and whatever else is needed to start a new game.
  virtual void NewGame() {}
  // Sets the position to search from in the future searches.
  virtual std::unique_ptr<SearchBase> CreateSearch(const GameState&) = 0;

 protected:
  SearchContext context_;
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
  virtual std::unique_ptr<SearchEnvironment> CreateEnvironment(
      UciResponder*, const OptionsDict*) const = 0;
};

}  // namespace lczero