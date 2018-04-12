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

#pragma once

#include <memory>
#include <mutex>
#include "chess/board.h"

namespace lczero {

struct Node {
  // Move corresponding to this node. From the point of view of a player,
  // i.e. black's e7e5 is stored as e2e4.
  // Root node contains move a1a1.
  Move move;
  // The board from the point of view of the player to move.
  ChessBoard board;
  // How many half-moves without capture or pawn move was there.
  std::uint8_t no_capture_ply;
  // How many repetitions this position had before. For new positions it's 0.
  std::uint8_t repetitions;
  // number of half-moves since beginning of the game.
  std::uint16_t ply_count;

  // (aka virtual loss). How many threads currently process this node (started
  // but not finished). This value is added to n during selection which node
  // to pick in MCTS, and also when selecting the best move.
  uint32_t n_in_flight;
  // How many completed visits this node had.
  uint32_t n;
  // Q value fetched from neural network.
  float v;
  // Average value (from value head of neural network) of all visited nodes in
  // subtree. Terminal nodes (which lead to checkmate or draw) may be visited
  // several times, those are counted several times. q = w / n
  float q;
  // Sum of values of all visited nodes in a subtree. Used to compute an
  // average.
  float w;
  // Probabality that this move will be made. From policy head of the neural
  // network.
  float p;

  // Maximum depth any subnodes of this node were looked at.
  uint16_t max_depth;
  // Complete depth all subnodes of this node were fully searched.
  uint16_t full_depth;
  // Does this node end game (with a winning of either sides or draw).
  bool is_terminal;

  // Pointer to a parent node. nullptr for the root.
  Node* parent;
  // Pointer to a first child. nullptr for leave node.
  Node* child;
  // Pointer to a next sibling. nullptr if there are no further siblings.
  Node* sibling;

  std::string DebugString() const;
};

int ComputeRepetitions(const Node*);

class NodePool {
 public:
  // Allocates a new node and initializes it with all zeros.
  Node* GetNode();
  // Return node to the pool.
  void ReleaseNode(Node*);
  // Releases all children of the node, except specified. Also updates pointers
  // accordingly.
  void ReleaseAllChildrenExceptOne(Node* root, Node* subtree);
  // Releases all children, but doesn't release the node isself.
  void ReleaseChildren(Node*);

  // Returns total number of nodes allocated.
  uint64_t GetAllocatedNodeCount() const;

 private:
  // Release all children of the node and the node itself.
  void ReleaseSubtree(Node*);
  void AllocateNewBatch();

  mutable std::mutex mutex_;
  std::vector<Node*> pool_;
  std::vector<std::unique_ptr<Node[]>> allocations_;
};

}  // namespace lczero