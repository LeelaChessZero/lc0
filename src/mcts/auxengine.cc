/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2019 The LCZero Authors

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

#include "mcts/search.h"

#include <algorithm>
#include <boost/process.hpp>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <thread>

#include "mcts/node.h"
#include "neural/cache.h"
#include "neural/encoder.h"
#include "utils/fastmath.h"
#include "utils/random.h"

namespace lczero {

boost::process::ipstream Search::auxengine_is_;
boost::process::opstream Search::auxengine_os_;
boost::process::child Search::auxengine_c_;
bool Search::auxengine_ready_ = false;

void Search::OpenAuxEngine() {
  if (params_.GetAuxEnginePath() == "") return;
  auxengine_threads_.emplace_back([this]() { AuxEngineWorker(); });
}

void SearchWorker::AuxMaybeEnqueueNode(Node* n) {
  if (params_.GetAuxEnginePath() != "" &&
      n->GetN() >= params_.GetAuxEngineThreshold() &&
      n->GetAuxEngineMove() == 0xffff &&
      !n->IsTerminal()) {
    n->SetAuxEngineMove(0xfffe); // TODO: magic for pending
    std::lock_guard<std::mutex> lock(search_->auxengine_mutex_);
    search_->auxengine_queue_.push(n);
    search_->auxengine_cv_.notify_one();
  }
}

void Search::AuxEngineWorker() {
  if (!auxengine_ready_) {
    auxengine_c_ = boost::process::child(params_.GetAuxEnginePath(), boost::process::std_in < auxengine_os_, boost::process::std_out > auxengine_is_);
    {
      std::istringstream iss(params_.GetAuxEngineOptions());
      std::string token;
      while(std::getline(iss, token, '=')) {
        std::ostringstream oss;
        oss << "setoption name " << token;
        std::getline(iss, token, ';');
        oss << " value " << token;
        auxengine_os_ << oss.str() << std::endl;
        LOGFILE << oss.str();
      }
      auxengine_os_ << "uci" << std::endl;
    }
    std::string line;
    while(std::getline(auxengine_is_, line)) {
      LOGFILE << line;
      std::istringstream iss(line);
      std::string token;
      iss >> token >> std::ws;
      if (token == "uciok") {
        break;
      } else if (token == "option") {
        iss >> token >> std::ws;
        if (token == "name") {
          iss >> token >> std::ws;
          if (token == "SyzygyPath") {
            std::ostringstream oss;
            oss << "setoption name SyzygyPath value " << syzygy_tb_->get_paths();
            auxengine_os_ << oss.str() << std::endl;
            LOGFILE << oss.str();
          }
        }
      }
    }
    auxengine_ready_ = true;
  }
  if (current_position_fen_ == "") {
    current_position_fen_ = ChessBoard::kStartposFen; // TODO
  }
  if (current_position_moves_.size()) {
    for (auto i = current_position_moves_.size(); i-- > 0;) {
      current_uci_ = current_position_moves_[i] + " " + current_uci_;
    }
  }
  current_uci_ = "position fen " + current_position_fen_ + " moves " + current_uci_;
  LOGFILE << current_uci_;

  Node* n;
  LOGFILE << "start";
  while (!stop_.load(std::memory_order_acquire)) {
    {
      std::unique_lock<std::mutex> lock(auxengine_mutex_);

      // Wait until there's some work to compute.
      auxengine_cv_.wait(lock, [&] { return stop_.load(std::memory_order_acquire) || !auxengine_queue_.empty(); });
      if (stop_.load(std::memory_order_acquire)) break;
      n = auxengine_queue_.front();
      auxengine_queue_.pop();
    } // release lock
    DoAuxEngine(n);
  }
  LOGFILE << "end";
}

void Search::DoAuxEngine(Node* n) {
  if (n->GetAuxEngineMove() < 0xfffe) {
    return;
  }
  int depth = 0;
  for (Node* n2 = n; n2 != root_node_; n2 = n2->GetParent()) {
    depth++;
  }
  std::string s = "";
  bool flip = played_history_.IsBlackToMove() ^ (depth % 2 == 0);
  for (Node* n2 = n; n2 != root_node_; n2 = n2->GetParent()) {
    s = n2->GetOwnEdge()->GetMove(flip).as_string() + " " + s;
    flip = !flip;
  }
  LOGFILE << "add pv=" << s;
  s = current_uci_ + " " + s;
  auxengine_os_ << s << std::endl;
  auxengine_os_ << "go depth " << params_.GetAuxEngineDepth() << std::endl;
  std::string prev_line;
  std::string line;
  std::string token;
  while(std::getline(auxengine_is_, line)) {
    //LOGFILE << "auxe:" << line;
    std::istringstream iss(line);
    iss >> token >> std::ws;
    if (token == "bestmove") {
      iss >> token;
      break;
    }
    prev_line = line;
  }
  std::istringstream iss(prev_line);
  std::string pv;
  std::vector<uint16_t> pv_moves;
  flip = played_history_.IsBlackToMove() ^ (depth % 2 == 0);
  auto bestmove_packed_int = Move(token, !flip).as_packed_int();
  while(iss >> pv >> std::ws) {
    if (pv == "pv") {
      while(iss >> pv >> std::ws) {
        auto m = Move(pv, !flip);
        pv_moves.push_back(m.as_packed_int());
        flip = !flip;
      }
    }
  }
  LOGFILE << "pv:" << prev_line;
  LOGFILE << "bestanswer:" << token;

  if (pv_moves[0] != bestmove_packed_int) {
    // TODO: Is it possible for PV to not match bestmove?
    LOGFILE << "error: pv doesn't match bestmove:" << pv_moves[0] << " " << "bm" << bestmove_packed_int;
    pv_moves.clear();
    pv_moves.push_back(bestmove_packed_int);
  }
  // Take the lock and update the P value of the bestmove
  SharedMutex::Lock lock(nodes_mutex_);
  AuxUpdateP(n, pv_moves, 0);
}

void Search::AuxUpdateP(Node* n, std::vector<uint16_t> pv_moves, int ply) {
  if (n->GetAuxEngineMove() < 0xfffe) {
    // This can happen because nodes are placed in the queue from
    // deepest node first during DoBackupSingeNode
    //if (n->GetAuxEngineMove() != pv_moves[ply]) {
    //  LOGFILE << "already done: curr:" << n->GetAuxEngineMove() << " new:" << pv_moves[ply] << " (error? mismatch)";
    //} else {
    //  LOGFILE << "already done";
    //}
    return;
  }
  for (const auto& edge : n->Edges()) {
    if (edge.GetMove().as_packed_int() == pv_moves[ply]) {
      auto new_p = edge.GetP() + params_.GetAuxEngineBoost()/100.0f;
      edge.edge()->SetP(std::min(new_p, 1.0f));
      // Modifying P invalidates best child logic.
      n->InvalidateBestChild();
      // Stop after AuxEngineFollowPvDepth plies deep.
      // or when e.g. ply=0, pv_moves.size()=1 (pv is exhasted)
      // of the edge does not have a node yet.
      // of the edge is a terminal.
      if (ply < params_.GetAuxEngineFollowPvDepth() &&
          ply+1 < pv_moves.size() &&
          edge.HasNode() &&
          !edge.IsTerminal()) {
        AuxUpdateP(edge.node(), pv_moves, ply+1);
      }
      n->SetAuxEngineMove(pv_moves[ply]);
      return;
    }
  }
  LOGFILE << "error: move not found";
}

void Search::AuxWait() {
  while (!auxengine_threads_.empty()) {
    LOGFILE << "Wait for auxengine_threads";
    auxengine_threads_.back().join();
    auxengine_threads_.pop_back();
  }
  // TODO: For now with this simple queue method,
  // mark unfinished nodes not done again, and delete the queue.
  // Next search iteration will fill it again.
  LOGFILE << "done waiting. auxengine_queue_ size " << auxengine_queue_.size();
  while (!auxengine_queue_.empty()) {
    auto n = auxengine_queue_.front();
    assert(n->GetAuxEngineMove() != 0xffff);
    if (n->GetAuxEngineMove() == 0xfffe) {
      n->SetAuxEngineMove(0xffff);
    }
    auxengine_queue_.pop();
  }
}

}  // namespace lczero
