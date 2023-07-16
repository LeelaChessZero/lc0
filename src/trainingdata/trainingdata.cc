/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2021 The LCZero Authors

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

#include "trainingdata/trainingdata.h"

namespace lczero {

namespace {
std::tuple<float, float> DriftCorrect(float q, float d) {
  // Training data doesn't have a high number of nodes, so there shouldn't be
  // too much drift. Highest known value not caused by backend bug was 1.5e-7.
  const float allowed_eps = 0.000001f;
  if (q > 1.0f) {
    if (q > 1.0f + allowed_eps) {
      CERR << "Unexpectedly large drift in q " << q;
    }
    q = 1.0f;
  }
  if (q < -1.0f) {
    if (q < -1.0f - allowed_eps) {
      CERR << "Unexpectedly large drift in q " << q;
    }
    q = -1.0f;
  }
  if (d > 1.0f) {
    if (d > 1.0f + allowed_eps) {
      CERR << "Unexpectedly large drift in d " << d;
    }
    d = 1.0f;
  }
  if (d < 0.0f) {
    if (d < 0.0f - allowed_eps) {
      CERR << "Unexpectedly large drift in d " << d;
    }
    d = 0.0f;
  }
  float w = (1.0f - d + q) / 2.0f;
  float l = w - q;
  // Assume q drift is rarer than d drift and apply all correction to d.
  if (w < 0.0f || l < 0.0f) {
    float drift = 2.0f * std::min(w, l);
    if (drift < -allowed_eps) {
      CERR << "Unexpectedly large drift correction for d based on q. " << drift;
    }
    d += drift;
    // Since q is in range -1 to 1 - this correction should never push d outside
    // of range, but precision could be lost in calculations so just in case.
    if (d < 0.0f) {
      d = 0.0f;
    }
  }
  return {q, d};
}
}  // namespace

void V6TrainingDataArray::Write(TrainingDataWriter* writer, GameResult result,
                                bool adjudicated) const {
  if (training_data_.empty()) return;
  // Base estimate off of best_m.  If needed external processing can use a
  // different approach.
  float m_estimate = training_data_.back().best_m + training_data_.size() - 1;
  for (auto chunk : training_data_) {
    bool black_to_move = chunk.side_to_move_or_enpassant;
    if (IsCanonicalFormat(static_cast<pblczero::NetworkFormat::InputFormat>(
            chunk.input_format))) {
      black_to_move = (chunk.invariance_info & (1u << 7)) != 0;
    }
    if (result == GameResult::WHITE_WON) {
      chunk.result_q = black_to_move ? -1 : 1;
      chunk.result_d = 0;
    } else if (result == GameResult::BLACK_WON) {
      chunk.result_q = black_to_move ? 1 : -1;
      chunk.result_d = 0;
    } else {
      chunk.result_q = 0;
      chunk.result_d = 1;
    }
    if (adjudicated) {
      chunk.invariance_info |= 1u << 5;  // Game adjudicated.
    }
    if (adjudicated && result == GameResult::UNDECIDED) {
      chunk.invariance_info |= 1u << 4;  // Max game length exceeded.
    }
    chunk.plies_left = m_estimate;
    m_estimate -= 1.0f;
    writer->WriteChunk(chunk);
  }
}

void V6TrainingDataArray::Add(const Node* node, const PositionHistory& history,
                              Eval best_eval, Eval played_eval,
                              bool best_is_proven, Move best_move,
                              Move played_move, const NNCacheLock& nneval) {
  V6TrainingData result;
  const auto& position = history.Last();

  // Set version.
  result.version = 6;
  result.input_format = input_format_;

  // Populate planes.
  int transform;
  InputPlanes planes = EncodePositionForNN(
      input_format_, history, 8, fill_empty_history_[position.IsBlackToMove()],
      &transform);
  int plane_idx = 0;
  for (auto& plane : result.planes) {
    plane = ReverseBitsInBytes(planes[plane_idx++].mask);
  }

  // Populate probabilities.
  auto total_n = node->GetChildrenVisits();
  // Prevent garbage/invalid training data from being uploaded to server.
  // It's possible to have N=0 when there is only one legal move in position
  // (due to smart pruning).
  if (total_n == 0 && node->GetNumEdges() != 1) {
    throw Exception("Search generated invalid data!");
  }
  // Set illegal moves to have -1 probability.
  std::fill(std::begin(result.probabilities), std::end(result.probabilities),
            -1);
  // Set moves probabilities according to their relative amount of visits.
  // Compute Kullback-Leibler divergence in nats (between policy and visits).
  float kld_sum = 0;
  float max_p = -std::numeric_limits<float>::infinity();
  std::vector<float> intermediate;
  if (nneval) {
    int last_idx = 0;
    for (const auto& child : node->Edges()) {
      auto nn_idx = child.edge()->GetMove().as_nn_index(transform);
      float p = 0;
      for (int i = 0; i < nneval->p.size(); i++) {
        // Optimization: usually moves are stored in the same order as queried.
        const auto& move = nneval->p[last_idx++];
        if (last_idx == nneval->p.size()) last_idx = 0;
        if (move.first == nn_idx) {
          p = move.second;
          break;
        }
      }
      intermediate.emplace_back(p);
      max_p = std::max(max_p, p);
    }
  }
  float total = 0.0;
  auto it = intermediate.begin();
  for (const auto& child : node->Edges()) {
    auto nn_idx = child.edge()->GetMove().as_nn_index(transform);
    float fracv = total_n > 0 ? child.GetN() / static_cast<float>(total_n) : 1;
    if (nneval) {
      float P = std::exp(*it - max_p);
      if (fracv > 0) {
        kld_sum += fracv * std::log(fracv / P);
      }
      total += P;
      it++;
    }
    result.probabilities[nn_idx] = fracv;
  }
  if (nneval) {
    // Add small epsilon for backward compatibility with earlier value of 0.
    auto epsilon = std::numeric_limits<float>::min();
    kld_sum = std::max(kld_sum + std::log(total), 0.0f) + epsilon;
  }
  result.policy_kld = kld_sum;

  const auto& castlings = position.GetBoard().castlings();
  // Populate castlings.
  // For non-frc trained nets, just send 1 like we used to.
  uint8_t our_queen_side = 1;
  uint8_t our_king_side = 1;
  uint8_t their_queen_side = 1;
  uint8_t their_king_side = 1;
  // If frc trained, send the bit mask representing rook position.
  if (Is960CastlingFormat(input_format_)) {
    our_queen_side <<= castlings.our_queenside_rook();
    our_king_side <<= castlings.our_kingside_rook();
    their_queen_side <<= castlings.their_queenside_rook();
    their_king_side <<= castlings.their_kingside_rook();
  }

  result.castling_us_ooo = castlings.we_can_000() ? our_queen_side : 0;
  result.castling_us_oo = castlings.we_can_00() ? our_king_side : 0;
  result.castling_them_ooo = castlings.they_can_000() ? their_queen_side : 0;
  result.castling_them_oo = castlings.they_can_00() ? their_king_side : 0;

  // Other params.
  if (IsCanonicalFormat(input_format_)) {
    result.side_to_move_or_enpassant =
        position.GetBoard().en_passant().as_int() >> 56;
    if ((transform & FlipTransform) != 0) {
      result.side_to_move_or_enpassant =
          ReverseBitsInBytes(result.side_to_move_or_enpassant);
    }
    // Send transform in deprecated move count so rescorer can reverse it to
    // calculate the actual move list from the input data.
    result.invariance_info =
        transform | (position.IsBlackToMove() ? (1u << 7) : 0u);
  } else {
    result.side_to_move_or_enpassant = position.IsBlackToMove() ? 1 : 0;
    result.invariance_info = 0;
  }
  if (best_is_proven) {
    result.invariance_info |= 1u << 3;  // Best node is proven best;
  }
  result.dummy = 0;
  result.rule50_count = position.GetRule50Ply();

  // Game result is undecided.
  result.result_q = 0;
  result.result_d = 1;

  Eval orig_eval;
  if (nneval) {
    orig_eval.wl = nneval->q;
    orig_eval.d = nneval->d;
    orig_eval.ml = nneval->m;
  } else {
    orig_eval.wl = std::numeric_limits<float>::quiet_NaN();
    orig_eval.d = std::numeric_limits<float>::quiet_NaN();
    orig_eval.ml = std::numeric_limits<float>::quiet_NaN();
  }

  // Aggregate evaluation WL.
  result.root_q = -node->GetWL();
  result.best_q = best_eval.wl;
  result.played_q = played_eval.wl;
  result.orig_q = orig_eval.wl;

  // Draw probability of WDL head.
  result.root_d = node->GetD();
  result.best_d = best_eval.d;
  result.played_d = played_eval.d;
  result.orig_d = orig_eval.d;

  std::tie(result.best_q, result.best_d) =
      DriftCorrect(result.best_q, result.best_d);
  std::tie(result.root_q, result.root_d) =
      DriftCorrect(result.root_q, result.root_d);
  std::tie(result.played_q, result.played_d) =
      DriftCorrect(result.played_q, result.played_d);

  result.root_m = node->GetM();
  result.best_m = best_eval.ml;
  result.played_m = played_eval.ml;
  result.orig_m = orig_eval.ml;

  result.visits = node->GetN();
  if (position.IsBlackToMove()) {
    best_move.Mirror();
    played_move.Mirror();
  }
  result.best_idx = best_move.as_nn_index(transform);
  result.played_idx = played_move.as_nn_index(transform);
  result.reserved = 0;

  // Unknown here - will be filled in once the full data has been collected.
  result.plies_left = 0;
  training_data_.push_back(result);
}

}  // namespace lczero
