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
*/

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "neural/backend.h"
#include "neural/shared_params.h"
#include "search/classic/params.h"
#include "search/classic/search.h"
#include "search/classic/stoppers/stoppers.h"
#ifdef HAVE_DAG_CLASSIC
#include "search/dag_classic/params.h"
#include "search/dag_classic/search.h"
#endif
#include "utils/optionsparser.h"

namespace lczero {
namespace {

class DummyComputation : public BackendComputation {
 public:
  AddInputResult AddInput(const EvalPosition&, EvalResultPtr) override {
    return ENQUEUED_FOR_EVAL;
  }
  size_t UsedBatchSize() const override { return 0; }
  void ComputeBlocking() override {}
};

class DummyBackend : public Backend {
 public:
  BackendAttributes GetAttributes() const override {
    BackendAttributes attrs{};
    attrs.recommended_batch_size = 1;
    attrs.maximum_batch_size = 1;
    return attrs;
  }
  std::unique_ptr<BackendComputation> CreateComputation() override {
    return std::make_unique<DummyComputation>();
  }
};

class NullUciResponder : public UciResponder {
 public:
  void OutputBestMove(BestMoveInfo*) override {}
  void OutputThinkingInfo(std::vector<ThinkingInfo>*) override {}
};

struct ClassicOptions {
  OptionsParser parser;
  ClassicOptions() {
    SharedBackendParams::Populate(&parser);
    classic::SearchParams::Populate(&parser);
    parser.GetMutableOptions()->Set<int>(
        classic::BaseSearchParams::kTaskWorkersPerSearchWorkerId, 0);
    parser.GetMutableOptions()->Set<bool>(
        classic::BaseSearchParams::kStickyEndgamesId, true);
  }
  const OptionsDict& Get() { return parser.GetOptionsDict(); }
};

#ifdef HAVE_DAG_CLASSIC
struct DagClassicOptions {
  OptionsParser parser;
  DagClassicOptions() {
    SharedBackendParams::Populate(&parser);
    dag_classic::SearchParams::Populate(&parser);
    parser.GetMutableOptions()->Set<int>(
        classic::BaseSearchParams::kTaskWorkersPerSearchWorkerId, 0);
    parser.GetMutableOptions()->Set<bool>(
        classic::BaseSearchParams::kStickyEndgamesId, true);
  }
  const OptionsDict& Get() { return parser.GetOptionsDict(); }
};
#endif

Move MakeMove(std::string_view uci) {
  return Move::White(Square::Parse(uci.substr(0, 2)),
                     Square::Parse(uci.substr(2, 2)));
}

}  // namespace

namespace classic {

struct SearchBoundsTestPeer {
  static void Backup(SearchWorker& worker, Node* leaf, float q, float d,
                     float m, uint16_t depth, int multivisit = 1) {
    SearchWorker::NodeToProcess ntp =
        SearchWorker::NodeToProcess::Visit(leaf, depth);
    ntp.eval->q = q;
    ntp.eval->d = d;
    ntp.eval->m = m;
    ntp.multivisit = multivisit;
    SharedMutex::Lock lock(worker.search_->nodes_mutex_);
    worker.DoBackupUpdateSingleNode(ntp);
  }

  static EdgeAndNode CurrentBestEdge(const Search& search) {
    return search.current_best_edge_;
  }
};

TEST(ClassicBoundsTest, RefineTerminalShorterMateUpdatesParentAndGrandparent) {
  NodeTree tree;
  tree.ResetToPosition(ChessBoard::kStartposFen, {});
  Node* root = tree.GetCurrentHead();

  // root -> gp -> p -> {c1 (M=9 EoG), c2 (M=2 EoG)}
  root->CreateEdges({MakeMove("e2e4")});
  root->TryStartScoreUpdate();
  root->FinalizeScoreUpdate(0.0f, 0.0f, 12.0f, 1);

  auto root_it = root->Edges();
  Node* gp = root_it.GetOrSpawnNode(root);
  gp->CreateEdges({MakeMove("e7e5"), MakeMove("d7d5")});
  gp->TryStartScoreUpdate();
  gp->FinalizeScoreUpdate(0.0f, 0.0f, 11.0f, 1);

  auto gp_it = gp->Edges();
  Node* p = gp_it.GetOrSpawnNode(gp);
  p->CreateEdges({MakeMove("d1h5"), MakeMove("f1c4")});
  p->MakeTerminal(GameResult::BLACK_WON, 10.0f, Node::Terminal::EndOfGame);
  // Simulate 5 prior visits to p (m = 10) already averaged into gp and root.
  for (int i = 0; i < 5; ++i) {
    p->TryStartScoreUpdate();
    p->FinalizeScoreUpdate(-1.0f, 0.0f, 10.0f, 1);
    gp->TryStartScoreUpdate();
    gp->FinalizeScoreUpdate(1.0f, 0.0f, 11.0f, 1);
    root->TryStartScoreUpdate();
    root->FinalizeScoreUpdate(-1.0f, 0.0f, 12.0f, 1);
  }
  ASSERT_EQ(p->GetN(), 5u);
  ASSERT_EQ(gp->GetN(), 6u);
  ASSERT_NEAR(gp->GetM(), 11.0f, 1e-4f);

  auto p_it = p->Edges();
  Node* c1 = p_it.GetOrSpawnNode(p);
  c1->MakeTerminal(GameResult::WHITE_WON, 9.0f, Node::Terminal::EndOfGame);
  c1->TryStartScoreUpdate();
  c1->FinalizeScoreUpdate(1.0f, 0.0f, 9.0f, 1);

  ++p_it;
  Node* c2 = p_it.GetOrSpawnNode(p);
  c2->MakeTerminal(GameResult::WHITE_WON, 2.0f, Node::Terminal::EndOfGame);

  // Start in-flight visit along path root -> gp -> p -> c2.
  ASSERT_TRUE(root->TryStartScoreUpdate());
  ASSERT_TRUE(gp->TryStartScoreUpdate());
  ASSERT_TRUE(p->TryStartScoreUpdate());
  ASSERT_TRUE(c2->TryStartScoreUpdate());

  DummyBackend backend;
  ClassicOptions opts;
  Search search(tree, &backend, std::make_unique<NullUciResponder>(), {},
                std::chrono::steady_clock::now(),
                std::make_unique<ChainedSearchStopper>(), false, false,
                opts.Get(), nullptr);
  SearchWorker worker(&search, search.GetParams());

  SearchBoundsTestPeer::Backup(worker, c2, 1.0f, 0.0f, 2.0f, 3);

  EXPECT_TRUE(p->IsTerminal());
  EXPECT_FALSE(p->IsTbTerminal());
  EXPECT_FLOAT_EQ(p->GetWL(), -1.0f);
  EXPECT_FLOAT_EQ(p->GetM(), 3.0f);
  EXPECT_EQ(p->GetN(), 6u);

  // gp had 1 initial visit (m=11) + 5 visits through p (shifted from 11 to 4)
  // + 1 new visit (m=4) => (11 + 6 * 4) / 7 = 35 / 7 = 5.0.
  EXPECT_EQ(gp->GetN(), 7u);
  EXPECT_NEAR(gp->GetM(), 5.0f, 1e-4f);
  EXPECT_FALSE(gp->IsTerminal());
}

TEST(ClassicBoundsTest, UpgradeTbToEndOfGamePropagatesThroughLosingParent) {
  NodeTree tree;
  tree.ResetToPosition(ChessBoard::kStartposFen, {});
  Node* root = tree.GetCurrentHead();

  // root -> rc (WHITE_WON losing for opp, TB, m=12) -> p (BLACK_WON, TB, m=11)
  //      -> {c1 (WHITE_WON, TB, m=10), c2 (WHITE_WON, EoG, m=4)}
  root->CreateEdges({MakeMove("e2e4")});
  root->TryStartScoreUpdate();
  root->FinalizeScoreUpdate(0.0f, 0.0f, 13.0f, 1);

  auto root_it = root->Edges();
  Node* rc = root_it.GetOrSpawnNode(root);
  rc->CreateEdges({MakeMove("e7e5")});
  rc->MakeTerminal(GameResult::WHITE_WON, 12.0f, Node::Terminal::Tablebase);
  rc->TryStartScoreUpdate();
  rc->FinalizeScoreUpdate(1.0f, 0.0f, 12.0f, 1);

  auto rc_it = rc->Edges();
  Node* p = rc_it.GetOrSpawnNode(rc);
  p->CreateEdges({MakeMove("d1h5"), MakeMove("f1c4")});
  p->MakeTerminal(GameResult::BLACK_WON, 11.0f, Node::Terminal::Tablebase);
  p->TryStartScoreUpdate();
  p->FinalizeScoreUpdate(-1.0f, 0.0f, 11.0f, 1);

  auto p_it = p->Edges();
  Node* c1 = p_it.GetOrSpawnNode(p);
  c1->MakeTerminal(GameResult::WHITE_WON, 10.0f, Node::Terminal::Tablebase);
  c1->TryStartScoreUpdate();
  c1->FinalizeScoreUpdate(1.0f, 0.0f, 10.0f, 1);

  ++p_it;
  Node* c2 = p_it.GetOrSpawnNode(p);
  c2->MakeTerminal(GameResult::WHITE_WON, 4.0f, Node::Terminal::EndOfGame);

  ASSERT_TRUE(root->TryStartScoreUpdate());
  ASSERT_TRUE(rc->TryStartScoreUpdate());
  ASSERT_TRUE(p->TryStartScoreUpdate());
  ASSERT_TRUE(c2->TryStartScoreUpdate());

  DummyBackend backend;
  ClassicOptions opts;
  Search search(tree, &backend, std::make_unique<NullUciResponder>(), {},
                std::chrono::steady_clock::now(),
                std::make_unique<ChainedSearchStopper>(), false, false,
                opts.Get(), nullptr);
  SearchWorker worker(&search, search.GetParams());

  SearchBoundsTestPeer::Backup(worker, c2, 1.0f, 0.0f, 4.0f, 3);

  EXPECT_TRUE(p->IsTerminal());
  EXPECT_FALSE(p->IsTbTerminal());
  EXPECT_FLOAT_EQ(p->GetM(), 5.0f);

  // rc only had child p, so upgrading p upgrades rc to EndOfGame with m = 6.
  EXPECT_TRUE(rc->IsTerminal());
  EXPECT_FALSE(rc->IsTbTerminal());
  EXPECT_FLOAT_EQ(rc->GetWL(), 1.0f);
  EXPECT_FLOAT_EQ(rc->GetM(), 6.0f);
  EXPECT_EQ(SearchBoundsTestPeer::CurrentBestEdge(search).node(), rc);
}

TEST(ClassicBoundsTest, NeverDowngradesEndOfGameToTablebase) {
  NodeTree tree;
  tree.ResetToPosition(ChessBoard::kStartposFen, {});
  Node* root = tree.GetCurrentHead();

  root->CreateEdges({MakeMove("e2e4")});
  root->TryStartScoreUpdate();
  root->FinalizeScoreUpdate(0.0f, 0.0f, 10.0f, 1);

  auto root_it = root->Edges();
  Node* p = root_it.GetOrSpawnNode(root);
  p->CreateEdges({MakeMove("e7e5"), MakeMove("d7d5")});
  p->MakeTerminal(GameResult::BLACK_WON, 8.0f, Node::Terminal::EndOfGame);
  p->TryStartScoreUpdate();
  p->FinalizeScoreUpdate(-1.0f, 0.0f, 8.0f, 1);

  auto p_it = p->Edges();
  Node* c1 = p_it.GetOrSpawnNode(p);
  c1->MakeTerminal(GameResult::WHITE_WON, 7.0f, Node::Terminal::EndOfGame);
  c1->TryStartScoreUpdate();
  c1->FinalizeScoreUpdate(1.0f, 0.0f, 7.0f, 1);

  ++p_it;
  Node* c2 = p_it.GetOrSpawnNode(p);
  c2->MakeTerminal(GameResult::WHITE_WON, 1.0f, Node::Terminal::Tablebase);

  ASSERT_TRUE(root->TryStartScoreUpdate());
  ASSERT_TRUE(p->TryStartScoreUpdate());
  ASSERT_TRUE(c2->TryStartScoreUpdate());

  DummyBackend backend;
  ClassicOptions opts;
  Search search(tree, &backend, std::make_unique<NullUciResponder>(), {},
                std::chrono::steady_clock::now(),
                std::make_unique<ChainedSearchStopper>(), false, false,
                opts.Get(), nullptr);
  SearchWorker worker(&search, search.GetParams());

  SearchBoundsTestPeer::Backup(worker, c2, 1.0f, 0.0f, 1.0f, 2);

  EXPECT_TRUE(p->IsTerminal());
  EXPECT_FALSE(p->IsTbTerminal());
  EXPECT_FLOAT_EQ(p->GetM(), 8.0f);
}

}  // namespace classic

#ifdef HAVE_DAG_CLASSIC
namespace dag_classic {

struct SearchBoundsTestPeer {
  static void Backup(SearchWorker& worker, const BackupPath& path,
                     const PositionHistory& history, int multivisit = 1) {
    SearchWorker::NodeToProcess ntp =
        SearchWorker::NodeToProcess::Visit(path, history);
    ntp.multivisit = multivisit;
    SharedMutex::Lock lock(worker.search_->nodes_mutex_);
    worker.DoBackupUpdateSingleNode(ntp);
  }
};

TEST(DagClassicBoundsTest, RefineTerminalAndSyncAcrossTransposition) {
  NodeTree tree;
  tree.ResetToPosition(ChessBoard::kStartposFen, {});
  Node* root = tree.GetCurrentHead();
  auto root_ln = std::make_shared<LowNode>(
      MoveList{MakeMove("e2e4"), MakeMove("d2d4")});
  root_ln->FinalizeScoreUpdate(0.0f, 0.0f, 12.0f, 1);
  root->SetLowNode(root_ln);
  root->TryStartScoreUpdate();
  root->FinalizeScoreUpdate(0.0f, 0.0f, 13.0f, 1);

  auto root_it = root->Edges();
  Node* gp1 = root_it.GetOrSpawnNode(root);
  ++root_it;
  Node* gp2 = root_it.GetOrSpawnNode(root);

  auto gp1_ln = std::make_shared<LowNode>(
      MoveList{MakeMove("e7e5"), MakeMove("c7c5")});
  gp1_ln->FinalizeScoreUpdate(0.0f, 0.0f, 10.0f, 1);
  gp1->SetLowNode(gp1_ln);
  gp1->TryStartScoreUpdate();
  gp1->FinalizeScoreUpdate(0.0f, 0.0f, 11.0f, 1);

  auto gp2_ln = std::make_shared<LowNode>(
      MoveList{MakeMove("d7d5"), MakeMove("g8f6")});
  gp2_ln->FinalizeScoreUpdate(0.0f, 0.0f, 10.0f, 1);
  gp2->SetLowNode(gp2_ln);
  gp2->TryStartScoreUpdate();
  gp2->FinalizeScoreUpdate(0.0f, 0.0f, 11.0f, 1);

  // Shared transposition LowNode pl reached via both p1 (under gp1) and p2
  // (under gp2).
  auto pl = std::make_shared<LowNode>(
      MoveList{MakeMove("d1h5"), MakeMove("f1c4")});
  pl->MakeTerminal(GameResult::WHITE_WON, 9.0f, Terminal::Tablebase);
  pl->FinalizeScoreUpdate(1.0f, 0.0f, 9.0f, 2);

  Node* p1 = gp1->Edges().GetOrSpawnNode(gp1);
  p1->SetLowNode(pl);
  p1->MakeTerminal(GameResult::BLACK_WON, 10.0f, Terminal::Tablebase);
  p1->TryStartScoreUpdate();
  p1->FinalizeScoreUpdate(-1.0f, 0.0f, 10.0f, 1);
  gp1_ln->FinalizeScoreUpdate(-1.0f, 0.0f, 10.0f, 1);
  gp1->TryStartScoreUpdate();
  gp1->FinalizeScoreUpdate(1.0f, 0.0f, 11.0f, 1);
  root_ln->FinalizeScoreUpdate(1.0f, 0.0f, 11.0f, 1);
  root->TryStartScoreUpdate();
  root->FinalizeScoreUpdate(-1.0f, 0.0f, 12.0f, 1);

  Node* p2 = gp2->Edges().GetOrSpawnNode(gp2);
  p2->SetLowNode(pl);
  p2->MakeTerminal(GameResult::BLACK_WON, 10.0f, Terminal::Tablebase);
  p2->TryStartScoreUpdate();
  p2->FinalizeScoreUpdate(-1.0f, 0.0f, 10.0f, 1);
  gp2_ln->FinalizeScoreUpdate(-1.0f, 0.0f, 10.0f, 1);
  gp2->TryStartScoreUpdate();
  gp2->FinalizeScoreUpdate(1.0f, 0.0f, 11.0f, 1);
  root_ln->FinalizeScoreUpdate(1.0f, 0.0f, 11.0f, 1);
  root->TryStartScoreUpdate();
  root->FinalizeScoreUpdate(-1.0f, 0.0f, 12.0f, 1);

  ASSERT_TRUE(pl->IsTransposition());

  auto pl_it = p1->Edges();
  Node* c1 = pl_it.GetOrSpawnNode(p1);
  c1->MakeTerminal(GameResult::WHITE_WON, 9.0f, Terminal::Tablebase);
  c1->TryStartScoreUpdate();
  c1->FinalizeScoreUpdate(1.0f, 0.0f, 9.0f, 1);

  ++pl_it;
  Node* c2 = pl_it.GetOrSpawnNode(p1);
  c2->MakeTerminal(GameResult::WHITE_WON, 2.0f, Terminal::EndOfGame);

  DummyBackend backend;
  DagClassicOptions opts;
  TranspositionTable tt;
  Search search(tree, &backend, std::make_unique<NullUciResponder>(), {},
                std::chrono::steady_clock::now(),
                std::make_unique<classic::ChainedSearchStopper>(), false, false,
                opts.Get(), &tt, nullptr);
  SearchWorker worker(&search, search.GetParams());

  // 1. Backup c2 through path root -> gp1 -> p1 -> c2.
  ASSERT_TRUE(root->TryStartScoreUpdate());
  ASSERT_TRUE(gp1->TryStartScoreUpdate());
  ASSERT_TRUE(p1->TryStartScoreUpdate());
  ASSERT_TRUE(c2->TryStartScoreUpdate());
  BackupPath path1 = {{root, 0, 0}, {gp1, 0, 0}, {p1, 0, 0}, {c2, 0, 0}};
  SearchBoundsTestPeer::Backup(worker, path1, tree.GetPositionHistory());

  EXPECT_EQ(pl->GetTerminalType(), Terminal::EndOfGame);
  EXPECT_FLOAT_EQ(pl->GetM(), 2.0f);
  EXPECT_FALSE(p1->IsTbTerminal());
  EXPECT_FLOAT_EQ(p1->GetM(), 3.0f);
  EXPECT_TRUE(pl->WLDMInvariantsHold());
  EXPECT_TRUE(p1->WLDMInvariantsHold());
  EXPECT_TRUE(gp1_ln->WLDMInvariantsHold());
  EXPECT_TRUE(gp1->WLDMInvariantsHold());
  EXPECT_TRUE(root_ln->WLDMInvariantsHold());
  EXPECT_TRUE(root->WLDMInvariantsHold());

  // p2 hasn't been visited yet, so it still holds old TB m=10.
  EXPECT_TRUE(p2->IsTbTerminal());
  EXPECT_FLOAT_EQ(p2->GetM(), 10.0f);

  // 2. Backup a terminal visit through transposition path root -> gp2 -> p2.
  ASSERT_TRUE(root->TryStartScoreUpdate());
  ASSERT_TRUE(gp2->TryStartScoreUpdate());
  ASSERT_TRUE(p2->TryStartScoreUpdate());
  BackupPath path2 = {{root, 0, 0}, {gp2, 0, 0}, {p2, 0, 0}};
  SearchBoundsTestPeer::Backup(worker, path2, tree.GetPositionHistory());

  EXPECT_FALSE(p2->IsTbTerminal());
  EXPECT_FLOAT_EQ(p2->GetM(), 3.0f);
  EXPECT_TRUE(p2->WLDMInvariantsHold());
  EXPECT_TRUE(gp2_ln->WLDMInvariantsHold());
  EXPECT_TRUE(gp2->WLDMInvariantsHold());
  EXPECT_TRUE(root_ln->WLDMInvariantsHold());
  EXPECT_TRUE(root->WLDMInvariantsHold());
}

TEST(DagClassicBoundsTest, LosingParentRecomputesLongestLoss) {
  NodeTree tree;
  tree.ResetToPosition(ChessBoard::kStartposFen, {});
  Node* root = tree.GetCurrentHead();
  auto root_ln = std::make_shared<LowNode>(MoveList{MakeMove("e2e4")});
  root_ln->FinalizeScoreUpdate(0.0f, 0.0f, 12.0f, 1);
  root->SetLowNode(root_ln);
  root->TryStartScoreUpdate();
  root->FinalizeScoreUpdate(0.0f, 0.0f, 13.0f, 1);

  // gp is losing for its mover (gp_ln BLACK_WON, m=10; gp WHITE_WON, m=11).
  // Its two children are p1 (m=10) and p2 (m=4).
  Node* gp = root->Edges().GetOrSpawnNode(root);
  auto gp_ln = std::make_shared<LowNode>(
      MoveList{MakeMove("e7e5"), MakeMove("d7d5")});
  gp_ln->MakeTerminal(GameResult::BLACK_WON, 10.0f, Terminal::EndOfGame);
  gp_ln->FinalizeScoreUpdate(-1.0f, 0.0f, 10.0f, 2);
  gp->SetLowNode(gp_ln);
  gp->MakeTerminal(GameResult::WHITE_WON, 11.0f, Terminal::EndOfGame);
  gp->TryStartScoreUpdate();
  gp->FinalizeScoreUpdate(1.0f, 0.0f, 11.0f, 1);
  root_ln->FinalizeScoreUpdate(1.0f, 0.0f, 11.0f, 1);
  root->TryStartScoreUpdate();
  root->FinalizeScoreUpdate(-1.0f, 0.0f, 12.0f, 1);

  auto gp_it = gp->Edges();
  Node* p1 = gp_it.GetOrSpawnNode(gp);
  ++gp_it;
  Node* p2 = gp_it.GetOrSpawnNode(gp);

  p2->MakeTerminal(GameResult::BLACK_WON, 4.0f, Terminal::EndOfGame);
  p2->TryStartScoreUpdate();
  p2->FinalizeScoreUpdate(-1.0f, 0.0f, 4.0f, 1);

  auto p1_ln = std::make_shared<LowNode>(
      MoveList{MakeMove("d1h5"), MakeMove("f1c4")});
  p1_ln->MakeTerminal(GameResult::WHITE_WON, 9.0f, Terminal::EndOfGame);
  p1_ln->FinalizeScoreUpdate(1.0f, 0.0f, 9.0f, 1);
  p1->SetLowNode(p1_ln);
  p1->MakeTerminal(GameResult::BLACK_WON, 10.0f, Terminal::EndOfGame);
  p1->TryStartScoreUpdate();
  p1->FinalizeScoreUpdate(-1.0f, 0.0f, 10.0f, 1);

  auto p1_it = p1->Edges();
  Node* c1 = p1_it.GetOrSpawnNode(p1);
  c1->MakeTerminal(GameResult::WHITE_WON, 9.0f, Terminal::EndOfGame);
  c1->TryStartScoreUpdate();
  c1->FinalizeScoreUpdate(1.0f, 0.0f, 9.0f, 1);

  ++p1_it;
  Node* c2 = p1_it.GetOrSpawnNode(p1);
  c2->MakeTerminal(GameResult::WHITE_WON, 1.0f, Terminal::EndOfGame);

  DummyBackend backend;
  DagClassicOptions opts;
  TranspositionTable tt;
  Search search(tree, &backend, std::make_unique<NullUciResponder>(), {},
                std::chrono::steady_clock::now(),
                std::make_unique<classic::ChainedSearchStopper>(), false, false,
                opts.Get(), &tt, nullptr);
  SearchWorker worker(&search, search.GetParams());

  ASSERT_TRUE(root->TryStartScoreUpdate());
  ASSERT_TRUE(gp->TryStartScoreUpdate());
  ASSERT_TRUE(p1->TryStartScoreUpdate());
  ASSERT_TRUE(c2->TryStartScoreUpdate());
  BackupPath path = {{root, 0, 0}, {gp, 0, 0}, {p1, 0, 0}, {c2, 0, 0}};
  SearchBoundsTestPeer::Backup(worker, path, tree.GetPositionHistory());

  // p1 shortens from 10 to 2; gp's longest loss among {p1(2), p2(4)} is now 4!
  EXPECT_FLOAT_EQ(p1_ln->GetM(), 1.0f);
  EXPECT_FLOAT_EQ(p1->GetM(), 2.0f);
  EXPECT_FLOAT_EQ(gp_ln->GetM(), 4.0f);
  EXPECT_FLOAT_EQ(gp->GetM(), 5.0f);
  EXPECT_TRUE(gp_ln->WLDMInvariantsHold());
  EXPECT_TRUE(gp->WLDMInvariantsHold());
  EXPECT_TRUE(root_ln->WLDMInvariantsHold());
  EXPECT_TRUE(root->WLDMInvariantsHold());
}

}  // namespace dag_classic
#endif
}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
