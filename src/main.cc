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

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#include <boost/asio.hpp>
#include "benchmark/benchmark.h"
#include "chess/board.h"
#include "engine.h"
#include "selfplay/loop.h"
#include "utils/commandline.h"
#include "utils/logging.h"
#include "version.h"

int main(int argc, const char** argv) {
  LOGFILE << "Lc0 started.";
  CERR << "       _";
  CERR << "|   _ | |";
  CERR << "|_ |_ |_| v" << GetVersionStr() << " built " << __DATE__;
  using namespace lczero;

  InitializeMagicBitboards();

  CommandLine::Init(argc, argv);
  CommandLine::RegisterMode("uci", "(default) Act as UCI engine");
  CommandLine::RegisterMode("selfplay", "Play games with itself");
  CommandLine::RegisterMode("benchmark", "Quick benchmark");

  if (CommandLine::ConsumeCommand("selfplay")) {
    // Selfplay mode.
    SelfPlayLoop loop;
    loop.RunLoop();
  } else if (CommandLine::ConsumeCommand("benchmark")) {
    // Benchmark mode.
    Benchmark benchmark;
    benchmark.Run();
  } else {
    // Consuming optional "server" mode.
    CommandLine::ConsumeCommand("server");
    boost::asio::io_context io_context;
    boost::asio::ip::tcp::acceptor acceptor(
        io_context,
        boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), 45312));
    std::vector<std::thread> threads;
    for (;;) {
      boost::asio::ip::tcp::iostream stream;
      acceptor.accept(*stream.rdbuf());
      threads.emplace_back([stream{std::move(stream)}]() mutable {
        EngineLoop loop(std::move(stream));
        loop.RunLoop();
      });
    }
  }
}
