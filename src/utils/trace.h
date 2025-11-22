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

#include "trace_config.h"

#if USE_PERFETTO_TRACE
#include <perfetto.h>

PERFETTO_DEFINE_CATEGORIES(
    perfetto::Category("lc0").SetDescription("Leela Chess Zero"));
#endif

#if USE_NVTX_TRACE
#include <nvtx3/nvtx3.hpp>
#endif

namespace lczero {
#if USE_PERFETTO_TRACE
#define LCTRACE_DECLARE_CATEGORIES PERFETTO_TRACK_EVENT_STATIC_STORAGE();

#define LCTRACE_INITIALIZE                     \
  do {                                         \
    perfetto::TracingInitArgs args;            \
    args.backends |= perfetto::kSystemBackend; \
    perfetto::Tracing::Initialize(args);       \
    perfetto::TrackEvent::Register();          \
  } while (false)

#define LCTRACE_FUNCTION_SCOPE \
  const auto& name = __func__; \
  TRACE_EVENT("lc0", name)

#elif USE_NVTX_TRACE
#define LCTRACE_DECLARE_CATEGORIES /* nop */
#define LCTRACE_INITIALIZE         /* nop */
struct lc0_domain {
  static constexpr char name[] = "lc0";
};
#define LCTRACE_FUNCTION_SCOPE NVTX3_FUNC_RANGE_IN(lc0_domain)
#else

#define LCTRACE_DECLARE_CATEGORIES
#define LCTRACE_INITIALIZE
#define LCTRACE_FUNCTION_SCOPE
#endif
}  // namespace lczero
