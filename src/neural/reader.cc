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

#include "neural/reader.h"

#include "neural/encoder.h"
#include "utils/bititer.h"

namespace lczero {

InputPlanes PlanesFromTrainingData(const V6TrainingData& data) {
  InputPlanes result;
  for (int i = 0; i < 104; i++) {
    result.emplace_back();
    result.back().mask = ReverseBitsInBytes(data.planes[i]);
  }
  switch (data.input_format) {
    case pblczero::NetworkFormat::InputFormat::INPUT_CLASSICAL_112_PLANE: {
      result.emplace_back();
      result.back().mask = data.castling_us_ooo != 0 ? ~0LL : 0LL;
      result.emplace_back();
      result.back().mask = data.castling_us_oo != 0 ? ~0LL : 0LL;
      result.emplace_back();
      result.back().mask = data.castling_them_ooo != 0 ? ~0LL : 0LL;
      result.emplace_back();
      result.back().mask = data.castling_them_oo != 0 ? ~0LL : 0LL;
      break;
    }
    case pblczero::NetworkFormat::INPUT_112_WITH_CASTLING_PLANE:
    case pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION:
    case pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION_HECTOPLIES:
    case pblczero::NetworkFormat::
        INPUT_112_WITH_CANONICALIZATION_HECTOPLIES_ARMAGEDDON:
    case pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION_V2:
    case pblczero::NetworkFormat::
        INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON: {
      result.emplace_back();
      result.back().mask =
          data.castling_us_ooo |
          (static_cast<uint64_t>(data.castling_them_ooo) << 56);
      result.emplace_back();
      result.back().mask = data.castling_us_oo |
                           (static_cast<uint64_t>(data.castling_them_oo) << 56);
      // 2 empty planes in this format.
      result.emplace_back();
      result.emplace_back();
      break;
    }

    default:
      throw Exception("Unsupported input plane encoding " +
                      std::to_string(data.input_format));
  }
  result.emplace_back();
  auto typed_format =
      static_cast<pblczero::NetworkFormat::InputFormat>(data.input_format);
  if (IsCanonicalFormat(typed_format)) {
    result.back().mask = static_cast<uint64_t>(data.side_to_move_or_enpassant)
                         << 56;
  } else {
    result.back().mask = data.side_to_move_or_enpassant != 0 ? ~0LL : 0LL;
  }
  result.emplace_back();
  if (IsHectopliesFormat(typed_format)) {
    result.back().Fill(data.rule50_count / 100.0f);
  } else {
    result.back().Fill(data.rule50_count);
  }
  result.emplace_back();
  // Empty plane, except for canonical armageddon.
  if (IsCanonicalArmageddonFormat(typed_format) &&
      data.invariance_info >= 128) {
    result.back().SetAll();
  }
  result.emplace_back();
  // All ones plane.
  result.back().SetAll();
  if (IsCanonicalFormat(typed_format) && data.invariance_info != 0) {
    // Undo transformation here as it makes the calling code simpler.
    int transform = data.invariance_info;
    for (int i = 0; i <= result.size(); i++) {
      auto v = result[i].mask;
      if (v == 0 || v == ~0ULL) continue;
      if ((transform & TransposeTransform) != 0) {
        v = TransposeBitsInBytes(v);
      }
      if ((transform & MirrorTransform) != 0) {
        v = ReverseBytesInBytes(v);
      }
      if ((transform & FlipTransform) != 0) {
        v = ReverseBitsInBytes(v);
      }
      result[i].mask = v;
    }
  }
  return result;
}

TrainingDataReader::TrainingDataReader(std::string filename)
    : filename_(filename) {
  fin_ = gzopen(filename_.c_str(), "rb");
  if (!fin_) {
    throw Exception("Cannot open gzip file " + filename_);
  }
}

TrainingDataReader::~TrainingDataReader() { gzclose(fin_); }

bool TrainingDataReader::ReadChunk(V6TrainingData* data) {
  if (format_v6) {
    int read_size = gzread(fin_, reinterpret_cast<void*>(data), sizeof(*data));
    if (read_size < 0) throw Exception("Corrupt read.");
    return read_size == sizeof(*data);
  } else {
    size_t v6_extra = 48;
    size_t v5_extra = 16;
    size_t v4_extra = 16;
    size_t v3_size = sizeof(*data) - v4_extra - v5_extra - v6_extra;
    int read_size = gzread(fin_, reinterpret_cast<void*>(data), v3_size);
    if (read_size < 0) throw Exception("Corrupt read.");
    if (read_size != v3_size) return false;
    auto orig_version = data->version;
    switch (data->version) {
      case 3: {
        data->version = 4;
        // First convert 3 to 4 to reduce code duplication.
        char* v4_extra_start = reinterpret_cast<char*>(data) + v3_size;
        // Write 0 bytes for 16 extra bytes - corresponding to 4 floats of 0.0f.
        for (int i = 0; i < v4_extra; i++) {
          v4_extra_start[i] = 0;
        }
        // Deliberate fallthrough.
      }
      case 4: {
        // If actually 4, we need to read the additional data first.
        if (orig_version == 4) {
          read_size = gzread(
              fin_,
              reinterpret_cast<void*>(reinterpret_cast<char*>(data) + v3_size),
              v4_extra);
          if (read_size < 0) throw Exception("Corrupt read.");
          if (read_size != v4_extra) return false;
        }
        data->version = 5;
        char* data_ptr = reinterpret_cast<char*>(data);
        // Shift data after version back 4 bytes.
        memmove(data_ptr + 2 * sizeof(uint32_t), data_ptr + sizeof(uint32_t),
                v3_size + v4_extra - sizeof(uint32_t));
        data->input_format = pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE;
        data->root_m = 0.0f;
        data->best_m = 0.0f;
        data->plies_left = 0.0f;
        // Deliberate fallthrough.
      }
      case 5: {
        // If actually 5, we need to read the additional data first.
        if (orig_version == 5) {
          read_size = gzread(
              fin_,
              reinterpret_cast<void*>(reinterpret_cast<char*>(data) + v3_size),
              v4_extra + v5_extra);
          if (read_size < 0) throw Exception("Corrupt read.");
          if (read_size != v4_extra + v5_extra) return false;
        }
        data->version = 6;
        // Type of dummy was changed from signed to unsigned - which means -1 on
        // disk is read in as 255.
        if (data->dummy > 1 && data->dummy < 255) {
          throw Exception("Invalid result read in v5 data before upgrade.");
        }
        data->result_q =
            data->dummy == 255 ? -1.0f : (data->dummy == 0 ? 0.0f : 1.0f);
        data->result_d = data->dummy == 0 ? 1.0f : 0.0f;
        data->dummy = 0;
        data->played_q = 0.0f;
        data->played_d = 0.0f;
        data->played_m = 0.0f;
        // Mark orig as NaN since scripts further downstream already have to
        // handle that case.
        data->orig_q = std::numeric_limits<float>::quiet_NaN();
        data->orig_d = std::numeric_limits<float>::quiet_NaN();
        data->orig_m = std::numeric_limits<float>::quiet_NaN();
        data->visits = 0;
        data->played_idx = 0;
        data->best_idx = 0;
        data->policy_kld = 0.0f;
        data->reserved = 0;
        return true;
      }
      case 6: {
        format_v6 = true;
        read_size = gzread(
            fin_,
            reinterpret_cast<void*>(reinterpret_cast<char*>(data) + v3_size),
            v4_extra + v5_extra + v6_extra);
        if (read_size < 0) throw Exception("Corrupt read.");
        return read_size == v4_extra + v5_extra + v6_extra;
      }
    }
  }
}

}  // namespace lczero
