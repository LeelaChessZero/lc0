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

#include "analyzer/table.h"
#include <sstream>

namespace lczero {

void Table3d::Add3dVal(const std::string& col, const std::string& row,
                       const std::string& metric, const std::string& val) {
  data_3d_[col][row][metric] = val;
}

void Table3d::AddRowVal(const std::string& row, const std::string& metric,
                        const std::string& val) {
  data_row_[row][metric] = val;
}

void Table3d::AddColVal(const std::string& col, const std::string& metric,
                        const std::string& val) {
  data_col_[col][metric] = val;
}

namespace {
std::string BuildDimensionsStr(int cols, int rows, int attrs, int rattrs,
                               int cattrs) {
  std::ostringstream oss;
  oss << "(##" << cols << 'x' << rows << 'x' << attrs << 'x' << rattrs << 'x'
      << cattrs << ")";
  return oss.str();
}

}  // namespace

std::vector<std::vector<std::string>> Table3d::RenderTable(
    const std::vector<std::string>& cols, const std::vector<std::string>& rows,
    const std::vector<std::string>& attrs,
    const std::vector<std::string>& row_attrs,
    const std::vector<std::string>& col_attrs) {
  std::vector<std::vector<std::string>> lines;
  const bool has_row_attrs = !row_attrs.empty();

  std::vector<std::string> line;

  // Header.
  // Topleft cell contains format for appscript to know the table dimensions.
  line.push_back(BuildDimensionsStr(cols.size(), rows.size(), attrs.size(),
                                    row_attrs.size(), col_attrs.size()));
  for (int i = 0; i < (has_row_attrs ? 3 : 1); ++i) line.emplace_back();
  for (const auto& col : cols) line.emplace_back(col);
  lines.emplace_back(std::move(line));

  // Col-level values.
  for (const auto& attr : col_attrs) {
    line.clear();
    for (int i = 0; i < (has_row_attrs ? 3 : 1); ++i) line.emplace_back();
    line.push_back(attr + ":");
    for (const auto& col : cols) line.emplace_back(data_col_[col][attr]);
    lines.emplace_back(std::move(line));
  }

  // Main table.
  for (const auto& row : rows) {
    const int stripes = std::max(attrs.size(), row_attrs.size());
    for (int i = 0; i < stripes; ++i) {
      line.clear();
      if (i == 0) {
        line.push_back(row);
      } else {
        line.emplace_back();
      }
      if (has_row_attrs) {
        if (i < row_attrs.size()) {
          line.push_back(row_attrs[i] + ":");
          line.push_back(data_row_[row][row_attrs[i]]);
        } else {
          line.emplace_back();
          line.emplace_back();
        }
      }
      if (i < attrs.size()) {
        auto attr = attrs[i];
        line.emplace_back(attr + ":");
        for (const auto& col : cols)
          line.emplace_back(data_3d_[col][row][attr]);
      }
      lines.emplace_back(std::move(line));
    }
  }

  return lines;
}

}  // namespace lczero