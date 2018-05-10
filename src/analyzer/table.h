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

#include <map>
#include <string>
#include <vector>

namespace lczero {

class Table3d {
 public:
  void Add3dVal(const std::string& col, const std::string& row,
                const std::string& metric, const std::string& val);

  void AddRowVal(const std::string& row, const std::string& metric,
                 const std::string& val);

  void AddColVal(const std::string& col, const std::string& metric,
                 const std::string& val);

  std::vector<std::vector<std::string>> RenderTable(
      const std::vector<std::string>& cols,
      const std::vector<std::string>& rows,
      const std::vector<std::string>& attrs,
      const std::vector<std::string>& row_attrs = {},
      const std::vector<std::string>& col_attrs = {});

 private:
  using AttrToVal = std::map<std::string, std::string>;
  using RowAttrToVal = std::map<std::string, AttrToVal>;
  using ColRowAttrToVal = std::map<std::string, RowAttrToVal>;

  ColRowAttrToVal data_3d_;
  RowAttrToVal data_row_;
  RowAttrToVal data_col_;
};

}  // namespace lczero