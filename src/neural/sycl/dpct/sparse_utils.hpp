//==---- sparse_utils.hpp -------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_SPARSE_UTILS_HPP__
#define __DPCT_SPARSE_UTILS_HPP__

#include "lib_common_utils.hpp"
#include <oneapi/mkl.hpp>
#include <sycl/sycl.hpp>

namespace dpct {
namespace sparse {
/// Describes properties of a sparse matrix.
/// The properties are matrix type, diag, uplo and index base.
class matrix_info {
public:
  /// Matrix types are:
  /// ge: General matrix
  /// sy: Symmetric matrix
  /// he: Hermitian matrix
  /// tr: Triangular matrix
  enum class matrix_type : int { ge = 0, sy, he, tr };

  auto get_matrix_type() const { return _matrix_type; }
  auto get_diag() const { return _diag; }
  auto get_uplo() const { return _uplo; }
  auto get_index_base() const { return _index_base; }
  void set_matrix_type(matrix_type mt) { _matrix_type = mt; }
  void set_diag(oneapi::mkl::diag d) { _diag = d; }
  void set_uplo(oneapi::mkl::uplo u) { _uplo = u; }
  void set_index_base(oneapi::mkl::index_base ib) { _index_base = ib; }

private:
  matrix_type _matrix_type = matrix_type::ge;
  oneapi::mkl::diag _diag = oneapi::mkl::diag::nonunit;
  oneapi::mkl::uplo _uplo = oneapi::mkl::uplo::upper;
  oneapi::mkl::index_base _index_base = oneapi::mkl::index_base::zero;
};

/// Computes a CSR format sparse matrix-dense vector product.
/// y = alpha * op(A) * x + beta * y
/// \param [in] queue The queue where the routine should be executed. It must
/// have the in_order property when using the USM mode.
/// \param [in] trans The operation applied to the matrix A.
/// \param [in] num_rows Number of rows of the matrix A.
/// \param [in] num_cols Number of columns of the matrix A.
/// \param [in] alpha Scaling factor for the matrix A.
/// \param [in] info Matrix info of the matrix A.
/// \param [in] val An array containing the non-zero elements of the matrix A.
/// \param [in] row_ptr An array of length \p num_rows + 1.
/// \param [in] col_ind An array containing the column indices in index-based
/// numbering.
/// \param [in] x Data of the vector x.
/// \param [in] beta Scaling factor for the vector x.
/// \param [in, out] y Data of the vector y.
template <typename T>
void csrmv(sycl::queue &queue, oneapi::mkl::transpose trans, int num_rows,
           int num_cols, const T *alpha,
           const std::shared_ptr<matrix_info> info, const T *val,
           const int *row_ptr, const int *col_ind, const T *x, const T *beta,
           T *y) {
#ifndef __INTEL_MKL__
  throw std::runtime_error("The oneAPI Math Kernel Library (oneMKL) Interfaces "
                           "Project does not support this API.");
#else
  using Ty = typename dpct::DataType<T>::T2;
  auto alpha_value =
      dpct::detail::get_value(reinterpret_cast<const Ty *>(alpha), queue);
  auto beta_value =
      dpct::detail::get_value(reinterpret_cast<const Ty *>(beta), queue);

  oneapi::mkl::sparse::matrix_handle_t *sparse_matrix_handle =
      new oneapi::mkl::sparse::matrix_handle_t;
  oneapi::mkl::sparse::init_matrix_handle(sparse_matrix_handle);
  auto data_row_ptr = dpct::detail::get_memory(const_cast<int *>(row_ptr));
  auto data_col_ind = dpct::detail::get_memory(const_cast<int *>(col_ind));
  auto data_val =
      dpct::detail::get_memory(reinterpret_cast<Ty *>(const_cast<T *>(val)));
  oneapi::mkl::sparse::set_csr_data(queue, *sparse_matrix_handle, num_rows,
                                    num_cols, info->get_index_base(),
                                    data_row_ptr, data_col_ind, data_val);

  auto data_x =
      dpct::detail::get_memory(reinterpret_cast<Ty *>(const_cast<T *>(x)));
  auto data_y = dpct::detail::get_memory(reinterpret_cast<Ty *>(y));
  switch (info->get_matrix_type()) {
  case matrix_info::matrix_type::ge: {
    oneapi::mkl::sparse::optimize_gemv(queue, trans, *sparse_matrix_handle);
    oneapi::mkl::sparse::gemv(queue, trans, alpha_value, *sparse_matrix_handle,
                              data_x, beta_value, data_y);
    break;
  }
  case matrix_info::matrix_type::sy: {
    oneapi::mkl::sparse::symv(queue, info->get_uplo(), alpha_value,
                              *sparse_matrix_handle, data_x, beta_value,
                              data_y);
    break;
  }
  case matrix_info::matrix_type::tr: {
    oneapi::mkl::sparse::optimize_trmv(queue, info->get_uplo(), trans,
                                       info->get_diag(), *sparse_matrix_handle);
    oneapi::mkl::sparse::trmv(queue, info->get_uplo(), trans, info->get_diag(),
                              alpha_value, *sparse_matrix_handle, data_x,
                              beta_value, data_y);
    break;
  }
  default:
    throw std::runtime_error(
        "the spmv does not support matrix_info::matrix_type::he");
  }

  sycl::event e =
      oneapi::mkl::sparse::release_matrix_handle(queue, sparse_matrix_handle);
  queue.submit([&](sycl::handler &cgh) {
    cgh.depends_on(e);
    cgh.host_task([=] { delete sparse_matrix_handle; });
  });
#endif
}

/// Computes a CSR format sparse matrix-dense matrix product.
/// C = alpha * op(A) * B + beta * C
/// \param [in] queue The queue where the routine should be executed. It must
/// have the in_order property when using the USM mode.
/// \param [in] trans The operation applied to the matrix A.
/// \param [in] sparse_rows Number of rows of the matrix A.
/// \param [in] dense_cols Number of columns of the matrix B or C.
/// \param [in] sparse_cols Number of columns of the matrix A.
/// \param [in] alpha Scaling factor for the matrix A.
/// \param [in] info Matrix info of the matrix A.
/// \param [in] val An array containing the non-zero elements of the matrix A.
/// \param [in] row_ptr An array of length \p num_rows + 1.
/// \param [in] col_ind An array containing the column indices in index-based
/// numbering.
/// \param [in] b Data of the matrix B.
/// \param [in] ldb Leading dimension of the matrix B.
/// \param [in] beta Scaling factor for the matrix B.
/// \param [in, out] c Data of the matrix C.
/// \param [in] ldc Leading dimension of the matrix C.
template <typename T>
void csrmm(sycl::queue &queue, oneapi::mkl::transpose trans, int sparse_rows,
           int dense_cols, int sparse_cols, const T *alpha,
           const std::shared_ptr<matrix_info> info, const T *val,
           const int *row_ptr, const int *col_ind, const T *b, int ldb,
           const T *beta, T *c, int ldc) {
#ifndef __INTEL_MKL__
  throw std::runtime_error("The oneAPI Math Kernel Library (oneMKL) Interfaces "
                           "Project does not support this API.");
#else
  using Ty = typename dpct::DataType<T>::T2;
  auto alpha_value =
      dpct::detail::get_value(reinterpret_cast<const Ty *>(alpha), queue);
  auto beta_value =
      dpct::detail::get_value(reinterpret_cast<const Ty *>(beta), queue);

  oneapi::mkl::sparse::matrix_handle_t *sparse_matrix_handle =
      new oneapi::mkl::sparse::matrix_handle_t;
  oneapi::mkl::sparse::init_matrix_handle(sparse_matrix_handle);
  auto data_row_ptr = dpct::detail::get_memory(const_cast<int *>(row_ptr));
  auto data_col_ind = dpct::detail::get_memory(const_cast<int *>(col_ind));
  auto data_val =
      dpct::detail::get_memory(reinterpret_cast<Ty *>(const_cast<T *>(val)));
  oneapi::mkl::sparse::set_csr_data(queue, *sparse_matrix_handle, sparse_rows,
                                    sparse_cols, info->get_index_base(),
                                    data_row_ptr, data_col_ind, data_val);

  auto data_b =
      dpct::detail::get_memory(reinterpret_cast<Ty *>(const_cast<T *>(b)));
  auto data_c = dpct::detail::get_memory(reinterpret_cast<Ty *>(c));
  switch (info->get_matrix_type()) {
  case matrix_info::matrix_type::ge: {
    oneapi::mkl::sparse::gemm(queue, oneapi::mkl::layout::row_major, trans,
                              oneapi::mkl::transpose::nontrans, alpha_value,
                              *sparse_matrix_handle, data_b, dense_cols, ldb,
                              beta_value, data_c, ldc);
    break;
  }
  default:
    throw std::runtime_error(
        "the csrmm does not support matrix_info::matrix_type::sy, "
        "matrix_info::matrix_type::tr and matrix_info::matrix_type::he");
  }

  sycl::event e =
      oneapi::mkl::sparse::release_matrix_handle(queue, sparse_matrix_handle);
  queue.submit([&](sycl::handler &cgh) {
    cgh.depends_on(e);
    cgh.host_task([=] { delete sparse_matrix_handle; });
  });
#endif
}

#ifdef __INTEL_MKL__ // The oneMKL Interfaces Project does not support this.
/// Saving the optimization information for solving a system of linear
/// equations.
class optimize_info {
public:
  /// Constructor
  optimize_info() { oneapi::mkl::sparse::init_matrix_handle(&_matrix_handle); }
  /// Destructor
  ~optimize_info() {
    oneapi::mkl::sparse::release_matrix_handle(get_default_queue(),
                                               &_matrix_handle, _deps)
        .wait();
  }
  /// Add dependency for the destructor.
  /// \param [in] e The event which the destructor depends on.
  void add_dependency(sycl::event e) { _deps.push_back(e); }
  /// Get the internal saved matrix handle.
  /// \return Returns the matrix handle.
  oneapi::mkl::sparse::matrix_handle_t get_matrix_handle() const noexcept {
    return _matrix_handle;
  }

private:
  oneapi::mkl::sparse::matrix_handle_t _matrix_handle = nullptr;
  std::vector<sycl::event> _deps;
};
#endif

#ifdef __INTEL_MKL__ // The oneMKL Interfaces Project does not support this.
/// Performs internal optimizations for solving a system of linear equations for
/// a CSR format sparse matrix.
/// \param [in] queue The queue where the routine should be executed. It must
/// have the in_order property when using the USM mode.
/// \param [in] trans The operation applied to the sparse matrix.
/// \param [in] row_col Number of rows of the sparse matrix.
/// \param [in] info Matrix info of the sparse matrix.
/// \param [in] val An array containing the non-zero elements of the sparse matrix.
/// \param [in] row_ptr An array of length \p num_rows + 1.
/// \param [in] col_ind An array containing the column indices in index-based
/// numbering.
/// \param [out] optimize_info The result of the optimizations.
template <typename T>
void optimize_csrsv(sycl::queue &queue, oneapi::mkl::transpose trans,
                    int row_col, const std::shared_ptr<matrix_info> info,
                    const T *val, const int *row_ptr, const int *col_ind,
                    std::shared_ptr<optimize_info> optimize_info) {
  using Ty = typename dpct::DataType<T>::T2;
  auto data_row_ptr = dpct::detail::get_memory(const_cast<int *>(row_ptr));
  auto data_col_ind = dpct::detail::get_memory(const_cast<int *>(col_ind));
  auto data_val =
      dpct::detail::get_memory(reinterpret_cast<Ty *>(const_cast<T *>(val)));
  oneapi::mkl::sparse::set_csr_data(queue, optimize_info->get_matrix_handle(),
                                    row_col, row_col, info->get_index_base(),
                                    data_row_ptr, data_col_ind, data_val);
  if (info->get_matrix_type() != matrix_info::matrix_type::tr)
    return;
#ifndef DPCT_USM_LEVEL_NONE
  sycl::event e;
  e =
#endif
      oneapi::mkl::sparse::optimize_trsv(queue, info->get_uplo(), trans,
                                         info->get_diag(),
                                         optimize_info->get_matrix_handle());
#ifndef DPCT_USM_LEVEL_NONE
  optimize_info->add_dependency(e);
#endif
}
#endif

class sparse_matrix_desc;

using sparse_matrix_desc_t = std::shared_ptr<sparse_matrix_desc>;

/// Structure for describe a dense vector
class dense_vector_desc {
public:
  dense_vector_desc(std::int64_t ele_num, void *value,
                    library_data_t value_type)
      : _ele_num(ele_num), _value(value), _value_type(value_type) {}
  void get_desc(std::int64_t *ele_num, const void **value,
                library_data_t *value_type) const noexcept {
    *ele_num = _ele_num;
    *value = _value;
    *value_type = _value_type;
  }
  void get_desc(std::int64_t *ele_num, void **value,
                library_data_t *value_type) const noexcept {
    get_desc(ele_num, const_cast<const void **>(value), value_type);
  }
  void *get_value() const noexcept { return _value; }
  void set_value(void *value) { _value = value; }

private:
  std::int64_t _ele_num;
  void *_value;
  library_data_t _value_type;
};

/// Structure for describe a dense matrix
class dense_matrix_desc {
public:
  dense_matrix_desc(std::int64_t row_num, std::int64_t col_num,
                    std::int64_t leading_dim, void *value,
                    library_data_t value_type, oneapi::mkl::layout layout)
      : _row_num(row_num), _col_num(col_num), _leading_dim(leading_dim),
        _value(value), _value_type(value_type), _layout(layout) {}
  void get_desc(std::int64_t *row_num, std::int64_t *col_num,
                std::int64_t *leading_dim, void **value,
                library_data_t *value_type,
                oneapi::mkl::layout *layout) const noexcept {
    *row_num = _row_num;
    *col_num = _col_num;
    *leading_dim = _leading_dim;
    *value = _value;
    *value_type = _value_type;
    *layout = _layout;
  }
  void *get_value() const noexcept { return _value; }
  void set_value(void *value) { _value = value; }
  std::int64_t get_col_num() const noexcept { return _col_num; }
  std::int64_t get_leading_dim() const noexcept { return _leading_dim; }
  oneapi::mkl::layout get_layout() const noexcept { return _layout; }

private:
  std::int64_t _row_num;
  std::int64_t _col_num;
  std::int64_t _leading_dim;
  void *_value;
  library_data_t _value_type;
  oneapi::mkl::layout _layout;
};

/// Sparse matrix data format
enum matrix_format : int {
  csr = 1,
};

/// Sparse matrix attribute
enum matrix_attribute : int { uplo = 0, diag };

#ifdef __INTEL_MKL__ // The oneMKL Interfaces Project does not support this.
/// Structure for describe a sparse matrix
class sparse_matrix_desc {
public:
  /// Constructor
  /// \param [out] desc The descriptor to be created
  /// \param [in] row_num Number of rows of the sparse matrix.
  /// \param [in] col_num Number of colums of the sparse matrix.
  /// \param [in] nnz Non-zero elements in the sparse matrix.
  /// \param [in] row_ptr An array of length \p row_num + 1.
  /// \param [in] col_ind An array containing the column indices in index-based
  /// numbering.
  /// \param [in] value An array containing the non-zero elements of the sparse matrix.
  /// \param [in] row_ptr_type Data type of the \p row_ptr .
  /// \param [in] col_ind_type Data type of the \p col_ind .
  /// \param [in] base Indicates how input arrays are indexed.
  /// \param [in] value_type Data type of the \p value .
  /// \param [in] data_format The matrix data format.
  sparse_matrix_desc(std::int64_t row_num, std::int64_t col_num,
                     std::int64_t nnz, void *row_ptr, void *col_ind,
                     void *value, library_data_t row_ptr_type,
                     library_data_t col_ind_type, oneapi::mkl::index_base base,
                     library_data_t value_type, matrix_format data_format)
      : _row_num(row_num), _col_num(col_num), _nnz(nnz), _row_ptr(row_ptr),
        _col_ind(col_ind), _value(value), _row_ptr_type(row_ptr_type),
        _col_ind_type(col_ind_type), _base(base), _value_type(value_type),
        _data_format(data_format) {
    if (_data_format != matrix_format::csr) {
      throw std::runtime_error("the sparse matrix data format is unsupported");
    }
    oneapi::mkl::sparse::init_matrix_handle(&_matrix_handle);
    construct();
  }
  /// Destructor
  ~sparse_matrix_desc() {
    oneapi::mkl::sparse::release_matrix_handle(get_default_queue(),
                                               &_matrix_handle, _deps)
        .wait();
  }

  /// Add dependency for the destroy method.
  /// \param [in] e The event which the destroy method depends on.
  void add_dependency(sycl::event e) { _deps.push_back(e); }
  /// Get the internal saved matrix handle.
  /// \return Returns the matrix handle.
  oneapi::mkl::sparse::matrix_handle_t get_matrix_handle() const noexcept {
    return _matrix_handle;
  }
  /// Get the values saved in the descriptor
  /// \param [out] row_num Number of rows of the sparse matrix.
  /// \param [out] col_num Number of colums of the sparse matrix.
  /// \param [out] nnz Non-zero elements in the sparse matrix.
  /// \param [out] row_ptr An array of length \p row_num + 1.
  /// \param [out] col_ind An array containing the column indices in index-based
  /// numbering.
  /// \param [out] value An array containing the non-zero elements of the sparse matrix.
  /// \param [out] row_ptr_type Data type of the \p row_ptr .
  /// \param [out] col_ind_type Data type of the \p col_ind .
  /// \param [out] base Indicates how input arrays are indexed.
  /// \param [out] value_type Data type of the \p value .
  void get_desc(int64_t *row_num, int64_t *col_num, int64_t *nnz,
                void **row_ptr, void **col_ind, void **value,
                library_data_t *row_ptr_type, library_data_t *col_ind_type,
                oneapi::mkl::index_base *base,
                library_data_t *value_type) const noexcept {
    *row_num = _row_num;
    *col_num = _col_num;
    *nnz = _nnz;
    *row_ptr = _row_ptr;
    *col_ind = _col_ind;
    *value = _value;
    *row_ptr_type = _row_ptr_type;
    *col_ind_type = _col_ind_type;
    *base = _base;
    *value_type = _value_type;
  }
  /// Get the sparse matrix data format of this descriptor
  /// \param [out] format The matrix data format result
  void get_format(matrix_format *data_format) const noexcept {
    *data_format = _data_format;
  }
  /// Get the index base of this descriptor
  /// \param [out] base The index base result
  void get_base(oneapi::mkl::index_base *base) const noexcept { *base = _base; }
  /// Get the value pointer of this descriptor
  /// \param [out] value The value pointer result
  void get_value(void **value) const noexcept { *value = _value; }
  /// Set the value pointer of this descriptor
  /// \param [in] value The input value pointer
  void set_value(void *value) {
    // Assume the new data is different from the old data
    _value = value;
    construct();
  }
  /// Get the size of the sparse matrix
  /// \param [out] row_num Number of rows of the sparse matrix.
  /// \param [out] col_num Number of colums of the sparse matrix.
  /// \param [out] nnz Non-zero elements in the sparse matrix.
  void get_size(int64_t *row_num, int64_t *col_num,
                int64_t *nnz) const noexcept {
    *row_num = _row_num;
    *col_num = _col_num;
    *nnz = _nnz;
  }
  /// Set the sparse matrix attribute
  /// \param [in] attribute The attribute type
  /// \param [in] data The attribute value
  /// \param [in] data_size The data size of the attribute value
  void set_attribute(matrix_attribute attribute, const void *data,
                     size_t data_size) {
    if (attribute == matrix_attribute::diag) {
      const oneapi::mkl::diag *diag_ptr =
          reinterpret_cast<const oneapi::mkl::diag *>(data);
      if (*diag_ptr == oneapi::mkl::diag::unit) {
        _diag = oneapi::mkl::diag::unit;
      } else if (*diag_ptr == oneapi::mkl::diag::nonunit) {
        _diag = oneapi::mkl::diag::nonunit;
      } else {
        throw std::runtime_error("unsupported diag value");
      }
    } else if (attribute == matrix_attribute::uplo) {
      const oneapi::mkl::uplo *uplo_ptr =
          reinterpret_cast<const oneapi::mkl::uplo *>(data);
      if (*uplo_ptr == oneapi::mkl::uplo::upper) {
        _uplo = oneapi::mkl::uplo::upper;
      } else if (*uplo_ptr == oneapi::mkl::uplo::lower) {
        _uplo = oneapi::mkl::uplo::lower;
      } else {
        throw std::runtime_error("unsupported uplo value");
      }
    } else {
      throw std::runtime_error("unsupported attribute");
    }
  }
  /// Get the sparse matrix attribute
  /// \param [out] attribute The attribute type
  /// \param [out] data The attribute value
  /// \param [out] data_size The data size of the attribute value
  void get_attribute(matrix_attribute attribute, void *data,
                     size_t data_size) const {
    if (attribute == matrix_attribute::diag) {
      oneapi::mkl::diag *diag_ptr = reinterpret_cast<oneapi::mkl::diag *>(data);
      if (_diag.has_value()) {
        *diag_ptr = _diag.value();
      } else {
        *diag_ptr = oneapi::mkl::diag::nonunit;
      }
    } else if (attribute == matrix_attribute::uplo) {
      oneapi::mkl::uplo *uplo_ptr = reinterpret_cast<oneapi::mkl::uplo *>(data);
      if (_uplo.has_value()) {
        *uplo_ptr = _uplo.value();
      } else {
        *uplo_ptr = oneapi::mkl::uplo::lower;
      }
    } else {
      throw std::runtime_error("unsupported attribute");
    }
  }
  /// Set the pointers for describing the sparse matrix
  /// \param [in] row_ptr An array of length \p row_num + 1.
  /// \param [in] col_ind An array containing the column indices in index-based
  /// numbering.
  /// \param [in] value An array containing the non-zero elements of the sparse matrix.
  void set_pointers(void *row_ptr, void *col_ind, void *value) {
    // Assume the new data is different from the old data
    _row_ptr = row_ptr;
    _col_ind = col_ind;
    _value = value;
    construct();
  }

  /// Get the diag attribute
  /// \return diag value
  std::optional<oneapi::mkl::diag> get_diag() const noexcept { return _diag; }
  /// Get the uplo attribute
  /// \return uplo value
  std::optional<oneapi::mkl::uplo> get_uplo() const noexcept { return _uplo; }

private:
  template <typename index_t, typename value_t> void set_data() {
    auto data_row_ptr =
        dpct::detail::get_memory(reinterpret_cast<index_t *>(_row_ptr));
    auto data_col_ind =
        dpct::detail::get_memory(reinterpret_cast<index_t *>(_col_ind));
    auto data_value =
        dpct::detail::get_memory(reinterpret_cast<value_t *>(_value));
    oneapi::mkl::sparse::set_csr_data(get_default_queue(), _matrix_handle,
                                      _row_num, _col_num, _base, data_row_ptr,
                                      data_col_ind, data_value);
    get_default_queue().wait();
  }
  void construct() {
    std::uint64_t key = dpct::detail::get_type_combination_id(
        _row_ptr_type, _col_ind_type, _value_type);
    switch (key) {
    case dpct::detail::get_type_combination_id(library_data_t::real_int32,
                                               library_data_t::real_int32,
                                               library_data_t::real_float): {
      set_data<std::int32_t, float>();
      break;
    }
    case dpct::detail::get_type_combination_id(library_data_t::real_int32,
                                               library_data_t::real_int32,
                                               library_data_t::real_double): {
      set_data<std::int32_t, double>();
      break;
    }
    case dpct::detail::get_type_combination_id(library_data_t::real_int32,
                                               library_data_t::real_int32,
                                               library_data_t::complex_float): {
      set_data<std::int32_t, std::complex<float>>();
      break;
    }
    case dpct::detail::get_type_combination_id(
        library_data_t::real_int32, library_data_t::real_int32,
        library_data_t::complex_double): {
      set_data<std::int32_t, std::complex<double>>();
      break;
    }
    case dpct::detail::get_type_combination_id(library_data_t::real_int64,
                                               library_data_t::real_int64,
                                               library_data_t::real_float): {
      set_data<std::int64_t, float>();
      break;
    }
    case dpct::detail::get_type_combination_id(library_data_t::real_int64,
                                               library_data_t::real_int64,
                                               library_data_t::real_double): {
      set_data<std::int64_t, double>();
      break;
    }
    case dpct::detail::get_type_combination_id(library_data_t::real_int64,
                                               library_data_t::real_int64,
                                               library_data_t::complex_float): {
      set_data<std::int64_t, std::complex<float>>();
      break;
    }
    case dpct::detail::get_type_combination_id(
        library_data_t::real_int64, library_data_t::real_int64,
        library_data_t::complex_double): {
      set_data<std::int64_t, std::complex<double>>();
      break;
    }
    default:
      throw std::runtime_error("the combination of data type is unsupported");
    }
  }

  std::int64_t _row_num;
  std::int64_t _col_num;
  std::int64_t _nnz;
  void *_row_ptr;
  void *_col_ind;
  void *_value;
  library_data_t _row_ptr_type;
  library_data_t _col_ind_type;
  oneapi::mkl::index_base _base;
  library_data_t _value_type;
  oneapi::mkl::sparse::matrix_handle_t _matrix_handle = nullptr;
  std::vector<sycl::event> _deps;
  matrix_format _data_format;
  std::optional<oneapi::mkl::uplo> _uplo;
  std::optional<oneapi::mkl::diag> _diag;
};

namespace detail {
#ifdef DPCT_USM_LEVEL_NONE
#define SPARSE_CALL(X)                                                         \
  do {                                                                         \
    X;                                                                         \
  } while (0)
#else
#define SPARSE_CALL(X)                                                         \
  do {                                                                         \
    sycl::event e = X;                                                         \
    a->add_dependency(e);                                                      \
  } while (0)
#endif

template <typename Ty>
inline void spmv_impl(sycl::queue queue, oneapi::mkl::transpose trans,
                      const void *alpha, sparse_matrix_desc_t a,
                      std::shared_ptr<dense_vector_desc> x, const void *beta,
                      std::shared_ptr<dense_vector_desc> y) {
  auto alpha_value =
      dpct::detail::get_value(reinterpret_cast<const Ty *>(alpha), queue);
  auto beta_value =
      dpct::detail::get_value(reinterpret_cast<const Ty *>(beta), queue);
  auto data_x =
      dpct::detail::get_memory(reinterpret_cast<Ty *>(x->get_value()));
  auto data_y =
      dpct::detail::get_memory(reinterpret_cast<Ty *>(y->get_value()));
  if (a->get_diag().has_value() && a->get_uplo().has_value()) {
    oneapi::mkl::sparse::optimize_trmv(queue, a->get_uplo().value(), trans,
                                       a->get_diag().value(),
                                       a->get_matrix_handle());
    SPARSE_CALL(oneapi::mkl::sparse::trmv(
        queue, a->get_uplo().value(), trans, a->get_diag().value(), alpha_value,
        a->get_matrix_handle(), data_x, beta_value, data_y));
  } else {
    oneapi::mkl::sparse::optimize_gemv(queue, trans, a->get_matrix_handle());
    SPARSE_CALL(oneapi::mkl::sparse::gemv(queue, trans, alpha_value,
                                          a->get_matrix_handle(), data_x,
                                          beta_value, data_y));
  }
}

template <typename Ty>
inline void spmm_impl(sycl::queue queue, oneapi::mkl::transpose trans_a,
                      oneapi::mkl::transpose trans_b, const void *alpha,
                      sparse_matrix_desc_t a,
                      std::shared_ptr<dense_matrix_desc> b, const void *beta,
                      std::shared_ptr<dense_matrix_desc> c) {
  auto alpha_value =
      dpct::detail::get_value(reinterpret_cast<const Ty *>(alpha), queue);
  auto beta_value =
      dpct::detail::get_value(reinterpret_cast<const Ty *>(beta), queue);
  auto data_b =
      dpct::detail::get_memory(reinterpret_cast<Ty *>(b->get_value()));
  auto data_c =
      dpct::detail::get_memory(reinterpret_cast<Ty *>(c->get_value()));
  SPARSE_CALL(oneapi::mkl::sparse::gemm(
      queue, b->get_layout(), trans_a, trans_b, alpha_value,
      a->get_matrix_handle(), data_b, b->get_col_num(), b->get_leading_dim(),
      beta_value, data_c, c->get_leading_dim()));
}
#undef SPARSE_CALL
} // namespace detail

/// Computes a sparse matrix-dense vector product: y = alpha * op(a) * x + beta * y.
/// \param [in] queue The queue where the routine should be executed. It must
/// have the in_order property when using the USM mode.
/// \param [in] trans Specifies operation on input matrix.
/// \param [in] alpha Specifies the scalar alpha.
/// \param [in] a Specifies the sparse matrix a.
/// \param [in] x Specifies the dense vector x.
/// \param [in] beta Specifies the scalar beta.
/// \param [in, out] y Specifies the dense vector y.
/// \param [in] data_type Specifies the data type of \param a, \param x and \param y .
inline void spmv(sycl::queue queue, oneapi::mkl::transpose trans,
                 const void *alpha, sparse_matrix_desc_t a,
                 std::shared_ptr<dense_vector_desc> x, const void *beta,
                 std::shared_ptr<dense_vector_desc> y,
                 library_data_t data_type) {
  switch (data_type) {
  case library_data_t::real_float: {
    detail::spmv_impl<float>(queue, trans, alpha, a, x, beta, y);
    break;
  }
  case library_data_t::real_double: {
    detail::spmv_impl<double>(queue, trans, alpha, a, x, beta, y);
    break;
  }
  case library_data_t::complex_float: {
    detail::spmv_impl<std::complex<float>>(queue, trans, alpha, a, x, beta, y);
    break;
  }
  case library_data_t::complex_double: {
    detail::spmv_impl<std::complex<double>>(queue, trans, alpha, a, x, beta, y);
    break;
  }
  default:
    throw std::runtime_error("the combination of data type is unsupported");
  }
}

/// Computes a sparse matrix-dense matrix product: c = alpha * op(a) * op(b) + beta * c.
/// \param [in] queue The queue where the routine should be executed. It must
/// have the in_order property when using the USM mode.
/// \param [in] trans_a Specifies operation on input matrix a.
/// \param [in] trans_b Specifies operation on input matrix b.
/// \param [in] alpha Specifies the scalar alpha.
/// \param [in] a Specifies the sparse matrix a.
/// \param [in] b Specifies the dense matrix b.
/// \param [in] beta Specifies the scalar beta.
/// \param [in, out] c Specifies the dense matrix c.
/// \param [in] data_type Specifies the data type of \param a, \param b and \param c .
inline void spmm(sycl::queue queue, oneapi::mkl::transpose trans_a,
                 oneapi::mkl::transpose trans_b, const void *alpha,
                 sparse_matrix_desc_t a, std::shared_ptr<dense_matrix_desc> b,
                 const void *beta, std::shared_ptr<dense_matrix_desc> c,
                 library_data_t data_type) {
  if (b->get_layout() != c->get_layout())
    throw std::runtime_error("the layout of b and c are different");

  switch (data_type) {
  case library_data_t::real_float: {
    detail::spmm_impl<float>(queue, trans_a, trans_b, alpha, a, b, beta, c);
    break;
  }
  case library_data_t::real_double: {
    detail::spmm_impl<double>(queue, trans_a, trans_b, alpha, a, b, beta, c);
    break;
  }
  case library_data_t::complex_float: {
    detail::spmm_impl<std::complex<float>>(queue, trans_a, trans_b, alpha, a, b,
                                           beta, c);
    break;
  }
  case library_data_t::complex_double: {
    detail::spmm_impl<std::complex<double>>(queue, trans_a, trans_b, alpha, a,
                                            b, beta, c);
    break;
  }
  default:
    throw std::runtime_error("the combination of data type is unsupported");
  }
}
#endif
} // namespace sparse
} // namespace dpct

#endif // __DPCT_SPARSE_UTILS_HPP__
