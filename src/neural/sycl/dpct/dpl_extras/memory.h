//==---- memory.h ---------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_MEMORY_H__
#define __DPCT_MEMORY_H__

#include <sycl/sycl.hpp>

// Memory management section:
// device_pointer, device_reference, swap, device_iterator, malloc_device,
// device_new, free_device, device_delete
namespace dpct {
namespace detail {
template <typename T>
struct make_allocatable
{
  using type = T;
};

template <>
struct make_allocatable<void>
{
  using type = dpct::byte_t;
};

#if defined(__LIBSYCL_MAJOR_VERSION) && defined(__LIBSYCL_MINOR_VERSION) &&    \
    defined(__LIBSYCL_PATCH_VERSION)
#define _DPCT_LIBSYCL_VERSION                                                  \
  (__LIBSYCL_MAJOR_VERSION * 10000 + __LIBSYCL_MINOR_VERSION * 100 +           \
   __LIBSYCL_PATCH_VERSION)
#else
#define _DPCT_LIBSYCL_VERSION 0
#endif

template <typename _DataT>
using __buffer_allocator =
#if _DPCT_LIBSYCL_VERSION >= 60000
    sycl::buffer_allocator<typename make_allocatable<_DataT>::type>;
#else
    sycl::buffer_allocator;
#endif
} // namespace detail

#ifdef DPCT_USM_LEVEL_NONE
template <typename T, sycl::access_mode Mode = sycl::access_mode::read_write,
          typename Allocator = detail::__buffer_allocator<T>>
class device_pointer;
#else
template <typename T> class device_pointer;
#endif

template <typename T> struct device_reference {
  using pointer = device_pointer<T>;
  using value_type = T;
  template <typename OtherT>
  device_reference(const device_reference<OtherT> &input)
      : value(input.value) {}
  device_reference(const pointer &input) : value((*input).value) {}
  device_reference(value_type &input) : value(input) {}
  template <typename OtherT>
  device_reference &operator=(const device_reference<OtherT> &input) {
    value = input;
    return *this;
  };
  device_reference &operator=(const device_reference &input) {
    T val = input.value;
    value = val;
    return *this;
  };
  device_reference &operator=(const value_type &x) {
    value = x;
    return *this;
  };
  pointer operator&() const { return pointer(&value); };
  operator value_type() const { return T(value); }
  device_reference &operator++() {
    ++value;
    return *this;
  };
  device_reference &operator--() {
    --value;
    return *this;
  };
  device_reference operator++(int) {
    device_reference ref(*this);
    ++(*this);
    return ref;
  };
  device_reference operator--(int) {
    device_reference ref(*this);
    --(*this);
    return ref;
  };
  device_reference &operator+=(const T &input) {
    value += input;
    return *this;
  };
  device_reference &operator-=(const T &input) {
    value -= input;
    return *this;
  };
  device_reference &operator*=(const T &input) {
    value *= input;
    return *this;
  };
  device_reference &operator/=(const T &input) {
    value /= input;
    return *this;
  };
  device_reference &operator%=(const T &input) {
    value %= input;
    return *this;
  };
  device_reference &operator&=(const T &input) {
    value &= input;
    return *this;
  };
  device_reference &operator|=(const T &input) {
    value |= input;
    return *this;
  };
  device_reference &operator^=(const T &input) {
    value ^= input;
    return *this;
  };
  device_reference &operator<<=(const T &input) {
    value <<= input;
    return *this;
  };
  device_reference &operator>>=(const T &input) {
    value >>= input;
    return *this;
  };
  void swap(device_reference &input) {
    T tmp = (*this);
    *this = (input);
    input = (tmp);
  }
  T &value;
};

template <typename T>
void swap(device_reference<T> &x, device_reference<T> &y) {
  x.swap(y);
}

template <typename T> void swap(T &x, T &y) {
  T tmp = x;
  x = y;
  y = tmp;
}

namespace internal {
// struct for checking if iterator is heterogeneous or not
template <typename Iter,
          typename Void = void> // for non-heterogeneous iterators
struct is_hetero_iterator : std::false_type {};

template <typename Iter> // for heterogeneous iterators
struct is_hetero_iterator<
    Iter, typename std::enable_if<Iter::is_hetero::value, void>::type>
    : std::true_type {};
} // namespace internal

#ifdef DPCT_USM_LEVEL_NONE
template <typename T, sycl::access_mode Mode, typename Allocator>
class device_iterator;

template <typename ValueType, typename Allocator, typename Derived>
class device_pointer_base {
protected:
  sycl::buffer<ValueType, 1, Allocator> buffer;
  std::size_t idx;

public:
  using pointer = ValueType *;
  using difference_type = std::make_signed<std::size_t>::type;

  device_pointer_base(sycl::buffer<ValueType, 1> in, std::size_t i = 0)
      : buffer(in), idx(i) {}
#ifdef __USE_DPCT
  template <typename OtherT>
  device_pointer_base(OtherT *ptr)
      : buffer(
            dpct::detail::mem_mgr::instance()
                .translate_ptr(ptr)
                .buffer.template reinterpret<ValueType, 1>(sycl::range<1>(
                    dpct::detail::mem_mgr::instance().translate_ptr(ptr).size /
                    sizeof(ValueType)))),
        idx(ptr - (ValueType*)dpct::detail::mem_mgr::instance()
                .translate_ptr(ptr).alloc_ptr) {}
#endif
  device_pointer_base(const std::size_t count)
      : buffer(sycl::range<1>(count / sizeof(ValueType))), idx() {}
  // buffer has no default ctor we pass zero-range to create an empty buffer
  device_pointer_base() : buffer(sycl::range<1>(0)) {}
  device_pointer_base(const device_pointer_base &in)
      : buffer(in.buffer), idx(in.idx) {}
  pointer get() const {
    auto res =
        (const_cast<device_pointer_base *>(this)
             ->buffer.template get_access<sycl::access_mode::read_write>())
            .get_pointer();
    return res + idx;
  }
  operator ValueType *() {
    auto res = (buffer.template get_access<sycl::access_mode::read_write>())
                   .get_pointer();
    return res + idx;
  }
  operator ValueType *() const {
    auto res =
        (const_cast<device_pointer_base *>(this)
             ->buffer.template get_access<sycl::access_mode::read_write>())
            .get_pointer();
    return res + idx;
  }
  Derived operator+(difference_type forward) const {
    return Derived{buffer, idx + forward};
  }
  Derived operator-(difference_type backward) const {
    return Derived{buffer, idx - backward};
  }
  Derived operator++(int) {
    Derived p(buffer, idx);
    idx += 1;
    return p;
  }
  Derived operator--(int) {
    Derived p(buffer, idx);
    idx -= 1;
    return p;
  }
  difference_type operator-(const Derived &it) const { return idx - it.idx; }
  template <typename OtherIterator>
  typename std::enable_if<internal::is_hetero_iterator<OtherIterator>::value,
                          difference_type>::type
  operator-(const OtherIterator &it) const {
    return idx - std::distance(oneapi::dpl::begin(buffer), it);
  }

  std::size_t get_idx() const { return idx; } // required

  sycl::buffer<ValueType, 1, Allocator> get_buffer() {
    return buffer;
  } // required
};

template <typename T, sycl::access_mode Mode, typename Allocator>
class device_pointer
    : public device_pointer_base<T, Allocator,
                                 device_pointer<T, Mode, Allocator>> {
private:
  using base_type = device_pointer_base<T, Allocator, device_pointer>;

public:
  using value_type = T;
  using difference_type = std::make_signed<std::size_t>::type;
  using pointer = T *;
  using reference = T &;
  using iterator_category = std::random_access_iterator_tag;
  using is_hetero = std::true_type; // required
  using is_passed_directly = std::false_type;
  static constexpr sycl::access_mode mode = Mode; // required

  device_pointer(sycl::buffer<T, 1> in, std::size_t i = 0) : base_type(in, i) {}
#ifdef __USE_DPCT
  template <typename OtherT> device_pointer(OtherT *ptr) : base_type(ptr) {}
#endif
  // needed for malloc_device, count is number of bytes to allocate
  device_pointer(const std::size_t count) : base_type(count) {}
  device_pointer() : base_type() {}
  device_pointer(const device_pointer &in) : base_type(in) {}
  device_pointer &operator+=(difference_type forward) {
    this->idx += forward;
    return *this;
  }
  device_pointer &operator-=(difference_type backward) {
    this->idx -= backward;
    return *this;
  }
  // include operators from base class
  using base_type::operator++;
  using base_type::operator--;
  device_pointer &operator++() {
    this->idx += 1;
    return *this;
  }
  device_pointer &operator--() {
    this->idx -= 1;
    return *this;
  }
};

template <sycl::access_mode Mode, typename Allocator>
class device_pointer<void, Mode, Allocator>
    : public device_pointer_base<dpct::byte_t, Allocator,
                                 device_pointer<void, Mode, Allocator>> {
private:
  using base_type =
      device_pointer_base<dpct::byte_t, Allocator, device_pointer>;

public:
  using value_type = dpct::byte_t;
  using difference_type = std::make_signed<std::size_t>::type;
  using pointer = void *;
  using reference = value_type &;
  using iterator_category = std::random_access_iterator_tag;
  using is_hetero = std::true_type; // required
  using is_passed_directly = std::false_type;
  static constexpr sycl::access_mode mode = Mode; // required

  device_pointer(sycl::buffer<value_type, 1> in, std::size_t i = 0)
      : base_type(in, i) {}
#ifdef __USE_DPCT
  template <typename OtherT> device_pointer(OtherT *ptr) : base_type(ptr) {}
#endif
  // needed for malloc_device, count is number of bytes to allocate
  device_pointer(const std::size_t count) : base_type(count) {}
  device_pointer() : base_type() {}
  device_pointer(const device_pointer &in) : base_type(in) {}
  device_pointer &operator+=(difference_type forward) {
    this->idx += forward;
    return *this;
  }
  device_pointer &operator-=(difference_type backward) {
    this->idx -= backward;
    return *this;
  }
  // include operators from base class
  using base_type::operator++;
  using base_type::operator--;
  device_pointer &operator++() {
    this->idx += 1;
    return *this;
  }
  device_pointer &operator--() {
    this->idx -= 1;
    return *this;
  }
};
#else
template <typename T> class device_iterator;

template <typename ValueType, typename Derived> class device_pointer_base {
protected:
  ValueType *ptr;

public:
  using pointer = ValueType *;
  using difference_type = std::make_signed<std::size_t>::type;

  device_pointer_base(ValueType *p) : ptr(p) {}
  device_pointer_base(const std::size_t count) {
    sycl::queue default_queue = dpct::get_default_queue();
    ptr = static_cast<ValueType *>(sycl::malloc_shared(
        count, default_queue.get_device(), default_queue.get_context()));
  }
  device_pointer_base() {}
  pointer get() const { return ptr; }
  operator ValueType *() { return ptr; }
  operator ValueType *() const { return ptr; }

  ValueType &operator[](difference_type idx) { return ptr[idx]; }
  ValueType &operator[](difference_type idx) const { return ptr[idx]; }

  Derived operator+(difference_type forward) const {
    return Derived{ptr + forward};
  }
  Derived operator-(difference_type backward) const {
    return Derived{ptr - backward};
  }
  Derived operator++(int) {
    Derived p(ptr);
    ++ptr;
    return p;
  }
  Derived operator--(int) {
    Derived p(ptr);
    --ptr;
    return p;
  }
  difference_type operator-(const Derived &it) const { return ptr - it.ptr; }
};

template <typename T>
class device_pointer : public device_pointer_base<T, device_pointer<T>> {
private:
  using base_type = device_pointer_base<T, device_pointer<T>>;

public:
  using value_type = T;
  using difference_type = std::make_signed<std::size_t>::type;
  using pointer = T *;
  using reference = T &;
  using const_reference = const T &;
  using iterator_category = std::random_access_iterator_tag;
  using is_hetero = std::false_type;         // required
  using is_passed_directly = std::true_type; // required

  device_pointer(T *p) : base_type(p) {}
  // needed for malloc_device, count is number of bytes to allocate
  device_pointer(const std::size_t count) : base_type(count) {}
  device_pointer() : base_type() {}
  device_pointer &operator=(const device_iterator<T> &in) {
    this->ptr = static_cast<device_pointer<T>>(in).ptr;
    return *this;
  }

  // include operators from base class
  using base_type::operator++;
  using base_type::operator--;
  device_pointer &operator++() {
    ++(this->ptr);
    return *this;
  }
  device_pointer &operator--() {
    --(this->ptr);
    return *this;
  }
  device_pointer &operator+=(difference_type forward) {
    this->ptr = this->ptr + forward;
    return *this;
  }
  device_pointer &operator-=(difference_type backward) {
    this->ptr = this->ptr - backward;
    return *this;
  }
};

template <>
class device_pointer<void>
    : public device_pointer_base<dpct::byte_t, device_pointer<void>> {
private:
  using base_type = device_pointer_base<dpct::byte_t, device_pointer<void>>;

public:
  using value_type = dpct::byte_t;
  using difference_type = std::make_signed<std::size_t>::type;
  using pointer = void *;
  using reference = value_type &;
  using const_reference = const value_type &;
  using iterator_category = std::random_access_iterator_tag;
  using is_hetero = std::false_type;         // required
  using is_passed_directly = std::true_type; // required

  device_pointer(void *p) : base_type(static_cast<value_type *>(p)) {}
  // needed for malloc_device, count is number of bytes to allocate
  device_pointer(const std::size_t count) : base_type(count) {}
  device_pointer() : base_type() {}
  pointer get() const { return static_cast<pointer>(this->ptr); }
  operator void *() { return this->ptr; }
  operator void *() const { return this->ptr; }

  // include operators from base class
  using base_type::operator++;
  using base_type::operator--;
  device_pointer &operator++() {
    ++(this->ptr);
    return *this;
  }
  device_pointer &operator--() {
    --(this->ptr);
    return *this;
  }
  device_pointer &operator+=(difference_type forward) {
    this->ptr = this->ptr + forward;
    return *this;
  }
  device_pointer &operator-=(difference_type backward) {
    this->ptr = this->ptr - backward;
    return *this;
  }
};
#endif

#ifdef DPCT_USM_LEVEL_NONE
template <typename T, sycl::access_mode Mode = sycl::access_mode::read_write,
          typename Allocator = detail::__buffer_allocator<T>>
class device_iterator : public device_pointer<T, Mode, Allocator> {
  using Base = device_pointer<T, Mode, Allocator>;

public:
  using value_type = T;
  using difference_type = std::make_signed<std::size_t>::type;
  using pointer = T *;
  using reference = T &;
  using iterator_category = std::random_access_iterator_tag;
  using is_hetero = std::true_type;                // required
  using is_passed_directly = std::false_type;      // required
  static constexpr sycl::access_mode mode = Mode; // required

  device_iterator() : Base() {}
  device_iterator(sycl::buffer<T, 1, Allocator> vec, std::size_t index)
      : Base(vec, index) {}
  device_iterator(const Base &dev_ptr) : Base(dev_ptr) {}
  template <sycl::access_mode inMode>
  device_iterator(const device_iterator<T, inMode, Allocator> &in)
      : Base(in.buffer, in.idx) {} // required for iter_mode
  device_iterator &operator=(const device_iterator &in) {
    Base::buffer = in.buffer;
    Base::idx = in.idx;
    return *this;
  }

  reference operator*() const {
    return const_cast<device_iterator *>(this)
        ->buffer.template get_access<mode>()[Base::idx];
  }

  reference operator[](difference_type i) const { return *(*this + i); }
  device_iterator &operator++() {
    ++Base::idx;
    return *this;
  }
  device_iterator &operator--() {
    --Base::idx;
    return *this;
  }
  device_iterator operator++(int) {
    device_iterator it(*this);
    ++(*this);
    return it;
  }
  device_iterator operator--(int) {
    device_iterator it(*this);
    --(*this);
    return it;
  }
  device_iterator operator+(difference_type forward) const {
    const auto new_idx = Base::idx + forward;
    return {Base::buffer, new_idx};
  }
  device_iterator &operator+=(difference_type forward) {
    Base::idx += forward;
    return *this;
  }
  device_iterator operator-(difference_type backward) const {
    return {Base::buffer, Base::idx - backward};
  }
  device_iterator &operator-=(difference_type backward) {
    Base::idx -= backward;
    return *this;
  }
  friend device_iterator operator+(difference_type forward,
                                   const device_iterator &it) {
    return it + forward;
  }
  difference_type operator-(const device_iterator &it) const {
    return Base::idx - it.idx;
  }
  template <typename OtherIterator>
  typename std::enable_if<internal::is_hetero_iterator<OtherIterator>::value,
                          difference_type>::type
  operator-(const OtherIterator &it) const {
    return Base::idx - std::distance(oneapi::dpl::begin(Base::buffer), it);
  }
  bool operator==(const device_iterator &it) const { return *this - it == 0; }
  bool operator!=(const device_iterator &it) const { return !(*this == it); }
  bool operator<(const device_iterator &it) const { return *this - it < 0; }
  bool operator>(const device_iterator &it) const { return it < *this; }
  bool operator<=(const device_iterator &it) const { return !(*this > it); }
  bool operator>=(const device_iterator &it) const { return !(*this < it); }

  std::size_t get_idx() const { return Base::idx; } // required

  sycl::buffer<T, 1, Allocator> get_buffer() {
    return Base::buffer;
  } // required
};
#else
template <typename T> class device_iterator : public device_pointer<T> {
  using Base = device_pointer<T>;

protected:
  std::size_t idx;

public:
  using value_type = T;
  using difference_type = std::make_signed<std::size_t>::type;
  using pointer = typename Base::pointer;
  using reference = typename Base::reference;
  using iterator_category = std::random_access_iterator_tag;
  using is_hetero = std::false_type;         // required
  using is_passed_directly = std::true_type; // required
  static constexpr sycl::access_mode mode =
      sycl::access_mode::read_write; // required

  device_iterator() : Base(nullptr), idx(0) {}
  device_iterator(T *vec, std::size_t index) : Base(vec), idx(index) {}
  device_iterator(const Base &dev_ptr) : Base(dev_ptr), idx(0) {}
  template <sycl::access_mode inMode>
  device_iterator(const device_iterator<T> &in)
      : Base(in.ptr), idx(in.idx) {} // required for iter_mode
  device_iterator &operator=(const device_iterator &in) {
    Base::operator=(in);
    idx = in.idx;
    return *this;
  }

  reference operator*() const { return *(Base::ptr + idx); }

  reference operator[](difference_type i) { return Base::ptr[idx + i]; }
  reference operator[](difference_type i) const { return Base::ptr[idx + i]; }
  device_iterator &operator++() {
    ++idx;
    return *this;
  }
  device_iterator &operator--() {
    --idx;
    return *this;
  }
  device_iterator operator++(int) {
    device_iterator it(*this);
    ++(*this);
    return it;
  }
  device_iterator operator--(int) {
    device_iterator it(*this);
    --(*this);
    return it;
  }
  device_iterator operator+(difference_type forward) const {
    const auto new_idx = idx + forward;
    return {Base::ptr, new_idx};
  }
  device_iterator &operator+=(difference_type forward) {
    idx += forward;
    return *this;
  }
  device_iterator operator-(difference_type backward) const {
    return {Base::ptr, idx - backward};
  }
  device_iterator &operator-=(difference_type backward) {
    idx -= backward;
    return *this;
  }
  friend device_iterator operator+(difference_type forward,
                                   const device_iterator &it) {
    return it + forward;
  }
  difference_type operator-(const device_iterator &it) const {
    return idx - it.idx;
  }

  template <typename OtherIterator>
  typename std::enable_if<internal::is_hetero_iterator<OtherIterator>::value,
                          difference_type>::type
  operator-(const OtherIterator &it) const {
    return idx - it.get_idx();
  }

  bool operator==(const device_iterator &it) const { return *this - it == 0; }
  bool operator!=(const device_iterator &it) const { return !(*this == it); }
  bool operator<(const device_iterator &it) const { return *this - it < 0; }
  bool operator>(const device_iterator &it) const { return it < *this; }
  bool operator<=(const device_iterator &it) const { return !(*this > it); }
  bool operator>=(const device_iterator &it) const { return !(*this < it); }

  std::size_t get_idx() const { return idx; } // required

  device_iterator &get_buffer() { return *this; } // required

  std::size_t size() const { return idx; }
};
#endif

template <typename T>
device_pointer<T> malloc_device(const std::size_t num_elements) {
  return device_pointer<T>(num_elements * sizeof(T));
}
static inline device_pointer<void> malloc_device(const std::size_t num_bytes) {
  return device_pointer<void>(num_bytes);
}
template <typename T>
device_pointer<T> device_new(device_pointer<T> p, const T &value,
                             const std::size_t count = 1) {
  std::vector<T> result(count, value);
  p.buffer = sycl::buffer<T, 1>(result.begin(), result.end());
  return p + count;
}
template <typename T>
device_pointer<T> device_new(device_pointer<T> p, const std::size_t count = 1) {
  return device_new(p, T{}, count);
}
template <typename T>
device_pointer<T> device_new(const std::size_t count = 1) {
  return device_pointer<T>(count);
}

template <typename T> void free_device(device_pointer<T> ptr) {}

template <typename T>
typename std::enable_if<!std::is_trivially_destructible<T>::value, void>::type
device_delete(device_pointer<T> p, const std::size_t count = 1) {
  for (std::size_t i = 0; i < count; ++i) {
    p[i].~T();
  }
}
template <typename T>
typename std::enable_if<std::is_trivially_destructible<T>::value, void>::type
device_delete(device_pointer<T>, const std::size_t count = 1) {}

template <typename T> device_pointer<T> get_device_pointer(T *ptr) {
  return device_pointer<T>(ptr);
}

template <typename T>
device_pointer<T> get_device_pointer(const device_pointer<T> &ptr) {
  return device_pointer<T>(ptr);
}

template <typename T> T *get_raw_pointer(const device_pointer<T> &ptr) {
  return ptr.get();
}

template <typename Pointer> Pointer get_raw_pointer(const Pointer &ptr) {
  return ptr;
}

template <typename T> const T &get_raw_reference(const device_reference<T> &ref) {
  return ref.value;
}

template <typename T> T &get_raw_reference(device_reference<T> &ref) {
  return ref.value;
}

template <typename T> const T &get_raw_reference(const T &ref) {
  return ref;
}

template <typename T> T &get_raw_reference(T &ref) {
  return ref;
}

} // namespace dpct

#endif
