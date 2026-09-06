project(
  'abseil-cpp',
  'cpp',
  version: '20240722.0',
  license: 'Apache-2.0',
  default_options: [
    'cpp_std=c++17',
  ],
)

cpp = meson.get_compiler('cpp')

flags = cpp.get_supported_arguments('/DNOMINMAX', '-Wno-sign-compare', '-Wno-gcc-compat')
add_project_arguments(flags, language: 'cpp')

arch_cpp_flags = []
hw_cpp_flags = []
if host_machine.cpu_family() == 'x86_64'
  hw_cpp_flags += ['-maes', '-msse4.1']
elif host_machine.cpu_family() == 'aarch64' and cpp.sizeof('void*') == 8
  hw_cpp_flags += ['-march=armv8-a+crypto']
elif host_machine.cpu_family() == 'arm' and cpp.sizeof('void*') == 4
  hw_cpp_flags += ['-mfpu=neon']
elif host_machine.cpu_family() == 'ppc' or host_machine.cpu_family() == 'ppc64'
  # This will work with glibc but not musl
  timebase_check = '''#include <sys/platform/ppc.h>
    int main() {
       __ppc_get_timebase_freq();
       return 0;
    }'''
  if not cpp.compiles(timebase_check)
    arch_cpp_flags += ['-DABSL_USE_UNSCALED_CYCLECLOCK=0']
  endif
endif
arch_flags = cpp.get_supported_arguments(arch_cpp_flags)
hw_flags = cpp.get_supported_arguments(hw_cpp_flags)

libatomic = dependency('', required: false)
if cpp.get_argument_syntax() != 'msvc' and not cpp.links('int main(){__sync_synchronize();}', name: 'atomic builtins')
  libatomic = cpp.find_library('atomic')
endif

absl_include_dir = include_directories('.')

# Group files by the containing library
absl_base_sources = files(
  'absl/base/internal/cycleclock.cc',
  'absl/base/internal/low_level_alloc.cc',
  'absl/base/internal/poison.cc',
  'absl/base/internal/raw_logging.cc',
  'absl/base/internal/scoped_set_env.cc',
  'absl/base/internal/spinlock.cc',
  'absl/base/internal/spinlock_wait.cc',
  'absl/base/internal/strerror.cc',
  'absl/base/internal/sysinfo.cc',
  'absl/base/internal/thread_identity.cc',
  'absl/base/internal/throw_delegate.cc',
  'absl/base/internal/unscaledcycleclock.cc',
  'absl/base/log_severity.cc',
)
absl_base_headers = files(
  'absl/base/attributes.h',
  'absl/base/call_once.h',
  'absl/base/casts.h',
  'absl/base/config.h',
  'absl/base/const_init.h',
  'absl/base/dynamic_annotations.h',
  'absl/base/internal/atomic_hook.h',
  'absl/base/internal/atomic_hook_test_helper.h',
  'absl/base/internal/cycleclock.h',
  'absl/base/internal/cycleclock_config.h',
  'absl/base/internal/direct_mmap.h',
  'absl/base/internal/dynamic_annotations.h',
  'absl/base/internal/endian.h',
  'absl/base/internal/errno_saver.h',
  'absl/base/internal/exception_safety_testing.h',
  'absl/base/internal/exception_testing.h',
  'absl/base/internal/fast_type_id.h',
  'absl/base/internal/hide_ptr.h',
  'absl/base/internal/identity.h',
  'absl/base/internal/inline_variable.h',
  'absl/base/internal/inline_variable_testing.h',
  'absl/base/internal/invoke.h',
  'absl/base/internal/low_level_alloc.h',
  'absl/base/internal/low_level_scheduling.h',
  'absl/base/internal/nullability_impl.h',
  'absl/base/internal/per_thread_tls.h',
  'absl/base/internal/poison.h',
  'absl/base/internal/pretty_function.h',
  'absl/base/internal/raw_logging.h',
  'absl/base/internal/scheduling_mode.h',
  'absl/base/internal/scoped_set_env.h',
  'absl/base/internal/spinlock.h',
  'absl/base/internal/spinlock_akaros.inc',
  'absl/base/internal/spinlock_linux.inc',
  'absl/base/internal/spinlock_posix.inc',
  'absl/base/internal/spinlock_wait.h',
  'absl/base/internal/spinlock_win32.inc',
  'absl/base/internal/strerror.h',
  'absl/base/internal/sysinfo.h',
  'absl/base/internal/thread_identity.h',
  'absl/base/internal/throw_delegate.h',
  'absl/base/internal/tsan_mutex_interface.h',
  'absl/base/internal/unaligned_access.h',
  'absl/base/internal/unscaledcycleclock.h',
  'absl/base/internal/unscaledcycleclock_config.h',
  'absl/base/log_severity.h',
  'absl/base/macros.h',
  'absl/base/no_destructor.h',
  'absl/base/nullability.h',
  'absl/base/optimization.h',
  'absl/base/options.h',
  'absl/base/policy_checks.h',
  'absl/base/port.h',
  'absl/base/thread_annotations.h',
  'absl/functional/any_invocable.h',
  'absl/functional/internal/any_invocable.h',
  'absl/memory/memory.h',
  'absl/meta/type_traits.h',
  'absl/utility/utility.h',
  # Dependent headers of absl_base
)

absl_container_sources = files(
  'absl/container/internal/hashtablez_sampler.cc',
  'absl/container/internal/hashtablez_sampler_force_weak_definition.cc',
  'absl/container/internal/raw_hash_set.cc',
)
absl_container_headers = files(
  'absl/container/btree_map.h',
  'absl/container/btree_set.h',
  'absl/container/hash_container_defaults.h',
  'absl/container/btree_test.h',
  'absl/container/fixed_array.h',
  'absl/container/flat_hash_map.h',
  'absl/container/flat_hash_set.h',
  'absl/container/inlined_vector.h',
  'absl/container/internal/btree.h',
  'absl/container/internal/btree_container.h',
  'absl/container/internal/common.h',
  'absl/container/internal/common_policy_traits.h',
  'absl/container/internal/compressed_tuple.h',
  'absl/container/internal/container_memory.h',
  'absl/container/internal/hash_function_defaults.h',
  'absl/container/internal/hash_generator_testing.h',
  'absl/container/internal/hash_policy_testing.h',
  'absl/container/internal/hash_policy_traits.h',
  'absl/container/internal/hashtable_debug.h',
  'absl/container/internal/hashtable_debug_hooks.h',
  'absl/container/internal/hashtablez_sampler.h',
  'absl/container/internal/inlined_vector.h',
  'absl/container/internal/layout.h',
  'absl/container/internal/node_slot_policy.h',
  'absl/container/internal/raw_hash_map.h',
  'absl/container/internal/raw_hash_set.h',
  'absl/container/internal/test_instance_tracker.h',
  'absl/container/internal/tracked.h',
  'absl/container/internal/unordered_map_constructor_test.h',
  'absl/container/internal/unordered_map_lookup_test.h',
  'absl/container/internal/unordered_map_members_test.h',
  'absl/container/internal/unordered_map_modifiers_test.h',
  'absl/container/internal/unordered_set_constructor_test.h',
  'absl/container/internal/unordered_set_lookup_test.h',
  'absl/container/internal/unordered_set_members_test.h',
  'absl/container/internal/unordered_set_modifiers_test.h',
  'absl/container/node_hash_map.h',
  'absl/container/node_hash_set.h',
)

absl_crc_sources = files(
  'absl/crc/crc32c.cc',
  'absl/crc/internal/cpu_detect.cc',
  'absl/crc/internal/crc.cc',
  'absl/crc/internal/crc_cord_state.cc',
  'absl/crc/internal/crc_memcpy_fallback.cc',
  'absl/crc/internal/crc_memcpy_x86_arm_combined.cc',
  'absl/crc/internal/crc_non_temporal_memcpy.cc',
  'absl/crc/internal/crc_x86_arm_combined.cc',
)
absl_crc_headers = files(
  'absl/crc/crc32c.h',
  'absl/crc/internal/cpu_detect.h',
  'absl/crc/internal/crc.h',
  'absl/crc/internal/crc32_x86_arm_combined_simd.h',
  'absl/crc/internal/crc32c.h',
  'absl/crc/internal/crc32c_inline.h',
  'absl/crc/internal/crc_cord_state.h',
  'absl/crc/internal/crc_internal.h',
  'absl/crc/internal/crc_memcpy.h',
  'absl/crc/internal/non_temporal_arm_intrinsics.h',
  'absl/crc/internal/non_temporal_memcpy.h',
)

absl_debugging_sources = files(
  'absl/debugging/failure_signal_handler.cc',
  'absl/debugging/internal/address_is_readable.cc',
  'absl/debugging/internal/decode_rust_punycode.cc',
  'absl/debugging/internal/demangle.cc',
  'absl/debugging/internal/demangle_rust.cc',
  'absl/debugging/internal/elf_mem_image.cc',
  'absl/debugging/internal/examine_stack.cc',
  'absl/debugging/internal/stack_consumption.cc',
  'absl/debugging/internal/vdso_support.cc',
  'absl/debugging/leak_check.cc',
  'absl/debugging/stacktrace.cc',
  'absl/debugging/symbolize.cc',
  'absl/debugging/internal/utf8_for_code_point.cc',
)
absl_debugging_headers = files(
  'absl/debugging/failure_signal_handler.h',
  'absl/debugging/internal/address_is_readable.h',
  'absl/debugging/internal/bounded_utf8_length_sequence.h',
  'absl/debugging/internal/decode_rust_punycode.h',
  'absl/debugging/internal/demangle.h',
  'absl/debugging/internal/demangle_rust.h',
  'absl/debugging/internal/elf_mem_image.h',
  'absl/debugging/internal/examine_stack.h',
  'absl/debugging/internal/stack_consumption.h',
  'absl/debugging/internal/stacktrace_aarch64-inl.inc',
  'absl/debugging/internal/stacktrace_arm-inl.inc',
  'absl/debugging/internal/stacktrace_config.h',
  'absl/debugging/internal/stacktrace_emscripten-inl.inc',
  'absl/debugging/internal/stacktrace_generic-inl.inc',
  'absl/debugging/internal/stacktrace_powerpc-inl.inc',
  'absl/debugging/internal/stacktrace_riscv-inl.inc',
  'absl/debugging/internal/stacktrace_unimplemented-inl.inc',
  'absl/debugging/internal/stacktrace_win32-inl.inc',
  'absl/debugging/internal/stacktrace_x86-inl.inc',
  'absl/debugging/internal/symbolize.h',
  'absl/debugging/internal/utf8_for_code_point.h',
  'absl/debugging/internal/vdso_support.h',
  'absl/debugging/leak_check.h',
  'absl/debugging/stacktrace.h',
  'absl/debugging/symbolize.h',
  'absl/debugging/symbolize_darwin.inc',
  'absl/debugging/symbolize_elf.inc',
  'absl/debugging/symbolize_emscripten.inc',
  'absl/debugging/symbolize_unimplemented.inc',
  'absl/debugging/symbolize_win32.inc',
)

absl_flags_sources = files(
  'absl/flags/commandlineflag.cc',
  'absl/flags/internal/flag.cc',
  'absl/flags/internal/commandlineflag.cc',
  'absl/flags/internal/flag.cc',
  'absl/flags/internal/private_handle_accessor.cc',
  'absl/flags/internal/program_name.cc',
  'absl/flags/internal/usage.cc',
  'absl/flags/marshalling.cc',
  'absl/flags/parse.cc',
  'absl/flags/reflection.cc',
  'absl/flags/usage.cc',
  'absl/flags/usage_config.cc',
)
absl_flags_headers = files(
  'absl/flags/commandlineflag.h',
  'absl/flags/config.h',
  'absl/flags/declare.h',
  'absl/flags/internal/flag.h',
  'absl/flags/internal/commandlineflag.h',
  'absl/flags/internal/flag.h',
  'absl/flags/internal/parse.h',
  'absl/flags/internal/path_util.h',
  'absl/flags/internal/private_handle_accessor.h',
  'absl/flags/internal/program_name.h',
  'absl/flags/internal/registry.h',
  'absl/flags/internal/sequence_lock.h',
  'absl/flags/internal/usage.h',
  'absl/flags/marshalling.h',
  'absl/flags/parse.h',
  'absl/flags/reflection.h',
  'absl/flags/usage.h',
  'absl/flags/usage_config.h',
)

absl_hash_sources = files(
  'absl/hash/internal/city.cc',
  'absl/hash/internal/hash.cc',
  'absl/hash/internal/low_level_hash.cc',
)
absl_hash_headers = files(
  'absl/hash/hash.h',
  'absl/hash/hash_testing.h',
  'absl/hash/internal/city.h',
  'absl/hash/internal/hash.h',
  'absl/hash/internal/low_level_hash.h',
  'absl/hash/internal/spy_hash_state.h',
)

absl_log_sources = files(
  'absl/log/die_if_null.cc',
  'absl/log/flags.cc',
  'absl/log/globals.cc',
  'absl/log/initialize.cc',
  'absl/log/internal/check_op.cc',
  'absl/log/internal/conditions.cc',
  'absl/log/internal/fnmatch.cc',
  'absl/log/internal/globals.cc',
  'absl/log/internal/log_format.cc',
  'absl/log/internal/log_message.cc',
  'absl/log/internal/log_sink_set.cc',
  'absl/log/internal/nullguard.cc',
  'absl/log/internal/proto.cc',
  'absl/log/internal/vlog_config.cc',
  'absl/log/log_entry.cc',
  'absl/log/log_sink.cc',
)
absl_log_headers = files(
  'absl/log/absl_check.h',
  'absl/log/absl_log.h',
  'absl/log/check.h',
  'absl/log/die_if_null.h',
  'absl/log/flags.h',
  'absl/log/globals.h',
  'absl/log/initialize.h',
  'absl/log/internal/append_truncated.h',
  'absl/log/internal/check_impl.h',
  'absl/log/internal/check_op.h',
  'absl/log/internal/conditions.h',
  'absl/log/internal/config.h',
  'absl/log/internal/fnmatch.h',
  'absl/log/internal/flags.h',
  'absl/log/internal/globals.h',
  'absl/log/internal/log_format.h',
  'absl/log/internal/log_impl.h',
  'absl/log/internal/log_message.h',
  'absl/log/internal/log_sink_set.h',
  'absl/log/internal/nullguard.h',
  'absl/log/internal/nullstream.h',
  'absl/log/internal/proto.h',
  'absl/log/internal/strip.h',
  'absl/log/internal/structured.h',
  'absl/log/internal/test_actions.h',
  'absl/log/internal/test_helpers.h',
  'absl/log/internal/test_matchers.h',
  'absl/log/internal/vlog_config.h',
  'absl/log/internal/voidify.h',
  'absl/log/log.h',
  'absl/log/log_entry.h',
  'absl/log/log_sink.h',
  'absl/log/log_sink_registry.h',
  'absl/log/log_streamer.h',
  'absl/log/scoped_mock_log.h',
  'absl/log/structured.h',
)

absl_numeric_sources = files(
  'absl/numeric/int128.cc',
)
absl_numeric_headers = files(
  'absl/numeric/bits.h',
  'absl/numeric/int128.h',
  'absl/numeric/int128_have_intrinsic.inc',
  'absl/numeric/int128_no_intrinsic.inc',
  'absl/numeric/internal/bits.h',
  'absl/numeric/internal/representation.h',
)

absl_profiling_sources = files(
  'absl/profiling/internal/exponential_biased.cc',
  'absl/profiling/internal/periodic_sampler.cc',
)
absl_profiling_headers = files(
  'absl/profiling/internal/exponential_biased.h',
  'absl/profiling/internal/periodic_sampler.h',
  'absl/profiling/internal/sample_recorder.h',
)

absl_random_sources = files(
  'absl/random/discrete_distribution.cc',
  'absl/random/gaussian_distribution.cc',
  'absl/random/internal/chi_square.cc',
  'absl/random/internal/pool_urbg.cc',
  'absl/random/internal/randen.cc',
  'absl/random/internal/randen_detect.cc',
  'absl/random/internal/randen_hwaes.cc',
  'absl/random/internal/randen_round_keys.cc',
  'absl/random/internal/randen_slow.cc',
  'absl/random/internal/seed_material.cc',
  'absl/random/seed_gen_exception.cc',
  'absl/random/seed_sequences.cc',
)
absl_random_headers = files(
  'absl/random/bernoulli_distribution.h',
  'absl/random/beta_distribution.h',
  'absl/random/bit_gen_ref.h',
  'absl/random/discrete_distribution.h',
  'absl/random/distributions.h',
  'absl/random/exponential_distribution.h',
  'absl/random/gaussian_distribution.h',
  'absl/random/internal/chi_square.h',
  'absl/random/internal/distribution_caller.h',
  'absl/random/internal/distribution_test_util.h',
  'absl/random/internal/explicit_seed_seq.h',
  'absl/random/internal/fast_uniform_bits.h',
  'absl/random/internal/fastmath.h',
  'absl/random/internal/generate_real.h',
  'absl/random/internal/iostream_state_saver.h',
  'absl/random/internal/mock_helpers.h',
  'absl/random/internal/mock_overload_set.h',
  'absl/random/internal/nanobenchmark.h',
  'absl/random/internal/nonsecure_base.h',
  'absl/random/internal/pcg_engine.h',
  'absl/random/internal/platform.h',
  'absl/random/internal/pool_urbg.h',
  'absl/random/internal/randen.h',
  'absl/random/internal/randen_detect.h',
  'absl/random/internal/randen_engine.h',
  'absl/random/internal/randen_hwaes.h',
  'absl/random/internal/randen_slow.h',
  'absl/random/internal/randen_traits.h',
  'absl/random/internal/salted_seed_seq.h',
  'absl/random/internal/seed_material.h',
  'absl/random/internal/sequence_urbg.h',
  'absl/random/internal/traits.h',
  'absl/random/internal/uniform_helper.h',
  'absl/random/internal/wide_multiply.h',
  'absl/random/log_uniform_int_distribution.h',
  'absl/random/mock_distributions.h',
  'absl/random/mocking_bit_gen.h',
  'absl/random/poisson_distribution.h',
  'absl/random/random.h',
  'absl/random/seed_gen_exception.h',
  'absl/random/seed_sequences.h',
  'absl/random/uniform_int_distribution.h',
  'absl/random/uniform_real_distribution.h',
  'absl/random/zipf_distribution.h',
)

absl_status_sources = files(
  'absl/status/internal/status_internal.cc',
  'absl/status/status.cc',
  'absl/status/status_payload_printer.cc',
  'absl/status/statusor.cc',
)
absl_status_headers = files(
  'absl/status/internal/status_internal.h',
  'absl/status/internal/statusor_internal.h',
  'absl/status/status.h',
  'absl/status/status_payload_printer.h',
  'absl/status/statusor.h',
)

absl_strings_sources = files(
  'absl/strings/ascii.cc',
  'absl/strings/charconv.cc',
  'absl/strings/cord.cc',
  'absl/strings/cord_analysis.cc',
  'absl/strings/cord_buffer.cc',
  'absl/strings/escaping.cc',
  'absl/strings/internal/charconv_bigint.cc',
  'absl/strings/internal/charconv_parse.cc',
  'absl/strings/internal/cord_internal.cc',
  'absl/strings/internal/cord_rep_btree.cc',
  'absl/strings/internal/cord_rep_btree_navigator.cc',
  'absl/strings/internal/cord_rep_btree_reader.cc',
  'absl/strings/internal/cord_rep_consume.cc',
  'absl/strings/internal/cord_rep_crc.cc',
  'absl/strings/internal/cordz_functions.cc',
  'absl/strings/internal/cordz_handle.cc',
  'absl/strings/internal/cordz_info.cc',
  'absl/strings/internal/cordz_sample_token.cc',
  'absl/strings/internal/damerau_levenshtein_distance.cc',
  'absl/strings/internal/escaping.cc',
  'absl/strings/internal/memutil.cc',
  'absl/strings/internal/ostringstream.cc',
  'absl/strings/internal/pow10_helper.cc',
  'absl/strings/internal/str_format/arg.cc',
  'absl/strings/internal/str_format/bind.cc',
  'absl/strings/internal/str_format/extension.cc',
  'absl/strings/internal/str_format/float_conversion.cc',
  'absl/strings/internal/str_format/output.cc',
  'absl/strings/internal/str_format/parser.cc',
  'absl/strings/internal/stringify_sink.cc',
  'absl/strings/internal/utf8.cc',
  'absl/strings/match.cc',
  'absl/strings/numbers.cc',
  'absl/strings/str_cat.cc',
  'absl/strings/str_replace.cc',
  'absl/strings/str_split.cc',
  'absl/strings/string_view.cc',
  'absl/strings/substitute.cc',
)
absl_strings_headers = files(
  'absl/strings/ascii.h',
  'absl/strings/charconv.h',
  'absl/strings/cord.h',
  'absl/strings/cord_analysis.h',
  'absl/strings/cord_buffer.h',
  'absl/strings/cord_test_helpers.h',
  'absl/strings/cordz_test_helpers.h',
  'absl/strings/escaping.h',
  'absl/strings/has_absl_stringify.h',
  'absl/strings/internal/charconv_bigint.h',
  'absl/strings/internal/charconv_parse.h',
  'absl/strings/internal/cord_data_edge.h',
  'absl/strings/internal/cord_internal.h',
  'absl/strings/internal/cord_rep_btree.h',
  'absl/strings/internal/cord_rep_btree_navigator.h',
  'absl/strings/internal/cord_rep_btree_reader.h',
  'absl/strings/internal/cord_rep_consume.h',
  'absl/strings/internal/cord_rep_crc.h',
  'absl/strings/internal/cord_rep_flat.h',
  'absl/strings/internal/cord_rep_test_util.h',
  'absl/strings/internal/cordz_functions.h',
  'absl/strings/internal/cordz_handle.h',
  'absl/strings/internal/cordz_info.h',
  'absl/strings/internal/cordz_sample_token.h',
  'absl/strings/internal/cordz_statistics.h',
  'absl/strings/internal/cordz_update_scope.h',
  'absl/strings/internal/cordz_update_tracker.h',
  'absl/strings/internal/damerau_levenshtein_distance.h',
  'absl/strings/internal/escaping.h',
  'absl/strings/internal/escaping_test_common.h',
  'absl/strings/internal/memutil.h',
  'absl/strings/internal/numbers_test_common.h',
  'absl/strings/internal/ostringstream.h',
  'absl/strings/internal/pow10_helper.h',
  'absl/strings/internal/resize_uninitialized.h',
  'absl/strings/internal/stl_type_traits.h',
  'absl/strings/internal/str_format/arg.h',
  'absl/strings/internal/str_format/bind.h',
  'absl/strings/internal/str_format/checker.h',
  'absl/strings/internal/str_format/constexpr_parser.h',
  'absl/strings/internal/str_format/extension.h',
  'absl/strings/internal/str_format/float_conversion.h',
  'absl/strings/internal/str_format/output.h',
  'absl/strings/internal/str_format/parser.h',
  'absl/strings/internal/str_join_internal.h',
  'absl/strings/internal/str_split_internal.h',
  'absl/strings/internal/string_constant.h',
  'absl/strings/internal/stringify_sink.h',
  'absl/strings/internal/utf8.h',
  'absl/strings/match.h',
  'absl/strings/numbers.h',
  'absl/strings/str_cat.h',
  'absl/strings/str_format.h',
  'absl/strings/str_join.h',
  'absl/strings/str_replace.h',
  'absl/strings/str_split.h',
  'absl/strings/string_view.h',
  'absl/strings/strip.h',
  'absl/strings/substitute.h',
)

absl_synchronization_sources = files(
  'absl/synchronization/barrier.cc',
  'absl/synchronization/blocking_counter.cc',
  'absl/synchronization/internal/create_thread_identity.cc',
  'absl/synchronization/internal/futex_waiter.cc',
  'absl/synchronization/internal/graphcycles.cc',
  'absl/synchronization/internal/kernel_timeout.cc',
  'absl/synchronization/internal/per_thread_sem.cc',
  'absl/synchronization/internal/pthread_waiter.cc',
  'absl/synchronization/internal/sem_waiter.cc',
  'absl/synchronization/internal/stdcpp_waiter.cc',
  'absl/synchronization/internal/waiter_base.cc',
  'absl/synchronization/internal/win32_waiter.cc',
  'absl/synchronization/mutex.cc',
  'absl/synchronization/notification.cc',
)
absl_synchronization_headers = files(
  'absl/synchronization/barrier.h',
  'absl/synchronization/blocking_counter.h',
  'absl/synchronization/internal/create_thread_identity.h',
  'absl/synchronization/internal/futex.h',
  'absl/synchronization/internal/graphcycles.h',
  'absl/synchronization/internal/kernel_timeout.h',
  'absl/synchronization/internal/per_thread_sem.h',
  'absl/synchronization/internal/thread_pool.h',
  'absl/synchronization/internal/waiter.h',
  'absl/synchronization/mutex.h',
  'absl/synchronization/notification.h',
)

absl_time_sources = files(
  'absl/time/civil_time.cc',
  'absl/time/clock.cc',
  'absl/time/duration.cc',
  'absl/time/format.cc',
  'absl/time/internal/cctz/src/civil_time_detail.cc',
  'absl/time/internal/cctz/src/time_zone_fixed.cc',
  'absl/time/internal/cctz/src/time_zone_format.cc',
  'absl/time/internal/cctz/src/time_zone_if.cc',
  'absl/time/internal/cctz/src/time_zone_impl.cc',
  'absl/time/internal/cctz/src/time_zone_info.cc',
  'absl/time/internal/cctz/src/time_zone_libc.cc',
  'absl/time/internal/cctz/src/time_zone_lookup.cc',
  'absl/time/internal/cctz/src/time_zone_posix.cc',
  'absl/time/internal/cctz/src/zone_info_source.cc',
  'absl/time/time.cc',
)
absl_time_headers = files(
  'absl/time/civil_time.h',
  'absl/time/clock.h',
  'absl/time/internal/cctz/include/cctz/civil_time.h',
  'absl/time/internal/cctz/include/cctz/civil_time_detail.h',
  'absl/time/internal/cctz/include/cctz/time_zone.h',
  'absl/time/internal/cctz/include/cctz/zone_info_source.h',
  'absl/time/internal/cctz/src/time_zone_fixed.h',
  'absl/time/internal/cctz/src/time_zone_if.h',
  'absl/time/internal/cctz/src/time_zone_impl.h',
  'absl/time/internal/cctz/src/time_zone_info.h',
  'absl/time/internal/cctz/src/time_zone_libc.h',
  'absl/time/internal/cctz/src/time_zone_posix.h',
  'absl/time/internal/cctz/src/tzfile.h',
  'absl/time/internal/get_current_time_chrono.inc',
  'absl/time/internal/get_current_time_posix.inc',
  'absl/time/internal/test_util.h',
  'absl/time/time.h',
)

absl_types_sources = files(
  'absl/types/bad_any_cast.cc',
  'absl/types/bad_optional_access.cc',
  'absl/types/bad_variant_access.cc',
)
absl_types_headers = files(
  'absl/types/any.h',
  'absl/types/bad_any_cast.h',
  'absl/types/bad_optional_access.h',
  'absl/types/bad_variant_access.h',
  'absl/types/compare.h',
  'absl/types/internal/optional.h',
  'absl/types/internal/span.h',
  'absl/types/internal/variant.h',
  'absl/types/optional.h',
  'absl/types/span.h',
  'absl/types/variant.h',
)

# Libraries
absl_base_lib = static_library(
  'absl_base',
  absl_base_sources,
  include_directories: absl_include_dir,
  cpp_args: arch_flags,
  dependencies: [dependency('threads'), libatomic],
)

absl_hash_lib = static_library(
  'absl_hash',
  absl_hash_sources,
  include_directories: absl_include_dir,
)

absl_numeric_lib = static_library(
  'absl_numeric',
  absl_numeric_sources,
  include_directories: absl_include_dir,
)

absl_profiling_lib = static_library(
  'absl_profiling',
  absl_profiling_sources,
  include_directories: absl_include_dir,
)

absl_crc_lib = static_library(
  'absl_crc',
  absl_crc_sources,
  include_directories: absl_include_dir,
  link_with: [
    absl_base_lib,
  ],
  dependencies: libatomic,
)

absl_strings_lib = static_library(
  'absl_strings',
  absl_strings_sources,
  include_directories: absl_include_dir,
  link_with: [
    absl_base_lib,
    absl_crc_lib,
    absl_numeric_lib,
    absl_profiling_lib,
  ],
)

absl_debugging_lib = static_library(
  'absl_debugging',
  absl_debugging_sources,
  include_directories: absl_include_dir,
  link_with: [
    absl_base_lib,
    absl_strings_lib,
  ],
  dependencies: libatomic,
)

absl_random_lib = static_library(
  'absl_random',
  absl_random_sources,
  include_directories: absl_include_dir,
  cpp_args: hw_flags,
  link_with: [
    absl_base_lib,
    absl_strings_lib,
  ],
  dependencies: libatomic,
)

absl_time_lib = static_library(
  'absl_time',
  absl_time_sources,
  include_directories: absl_include_dir,
  link_with: [
    absl_base_lib,
    absl_numeric_lib,
    absl_strings_lib,
  ],
  # macOS only, upstream: https://github.com/abseil/abseil-cpp/pull/280
  dependencies: dependency('appleframeworks', modules: 'CoreFoundation', required: host_machine.system() == 'darwin'),
)

absl_types_lib = static_library(
  'absl_types',
  absl_types_sources,
  include_directories: absl_include_dir,
)

absl_synchronization_lib = static_library(
  'absl_synchronization',
  absl_synchronization_sources,
  include_directories: absl_include_dir,
  link_with: [
    absl_base_lib,
    absl_debugging_lib,
    absl_time_lib,
  ],
)

absl_container_lib = static_library(
  'absl_container',
  absl_container_sources,
  include_directories: absl_include_dir,
  link_with: [
    absl_base_lib,
    absl_debugging_lib,
    absl_hash_lib,
    absl_synchronization_lib,
    absl_time_lib,
  ],
)

absl_flags_lib = static_library(
  'absl_flags',
  absl_flags_sources,
  include_directories: absl_include_dir,
  link_with: [
    absl_base_lib,
    absl_container_lib,
    absl_hash_lib,
    absl_strings_lib,
    absl_synchronization_lib,
  ],
  dependencies: libatomic,
)

absl_status_lib = static_library(
  'absl_status',
  absl_status_sources,
  include_directories: absl_include_dir,
  link_with: [
    absl_base_lib,
    absl_strings_lib,
  ],
)

absl_log_lib = static_library(
  'absl_log',
  absl_log_sources,
  include_directories: absl_include_dir,
  link_with: [
    absl_base_lib,
    absl_strings_lib,
    absl_flags_lib,
  ],
  dependencies: libatomic,
)

# Dependencies
absl_base_dep = declare_dependency(
  include_directories: absl_include_dir,
  link_with: absl_base_lib,
)

absl_hash_dep = declare_dependency(
  include_directories: absl_include_dir,
  link_with: absl_hash_lib,
)

absl_numeric_dep = declare_dependency(
  include_directories: absl_include_dir,
  link_with: absl_numeric_lib,
)

absl_profiling_dep = declare_dependency(
  include_directories: absl_include_dir,
  link_with: absl_profiling_lib,
)

absl_strings_dep = declare_dependency(
  include_directories: absl_include_dir,
  link_with: absl_strings_lib,
  dependencies: [
    absl_base_dep,
    absl_numeric_dep,
  ],
)

absl_debugging_dep = declare_dependency(
  include_directories: absl_include_dir,
  link_with: absl_debugging_lib,
  dependencies: [
    absl_base_dep,
    absl_strings_dep,
  ],
)

absl_random_dep = declare_dependency(
  include_directories: absl_include_dir,
  link_with: absl_random_lib,
  dependencies: [
    absl_base_dep,
    absl_strings_dep,
  ],
)

absl_crc_dep = declare_dependency(
  include_directories: absl_include_dir,
  link_with: absl_crc_lib,
  dependencies: [
    absl_base_dep,
  ],
)

absl_time_dep = declare_dependency(
  include_directories: absl_include_dir,
  link_with: absl_time_lib,
  dependencies: [
    absl_base_dep,
    absl_numeric_dep,
    absl_strings_dep,
  ],
)

absl_types_dep = declare_dependency(
  include_directories: absl_include_dir,
  link_with: absl_types_lib,
)

absl_synchronization_dep = declare_dependency(
  include_directories: absl_include_dir,
  link_with: absl_synchronization_lib,
  dependencies: [
    absl_base_dep,
    absl_debugging_dep,
    absl_time_dep,
  ],
)

absl_container_dep = declare_dependency(
  include_directories: absl_include_dir,
  link_with: absl_container_lib,
  dependencies: [
    absl_base_dep,
    absl_debugging_dep,
    absl_hash_dep,
    absl_synchronization_dep,
    absl_time_dep,
  ],
)

absl_flags_dep = declare_dependency(
  include_directories: absl_include_dir,
  link_with: absl_flags_lib,
  dependencies: [
    absl_base_dep,
    absl_container_dep,
    absl_hash_dep,
    absl_strings_dep,
    absl_synchronization_dep,
  ],
)

absl_log_dep = declare_dependency(
  include_directories: absl_include_dir,
  link_with: absl_log_lib,
  dependencies: [
    absl_base_dep,
    absl_strings_dep,
    absl_flags_dep,
  ],
)

absl_status_dep = declare_dependency(
  include_directories: absl_include_dir,
  link_with: absl_status_lib,
  dependencies: [
    absl_base_dep,
    absl_strings_dep,
  ],
)
