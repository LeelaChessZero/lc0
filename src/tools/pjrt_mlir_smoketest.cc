/*
  pjrt_mlir_smoketest.cc

  Check 0b: PJRT MLIR compile smoketest tool

  Usage:
    pjrt_mlir_smoketest <plugin_path> <input.mlir|input.mlirbc>

  This tool verifies that a PJRT plugin can compile StableHLO MLIR programs
  (both text and bytecode formats) using the same compile options path that
  Lc0 uses for HLO.

  Success criteria:
    - Plugin loads successfully
    - Plugin attributes are printed (especially StableHLO version range)
    - Client is created
    - Input file compiles with format="mlir"
    - Exit 0 on success, nonzero + error on failure
*/

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

#include "neural/backends/xla/pjrt.h"
#include "proto/hlo.pb.h"
#include "utils/exception.h"

namespace {

// Read entire file into string (binary mode for .mlirbc support)
std::string ReadFile(const char* path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error(std::string("Failed to open file: ") + path);
  }
  std::string content{std::istreambuf_iterator<char>(file),
                      std::istreambuf_iterator<char>()};
  if (file.bad()) {
    throw std::runtime_error(std::string("I/O error reading file: ") + path);
  }
  return content;
}

// Build CompileOptionsProto exactly as Lc0 does (from xla_runner.cc)
std::string BuildCompileOptions(int device_id) {
  pblczero::CompileOptionsProto options;

  auto* build_opts = options.mutable_executable_build_options();
  build_opts->set_num_replicas(1);
  build_opts->set_num_partitions(1);

  auto* device_assignment = build_opts->mutable_device_assignment();
  device_assignment->set_replica_count(1);
  device_assignment->set_computation_count(1);
  device_assignment->add_computation_devices()->add_replica_device_ids(device_id);

  return options.OutputAsString();
}

void PrintUsage(const char* prog_name) {
  std::cerr << "Usage: " << prog_name << " <plugin_path> <input.mlir|input.mlirbc>\n";
  std::cerr << "\n";
  std::cerr << "Check 0b: PJRT MLIR compile smoketest\n";
  std::cerr << "\n";
  std::cerr << "Arguments:\n";
  std::cerr << "  plugin_path    Path to PJRT plugin shared library (.so/.dll)\n";
  std::cerr << "  input          StableHLO MLIR text (.mlir) or bytecode (.mlirbc)\n";
  std::cerr << "\n";
  std::cerr << "Exit codes:\n";
  std::cerr << "  0  Success - plugin compiled the MLIR program\n";
  std::cerr << "  1  Failure - see error message\n";
  std::cerr << "  2  Usage error\n";
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc != 3) {
    PrintUsage(argv[0]);
    return 2;
  }

  const std::string plugin_path = argv[1];
  const std::string input_path = argv[2];

  std::cout << "=== PJRT MLIR Smoketest (Check 0b) ===\n";
  std::cout << "Plugin: " << plugin_path << "\n";
  std::cout << "Input:  " << input_path << "\n";
  std::cout << "\n";

  try {
    // 1. Load plugin
    std::cout << "[1/5] Loading PJRT plugin...\n";
    lczero::Pjrt pjrt(plugin_path.c_str());
    std::cout << "      OK\n";

    // 2. Print plugin attributes (StableHLO version range if present)
    std::cout << "[2/5] Plugin attributes:\n";
    bool found_stablehlo_version = false;
    for (const auto& attr : pjrt.GetAttributes()) {
      const std::string& key = attr.key();

      // Check for StableHLO version attributes
      if (key.find("stablehlo") != std::string::npos) {
        found_stablehlo_version = true;
      }

      std::cout << "      " << key << " = " << attr.value_as_string() << "\n";
    }

    if (!found_stablehlo_version) {
      std::cout << "      (No StableHLO version attributes found - plugin may not report them)\n";
    }

    // 3. Create client
    std::cout << "[3/5] Creating PJRT client...\n";
    auto client = pjrt.CreateClient();
    std::cout << "      OK\n";

    // 4. Get first addressable device (no hardcoded 0)
    std::cout << "[4/5] Getting first addressable device...\n";
    int device_id = client->FirstAddressableDeviceId();
    std::cout << "      Using device id: " << device_id << "\n";

    // 5. Read input file and compile
    std::cout << "[5/5] Compiling MLIR program...\n";

    std::string code = ReadFile(input_path.c_str());
    std::cout << "      Read " << code.size() << " bytes\n";

    std::string config = BuildCompileOptions(device_id);
    std::cout << "      Built compile options (" << config.size() << " bytes)\n";

    // Compile with format="mlir" (handles both text and bytecode)
    auto executable = client->CompileProgram(code, config, "mlir");

    std::cout << "      OK - Compiled successfully\n";
    std::cout << "\n";

    // Success summary
    std::cout << "=== SUCCESS ===\n";
    std::cout << "PJRT plugin compiled the MLIR program.\n";
    std::cout << "Executable outputs: " << executable->GetNumOutputs() << "\n";

    return 0;

  } catch (const lczero::PjrtException& e) {
    std::cerr << "\n";
    std::cerr << "=== FAILURE ===\n";
    std::cerr << "PJRT Error: " << e.what() << "\n";
    return 1;

  } catch (const lczero::Exception& e) {
    std::cerr << "\n";
    std::cerr << "=== FAILURE ===\n";
    std::cerr << "Lc0 Exception: " << e.what() << "\n";
    return 1;

  } catch (const std::exception& e) {
    std::cerr << "\n";
    std::cerr << "=== FAILURE ===\n";
    std::cerr << "Standard Exception: " << e.what() << "\n";
    return 1;
  }
}
