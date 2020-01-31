# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# bazel_to_cmake: DO NOT EDIT

if(${IREE_BUILD_COMPILER})
  add_subdirectory(dialect)

  iree_bytecode_module(
    NAME
      strings_module_test
    SRC
      "strings_module_test.mlir"
    CC_NAMESPACE
      "iree::strings_module_test"
    TRANSLATE_TOOL
      iree_modules_strings_dialect_strings-translate
    TRANSLATION
      "-iree-mlir-to-vm-bytecode-module"
    PUBLIC
  )

  iree_cc_test(
    NAME
      strings_module_test
    SRCS
      "strings_module_test.cc"
    DEPS
      iree::modules::strings::strings_module_test_cc
      iree::modules::strings::strings_module
      iree::base::api
      iree::base::logging
      iree::hal::api
      iree::modules::hal
      iree::testing::gtest_main
      iree::vm
      iree::vm::bytecode_module
      iree::vm::ref
      absl::core_headers
      absl::strings
  )
endif()

iree_cc_library(
  NAME
    strings_module
  HDRS
    "strings_module.h"
  SRCS
    "strings_module.cc"
  DEPS
    iree::base::api
    iree::base::api_util
    iree::base::tracing
    iree::hal::api
    iree::hal::command_queue
    iree::hal::device
    iree::vm
    iree::vm::module_abi_cc
    absl::core_headers
    absl::memory
    absl::strings
    absl::span
  PUBLIC
)
