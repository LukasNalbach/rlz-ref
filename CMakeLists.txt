cmake_minimum_required(VERSION 3.21)

# relative Lempel-Ziv (RLZ) encodings of differentially encoded integer sequences (e.g., the differential suffix
# array) with sequential decoding from any position. The tests are built by default only if rlz-ref is the
# top-level project.
project(rlz-ref LANGUAGES C CXX)

# add external/string-utils, if the parent project has not added it
if(NOT TARGET string_utils)
  file(GLOB rlz_ref_string_utils_contents "${CMAKE_CURRENT_SOURCE_DIR}/external/string-utils/*")

  if(rlz_ref_string_utils_contents STREQUAL "")
    find_package(Git REQUIRED)
    message(STATUS "Downloading the git submodule external/string-utils")
    execute_process(COMMAND "${GIT_EXECUTABLE}" submodule update --init -- external/string-utils
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
  endif()

  # the prefix of the options and environment variables (RLZ_REF_MARCH_NATIVE, RLZ_REF_TEST_SECONDS, ...)
  if(PROJECT_IS_TOP_LEVEL)
    set(STRING_UTILS_PREFIX RLZ_REF)
    include("${CMAKE_CURRENT_SOURCE_DIR}/external/string-utils/cmake/settings.cmake")
  endif()

  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/external/string-utils" EXCLUDE_FROM_ALL)
endif()

# the libraries of string-utils beyond its core that are used here
string_utils_add_components(libsais gtl emhash)

option(RLZ_REF_BUILD_TESTS "Build the tests" ${PROJECT_IS_TOP_LEVEL})

add_library(rlz_ref INTERFACE)
target_include_directories(rlz_ref INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include/")
target_link_libraries(rlz_ref INTERFACE string_utils libsais libsais40 gtl emhash)

if(RLZ_REF_BUILD_TESTS)
  include(GoogleTest)
  enable_testing()
  string_utils_add_components(tests)

  string_utils_add_executable(test-rlz-ref tests/test-rlz-ref.cpp "${CMAKE_BINARY_DIR}/tests/" rlz_ref
    string_utils_test)
  set_target_properties(test-rlz-ref PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)

  gtest_discover_tests(test-rlz-ref)
endif()
