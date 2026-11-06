cmake_minimum_required(VERSION 3.25)
project(directml_native_checks LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(TORCH_ROOT "" CACHE PATH "Target PyTorch torch directory, including runtime DLLs")

add_library(native_checks SHARED check_extension.cpp)
target_compile_options(native_checks PRIVATE
    "$<$<CXX_COMPILER_ID:Clang>:/clang:-fms-compatibility-version=19.41>")
target_include_directories(native_checks PRIVATE
    "${TORCH_ROOT}/include" "${TORCH_ROOT}/include/torch/csrc/api/include")
target_link_libraries(native_checks PRIVATE
    "${TORCH_ROOT}/lib/c10.lib" "${TORCH_ROOT}/lib/torch_cpu.lib"
    kernel32)
