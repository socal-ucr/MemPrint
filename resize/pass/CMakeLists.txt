cmake_minimum_required(VERSION 3.13)
project(MemPrintResize LANGUAGES CXX)

# cmake -S resize/pass -B build/pass -DLLVM_DIR=$HOME/llvm-project/build/lib/cmake/llvm && cmake --build build/pass
find_package(LLVM REQUIRED CONFIG)
message(STATUS "LLVM ${LLVM_PACKAGE_VERSION} from ${LLVM_DIR}")

add_library(MemPrintResize MODULE MemPrintResize.cpp)
target_compile_features(MemPrintResize PRIVATE cxx_std_17)
target_include_directories(MemPrintResize SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})
target_compile_definitions(MemPrintResize PRIVATE ${LLVM_DEFINITIONS})
if (NOT LLVM_ENABLE_RTTI)
  target_compile_options(MemPrintResize PRIVATE -fno-rtti)
endif()
