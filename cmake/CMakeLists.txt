cmake_minimum_required(VERSION 3.10)
project(tensor_module LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build tensor.so from tensor.cpp
add_library(tensor MODULE ../lumine/csrc/tensor.cpp)

# Ensure tensor.so goes directly into build/
set_target_properties(tensor PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    PREFIX ""  # Remove 'lib' prefix
    SUFFIX ".so"  # Ensure correct extension
)
