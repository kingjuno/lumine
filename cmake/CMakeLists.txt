cmake_minimum_required(VERSION 3.10)
project(ctensor LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define the source file
set(SRC ${CMAKE_SOURCE_DIR}/../lumine/csrc/tensor.cpp)

# Force output directory to be directly under the project root
set(OUT_DIR ${CMAKE_SOURCE_DIR}/../build)

# Create shared library
add_library(ctensor MODULE ${SRC})

# Set library output directory directly to build/
set_target_properties(ctensor PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${OUT_DIR}"
    OUTPUT_NAME "tensor"
    PREFIX ""
)

# Set CMake runtime output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUT_DIR})
