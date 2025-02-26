cmake_minimum_required(VERSION 3.10)
project(tensor_module LANGUAGES CXX)
project(LumineLibrary VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_SHARED_LIBRARY_PREFIX "")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/)

# Source files
set(TENSOR_SRC ../lumine/csrc/tensor.cpp)
set(MATH_SRC ../lumine/csrc/cpu/math.cpp)
set(STREAM_SRC ../lumine/csrc/utils/stream.cpp)

# Include directories
include_directories(lumine)
include_directories(lumine/csrc)
include_directories(lumine/csrc/cpu)
include_directories(lumine/csrc/utils)

# Add position-independent code flag
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

# Create shared library
add_library(tensor SHARED ${TENSOR_SRC} ${STREAM_SRC} ${MATH_SRC})

# Set output properties
set_target_properties(tensor PROPERTIES
    OUTPUT_NAME "lumine"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    PREFIX ""  # Remove 'lib' prefix
    SUFFIX ".so"  # Ensure correct extension
)