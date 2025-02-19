cmake_minimum_required(VERSION 3.10)
project(LumineLibrary VERSION 1.0)

set(CMAKE_SHARED_LIBRARY_PREFIX "")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/)

set(TENSOR_SRC ../lumine/csrc/tensor.cpp)
set(STREAM_SRC ../lumine/csrc/utils/stream.cpp)

include_directories(lumine/csrc)
include_directories(lumine/csrc/utils)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

add_library(tensor SHARED ${TENSOR_SRC} ${STREAM_SRC})

set_target_properties(tensor PROPERTIES OUTPUT_NAME "lumine")

