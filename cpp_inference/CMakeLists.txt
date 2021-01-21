cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(inference)

set(Torch_DIR /home/yxs/Documents/libtorch)
find_package(Torch REQUIRED)

add_executable(inference inference.cpp)
target_link_libraries(inference "${TORCH_LIBRARIES}")
set_property(TARGET inference PROPERTY CXX_STANDARD 14)