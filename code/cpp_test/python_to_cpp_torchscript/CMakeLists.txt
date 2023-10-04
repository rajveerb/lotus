cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

add_executable(model_deserialiser model_deserialiser.cpp)
target_link_libraries(model_deserialiser "${TORCH_LIBRARIES}")
set_property(TARGET model_deserialiser PROPERTY CXX_STANDARD 17)