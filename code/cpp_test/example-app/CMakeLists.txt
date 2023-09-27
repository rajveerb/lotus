cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(example_app)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
# set_target_properties(example_app PROPERTIES CUDA_ARCHITECTURES "35;50;72")
# set(CMAKE_PREFIX_PATH "libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
add_executable(${PROJECT_NAME} example_app.cpp)
target_link_libraries(${PROJECT_NAME} "${OpenCV_LIBS}")
target_link_libraries(${PROJECT_NAME}  "${TORCH_LIBRARIES}")
set_property(TARGET example_app PROPERTY CXX_STANDARD 17)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET example_app
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:example_app>)
endif (MSVC)