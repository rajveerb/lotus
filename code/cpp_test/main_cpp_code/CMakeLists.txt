cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(main)

find_package(Torch REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


add_library(resnet resnet.cpp)
target_link_libraries(resnet "${TORCH_LIBRARIES}")

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}" "${OpenCV_LIBS}" Threads::Threads resnet)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)


# cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
# project(example_app)

# set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
# set_target_properties(example_app PROPERTIES CUDA_ARCHITECTURES "35;50;72")
# set(CMAKE_PREFIX_PATH "libtorch/share/cmake/Torch")
# find_package(Torch REQUIRED)
# find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
# add_executable(${PROJECT_NAME} example_app.cpp)
# target_link_libraries(${PROJECT_NAME} "${OpenCV_LIBS}")
# target_link_libraries(${PROJECT_NAME}  "${TORCH_LIBRARIES}")
# set_property(TARGET example_app PROPERTY CXX_STANDARD 17)