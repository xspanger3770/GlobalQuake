cmake_minimum_required(VERSION 3.10)
project(MyCudaProject)

# Find CUDA package
find_package(CUDA REQUIRED)

# Set CUDA architecture (adjust it to your target GPU architecture)
set(CUDA_ARCH_LIST "35" CACHE STRING "List of CUDA architectures (e.g., 35, 61, 75)")

# Set source directory
set(SRC_DIR ./src)

# Collect source files
file(GLOB SRC_FILES ${SRC_DIR}/*.c*)

# Set output directory for binaries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Set output directory for the shared library
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# CUDA-specific settings
foreach(arch ${CUDA_ARCH_LIST})
    set(CUDA_NVCC_FLAGS_${arch} "-gencode arch=compute_${arch},code=sm_${arch}")
endforeach()

# Compile CUDA sources into a shared library
cuda_add_library(my_cuda_lib SHARED ${SRC_FILES})

# Add an executable that links to the shared library
add_executable(my_cuda_app src/main.cpp)

include_directories(/usr/lib/jvm/java-17-openjdk-amd64/include/)
include_directories(/usr/lib/jvm/java-17-openjdk-amd64/include/linux)

target_link_libraries(my_cuda_app my_cuda_lib)

# Link any additional libraries if needed
# target_link_libraries(my_cuda_app PRIVATE library_name)

# Set C++11 standard
set_property(TARGET my_cuda_app PROPERTY CXX_STANDARD 11)
