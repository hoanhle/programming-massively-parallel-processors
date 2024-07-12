cmake_minimum_required(VERSION 3.22)

# Specify the project and the languages used
project(programming_massively_parallel_processors LANGUAGES CXX CUDA)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Specify the CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 86)

# Specify the path to the CUDA compiler
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

# Collect all source files (in this case only the CUDA file)
file(GLOB_RECURSE SOURCES
        ${CMAKE_SOURCE_DIR}/exercises/*.cu
)

# Create an executable target for each .cu file
foreach(SOURCE_FILE ${SOURCES})
    # Extract the file name without extension
    get_filename_component(FILENAME ${SOURCE_FILE} NAME_WE)

    # Create an executable for each source file
    add_executable(${FILENAME} ${SOURCE_FILE})
endforeach()