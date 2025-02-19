cmake_minimum_required(VERSION 3.18)
project(spline_psf_gpu LANGUAGES C CXX CUDA)

find_package(pybind11 REQUIRED)
find_package(CUDA REQUIRED)

# Activer la compilation séparée pour CUDA avec -rdc=true
set(CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true")

# Définir les sources
set(SOURCES
    pybind_spline.cpp
    spline_psf.c       # fichier C
    spline_psf_gpu.cu  # fichier CUDA
)

pybind11_add_module(spline_psf_gpu ${SOURCES})

target_include_directories(spline_psf_gpu PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

set_target_properties(spline_psf_gpu PROPERTIES
    CUDA_ARCHITECTURES "86"  # Adapté à votre GPU
    CXX_STANDARD 11
    CUDA_RUNTIME_LIBRARY "Shared"
)

# Lier explicitement cudadevrt (ajustez le chemin si nécessaire)
target_link_libraries(spline_psf_gpu PRIVATE "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/lib/x64/cudadevrt.lib")
