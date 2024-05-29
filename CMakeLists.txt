
cmake_minimum_required(VERSION 3.20)

project(ADS VERSION 0.0.1 LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# require c++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS OFF)

# Main executable
add_executable(ADS src/autotuning.cpp)

target_compile_options(ADS PUBLIC -Wno-deprecated-declarations)

# Add CombBLAS
target_include_directories(ADS PUBLIC $ENV{COMBBLAS_DIR}/include)
target_link_directories(ADS PUBLIC $ENV{COMBBLAS_DIR}/lib)
target_link_libraries(ADS PUBLIC -lCombBLAS -lGraphGenlib -lUsortlib)

# Include main autotuning directory
target_include_directories(ADS PUBLIC include)

find_package(MPI REQUIRED)
find_package(OpenMP)
find_package(CUDA REQUIRED)
target_include_directories(ADS PUBLIC ${CUDA_INCLUDE_DIRS})


# XGB, MPI, OpenMP
if(DEFINED XGB_MODEL)
    add_compile_definitions(XGB_MODEL)
    find_package(xgboost REQUIRED)
    target_link_libraries(ADS PUBLIC xgboost::xgboost)
endif()

if(TARGET MPI::MPI_CXX) # Use target if available (CMake >= 3.9)
  target_link_libraries(ADS PUBLIC MPI::MPI_CXX)
else()
  target_compile_options(ADS PUBLIC "${MPI_CXX_COMPILE_FLAGS}")
  target_link_libraries(ADS PUBLIC "${MPI_CXX_LIBRARIES}" "${MPI_CXX_LINKFLAGS}")
  target_include_directories(ADS PUBLIC "${MPI_CXX_INCLUDE_PATH}")
endif()

if(TARGET OpenMP::OpenMP_CXX) # Use target if available (CMake >= 3.9)
  target_compile_definitions(ADS PUBLIC THREADED)
  target_link_libraries(ADS PUBLIC OpenMP::OpenMP_CXX)
elseif(OPENMP_FOUND)
  target_compile_definitions(ADS PUBLIC THREADED)
  target_compile_options(ADS PUBLIC "${OpenMP_CXX_FLAGS}")
  target_link_libraries(ADS PUBLIC "${OpenMP_CXX_FLAGS}")
endif()

add_subdirectory(test)
