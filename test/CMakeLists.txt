
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_executable(tuning-redist TuningRedist.cpp)


# Add CombBLAS
target_include_directories(tuning-redist PUBLIC $ENV{COMBBLAS_DIR}/include)
target_link_directories(tuning-redist PUBLIC $ENV{COMBBLAS_DIR}/lib)
target_link_libraries(tuning-redist PUBLIC -lCombBLAS -lGraphGenlib -lUsortlib)

# include ADS
target_include_directories(tuning-redist PUBLIC ../include/)

find_package(MPI REQUIRED)
find_package(OpenMP)
find_package(CUDA REQUIRED)
target_include_directories(tuning-redist PUBLIC ${CUDA_INCLUDE_DIRS})

if(TARGET MPI::MPI_CXX) # Use target if available (CMake >= 3.9)
  target_link_libraries(tuning-redist PUBLIC MPI::MPI_CXX)
else()
  target_compile_options(tuning-redist PUBLIC "${MPI_CXX_COMPILE_FLAGS}")
  target_link_libraries(tuning-redist PUBLIC "${MPI_CXX_LIBRARIES}" "${MPI_CXX_LINKFLAGS}")
  target_include_directories(tuning-redist PUBLIC "${MPI_CXX_INCLUDE_PATH}")
endif()

if(TARGET OpenMP::OpenMP_CXX) # Use target if available (CMake >= 3.9)
  target_compile_definitions(tuning-redist PUBLIC THREADED)
  target_link_libraries(tuning-redist PUBLIC OpenMP::OpenMP_CXX)
elseif(OPENMP_FOUND)
  target_compile_definitions(tuning-redist PUBLIC THREADED)
  target_compile_options(tuning-redist PUBLIC "${OpenMP_CXX_FLAGS}")
  target_link_libraries(tuning-redist PUBLIC "${OpenMP_CXX_FLAGS}")
endif()
