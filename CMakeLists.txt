cmake_minimum_required(VERSION 3.0)

project(unscented)

# use C++11
set(CMAKE_CXX_FLAGS "-std=c++11")

# Find Eigen
find_package(Eigen3 REQUIRED)

# Set output directory for targets
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Library
add_library(unscented INTERFACE)
target_include_directories(unscented INTERFACE include ${EIGEN3_INCLUDE_DIR})
target_compile_definitions(unscented INTERFACE 
  EIGEN_MATRIX_PLUGIN="unscented/matrix_plugins.h" 
  EIGEN_INITIALIZE_MATRICES_BY_ZERO=1
  )

# Airplane tracking example
add_executable(airplane_tracking examples/airplane_tracking/airplane_tracking.cpp)
target_include_directories(airplane_tracking PRIVATE examples/airplane_tracking)
target_link_libraries(airplane_tracking unscented)
target_compile_definitions(airplane_tracking PRIVATE
  EIGEN_MATRIX_PLUGIN="unscented/matrix_plugins.h" 
  EIGEN_INITIALIZE_MATRICES_BY_ZERO=1
  )

# Tests
enable_testing()
add_executable(test_ukf test/test_ukf.cpp) 
target_link_libraries(test_ukf unscented)
add_test(NAME test_ukf COMMAND test_ukf)
