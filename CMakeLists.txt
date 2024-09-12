cmake_minimum_required(VERSION 3.14)

project(unscented)

# use C++17
set(CMAKE_CXX_FLAGS "-std=c++17")

include(FetchContent)

FetchContent_Declare(matplotplusplus
  GIT_REPOSITORY https://github.com/alandefreitas/matplotplusplus
  GIT_TAG origin/master)

FetchContent_GetProperties(matplotplusplus)
if(NOT matplotplusplus_POPULATED)
    FetchContent_Populate(matplotplusplus)
    add_subdirectory(${matplotplusplus_SOURCE_DIR} ${matplotplusplus_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# Find Eigen
find_package(Eigen3 REQUIRED)

# Library
add_library(unscented INTERFACE)
target_include_directories(unscented INTERFACE include ${EIGEN3_INCLUDE_DIR})
target_compile_definitions(unscented INTERFACE 
  EIGEN_MATRIX_PLUGIN="unscented/matrix_plugins.h" 
  EIGEN_INITIALIZE_MATRICES_BY_ZERO=1
  )

# Airplane tracking example
add_executable(airplane_tracking examples/airplane_tracking.cpp)
target_link_libraries(airplane_tracking PUBLIC unscented matplot)

# Robot localization example
add_executable(robot_localization examples/robot_localization.cpp)
target_link_libraries(robot_localization PUBLIC unscented matplot)

# Tests
# enable_testing()
# add_executable(test_ukf test/test_ukf.cpp) 
# target_link_libraries(test_ukf unscented)
# add_test(NAME test_ukf COMMAND test_ukf)

add_executable(playground examples/playground.cpp)
target_link_libraries(playground unscented)
