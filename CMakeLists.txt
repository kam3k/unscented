cmake_minimum_required(VERSION 3.0)

project(unscented)

# use C++11
set(CMAKE_CXX_FLAGS "-std=c++11")

# Find Eigen
find_package(Eigen3 REQUIRED)

# Find Python Libraries
find_package(PythonLibs 2.7)

# Library
add_library(unscented INTERFACE)
target_include_directories(unscented INTERFACE include ${EIGEN3_INCLUDE_DIR})

# Airplane tracking example
add_executable(airplane_tracking examples/airplane_tracking.cpp)
target_include_directories(airplane_tracking PRIVATE optional ${PYTHON_INCLUDE_DIRS})
target_link_libraries(airplane_tracking unscented ${PYTHON_LIBRARIES})

# Robot localization example
add_executable(robot_localization_2d examples/robot_localization_2d.cpp)
target_link_libraries(robot_localization_2d unscented)
