cmake_minimum_required(VERSION 3.0)

project(unscented)

# use C++11
set(CMAKE_CXX_FLAGS "-std=c++11")

# Find Eigen
find_package(Eigen3 REQUIRED)

# Library
add_library(unscented INTERFACE)
target_include_directories(unscented INTERFACE include impl ${EIGEN3_INCLUDE_DIR})

# Examples
# add_executable(robot_2d examples/robot_2d.cpp)
# target_link_libraries(robot_2d unscented)
# add_executable(robot_3d examples/robot_3d.cpp)
# target_link_libraries(robot_3d unscented)

# Install
install(TARGETS unscented DESTINATION bin)

# Testing
# enable_testing()
# add_executable(test_motion test/catch/catch_main.cpp test/test_motion.cpp)
# target_include_directories(test_motion PRIVATE test)
# target_link_libraries(test_motion unscented)
# add_test(NAME test_motion COMMAND test_motion)

