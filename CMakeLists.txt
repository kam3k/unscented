cmake_minimum_required(VERSION 3.0)

project(unscented)

# use C++11
set(CMAKE_CXX_FLAGS "-std=c++11")

# Find Eigen
find_package(Eigen3 REQUIRED)

# Find python
find_package(PythonLibs 2.7)

# Library
add_library(unscented src/instantiations.cpp)
target_include_directories(unscented PUBLIC include impl ${EIGEN3_INCLUDE_DIR})

# Airplane tracking example
add_executable(airplane_tracking examples/airplane_tracking.cpp)
target_include_directories(airplane_tracking PRIVATE depends ${PYTHON_INCLUDE_DIRS})
target_link_libraries(airplane_tracking unscented ${PYTHON_LIBRARIES})

# Robot localization example
add_executable(robot_localization_2d examples/robot_localization_2d.cpp)
target_link_libraries(robot_localization_2d unscented)

# Install
install(TARGETS unscented DESTINATION bin)

# Testing
# enable_testing()
# add_executable(test_motion test/catch/catch_main.cpp test/test_motion.cpp)
# target_include_directories(test_motion PRIVATE test)
# target_link_libraries(test_motion unscented)
# add_test(NAME test_motion COMMAND test_motion)

