cmake_minimum_required(VERSION 3.8)
project(px4_pegasus)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Define executeable name
set(EXEC_NAME offboard_control)

# Ensure CMake prefers Conda libraries at runtime
set(CMAKE_INSTALL_RPATH "/home/airlab/miniconda3/envs/isaac_env/lib")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_SKIP_BUILD_RPATH FALSE)  # Ensure RPATH is used during build
set(CMAKE_BUILD_RPATH "/home/airlab/miniconda3/envs/isaac_env/lib")

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(std_msgs REQUIRED)

set(PYTHON_EXECUTABLE "/home/airlab/miniconda3/envs/isaac_env/bin/python3")
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Add executeable from source
add_executable(${EXEC_NAME} src/offboard_control.cpp)

# Handle ROS2 specific dependencies
ament_target_dependencies(${EXEC_NAME}
                          rclcpp
                          std_msgs
                          px4_msgs)

install(TARGETS ${EXEC_NAME} DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
