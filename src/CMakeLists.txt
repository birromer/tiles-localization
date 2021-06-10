cmake_minimum_required(VERSION 3.2.0)
project(test_param)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

#set(CMAKE_PREFIX_PATH "/opt/opencv3;/opt/ros/melodic;/home/birromer/Documents/tubex-lib/build_install")

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
include(FindPkgConfig)

# Adding Eigen3

  # In case you installed Eigen3 in a local directory, you need
  # to specify its path with the CMAKE_PREFIX_PATH option, e.g.
  # set(CMAKE_PREFIX_PATH "~/eigen/build_install")

  find_package(Eigen3 REQUIRED NO_MODULE)
  message(STATUS "Found Eigen3 version ${EIGEN3_VERSION}")

# Adding Tubex

  # In case you installed Tubex in a local directory, you need
  # to specify its path with the CMAKE_PREFIX_PATH option.
  # set(CMAKE_PREFIX_PATH "~/tubex-lib/build_install")
  #  set(CODAC_FOUND 1)

  find_package(CODAC REQUIRED)
  #pkg_search_module(CODAC REQUIRED tubex)
  message(STATUS "Found Tubex version ${CODAC_VERSION}")

# Adding IBEX

  # In case you installed IBEX in a local directory, you need
  # to specify its path with the CMAKE_PREFIX_PATH option.
  # set(CMAKE_PREFIX_PATH "~/ibex-lib/build_install")

  find_package(IBEX REQUIRED)
  ibex_init_common() # IBEX should have installed this function
  message(STATUS "Found IBEX version ${IBEX_VERSION}")

# Adding OpenCV

  set(OpenCV_DIR /opt/opencv3)
  find_package(OpenCV REQUIRED)
  include_directories(${OpenCV_INCLUDE_DIRS})
  message(STATUS "OpenCV library status:")
  message(STATUS "    version: ${OpenCV_VERSION}")
  message(STATUS "    libraries: ${OpenCV_LIBS}")
  message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# Compilation

  add_executable(${PROJECT_NAME} test_param.cpp)
  target_compile_options(${PROJECT_NAME} PUBLIC ${CODAC_CXX_FLAGS})
  target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC ${CODAC_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS})
  target_link_libraries(${PROJECT_NAME} opencv_highgui ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CODAC_LIBRARIES} Ibex::ibex ${OpenCV_LIBS} ${CODAC_LIBRARIES} ${OpenCV_LIBS} ${catkin_LIBRARIES} ${OpenCV_LIBS} Ibex::ibex ${OpenCV_LIBS} ${CODAC_LIBRARIES} ${OpenCV_LIBS})
