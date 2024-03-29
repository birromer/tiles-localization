cmake_minimum_required(VERSION 3.2.0)
project(tiles_loc)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++14)
set(CMAKE_CXX_STANDARD 14)
add_compile_options(-ggdb)

#set(CMAKE_PREFIX_PATH "/opt/opencv3;/opt/ros/melodic;/home/birromer/Documents/tubex-lib/build_install")

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
include(FindPkgConfig)

set(image_transport_DIR /opt/ros/noetic/share/image_transport/cmake/image_transportConfig.cmake)

# Adding ROOT
#
  find_package(ROOT REQUIRED)
  include_directories(${ROOT_INCLUDE_DIRS})
  message(STATUS "Found ROOT version ${ROOT_VERSION}")
  message(STATUS "  libraries   : ${ROOT_LIBRARIES}")

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

# Compilation

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  rospy
  std_msgs
  message_generation
)

set(OpenCV_DIR /opt/opencv3)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)



## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a exec_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a exec_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
add_message_files(
   FILES
   Observation.msg
   Cmd.msg
   State.msg
 )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

generate_messages(
  DEPENDENCIES
  std_msgs
)

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a exec_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES image
  CATKIN_DEPENDS
    message_generation
    message_runtime
    cv_bridge
    image_transport
    roscpp
    rospy
    std_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/image.cpp
# )

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
# add_executable(${PROJECT_NAME}_node src/image_node.cpp)

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
# target_link_libraries(${PROJECT_NAME}_node
#   ${catkin_LIBRARIES}
# )

add_executable(base src/robot.cpp)
add_executable(loc src/localization.cpp)

#add_executable(estim src/estimator.cpp)
add_executable(cmd src/command.cpp)
add_executable(ctrl src/controller.cpp)
add_executable(view src/viewer.cpp)

target_link_libraries(base opencv_highgui ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CODAC_LIBRARIES} Ibex::ibex ${OpenCV_LIBS} ${CODAC_LIBRARIES} ${OpenCV_LIBS} ${catkin_LIBRARIES} ${OpenCV_LIBS} Ibex::ibex ${OpenCV_LIBS} ${CODAC_LIBRARIES} ${OpenCV_LIBS})
target_compile_options(base PUBLIC ${CODAC_CFLAGS})
target_include_directories(base PUBLIC ${CODAC_INCLUDE_DIRS})

target_link_libraries(loc opencv_highgui ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CODAC_LIBRARIES} Ibex::ibex ${OpenCV_LIBS} ${CODAC_LIBRARIES} ${OpenCV_LIBS} ${catkin_LIBRARIES} ${OpenCV_LIBS} Ibex::ibex ${OpenCV_LIBS} ${CODAC_LIBRARIES} ${OpenCV_LIBS})
target_compile_options(loc PUBLIC ${CODAC_CFLAGS})
target_include_directories(loc PUBLIC ${CODAC_INCLUDE_DIRS})

target_link_libraries(view ${catkin_LIBRARIES} ${ROOT_LIBRARIES} ${CODAC_LIBRARIES} Ibex::ibex ${CODAC_LIBRARIES} ${catkin_LIBRARIES} ${CODAC_LIBRARIES} Ibex::ibex ${CODAC_LIBRARIES})
target_compile_options(view PUBLIC ${CODAC_CFLAGS})
target_include_directories(view PUBLIC ${CODAC_INCLUDE_DIRS} ${ROOT_INCLUDE_DIRS})

target_link_libraries(cmd ${catkin_LIBRARIES})
target_link_libraries(ctrl ${catkin_LIBRARIES})

#target_link_libraries(estim opencv_highgui ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CODAC_LIBRARIES} Ibex::ibex ${OpenCV_LIBS} ${CODAC_LIBRARIES} ${OpenCV_LIBS} ${catkin_LIBRARIES} ${OpenCV_LIBS} Ibex::ibex ${OpenCV_LIBS} ${CODAC_LIBRARIES} ${OpenCV_LIBS})
#target_compile_options(estim PUBLIC ${CODAC_CFLAGS})
#target_include_directories(estim PUBLIC ${CODAC_INCLUDE_DIRS})

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html
# install(TARGETS ${PROJECT_NAME}_node
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark libraries for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_libraries.html
# install(TARGETS ${PROJECT_NAME}
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_image.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
