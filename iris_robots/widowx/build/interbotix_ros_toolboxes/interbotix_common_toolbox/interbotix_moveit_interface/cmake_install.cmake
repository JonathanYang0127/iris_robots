# Install script for directory: /iris/u/jyang27/dev/iris_robots/iris_robots/widowx/src/interbotix_ros_toolboxes/interbotix_common_toolbox/interbotix_moveit_interface

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/interbotix_moveit_interface/srv" TYPE FILE FILES "/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/src/interbotix_ros_toolboxes/interbotix_common_toolbox/interbotix_moveit_interface/srv/MoveItPlan.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/interbotix_moveit_interface/cmake" TYPE FILE FILES "/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/build/interbotix_ros_toolboxes/interbotix_common_toolbox/interbotix_moveit_interface/catkin_generated/installspace/interbotix_moveit_interface-msg-paths.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/include/interbotix_moveit_interface")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/roseus/ros" TYPE DIRECTORY FILES "/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/share/roseus/ros/interbotix_moveit_interface")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/common-lisp/ros" TYPE DIRECTORY FILES "/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/share/common-lisp/ros/interbotix_moveit_interface")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/gennodejs/ros" TYPE DIRECTORY FILES "/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/share/gennodejs/ros/interbotix_moveit_interface")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  execute_process(COMMAND "/usr/bin/python3" -m compileall "/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/python3/dist-packages/interbotix_moveit_interface")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python3/dist-packages" TYPE DIRECTORY FILES "/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/python3/dist-packages/interbotix_moveit_interface")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/build/interbotix_ros_toolboxes/interbotix_common_toolbox/interbotix_moveit_interface/catkin_generated/installspace/interbotix_moveit_interface.pc")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/interbotix_moveit_interface/cmake" TYPE FILE FILES "/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/build/interbotix_ros_toolboxes/interbotix_common_toolbox/interbotix_moveit_interface/catkin_generated/installspace/interbotix_moveit_interface-msg-extras.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/interbotix_moveit_interface/cmake" TYPE FILE FILES
    "/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/build/interbotix_ros_toolboxes/interbotix_common_toolbox/interbotix_moveit_interface/catkin_generated/installspace/interbotix_moveit_interfaceConfig.cmake"
    "/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/build/interbotix_ros_toolboxes/interbotix_common_toolbox/interbotix_moveit_interface/catkin_generated/installspace/interbotix_moveit_interfaceConfig-version.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/interbotix_moveit_interface" TYPE FILE FILES "/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/src/interbotix_ros_toolboxes/interbotix_common_toolbox/interbotix_moveit_interface/package.xml")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/interbotix_moveit_interface" TYPE PROGRAM FILES "/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/build/interbotix_ros_toolboxes/interbotix_common_toolbox/interbotix_moveit_interface/catkin_generated/installspace/moveit_interface_gui")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/interbotix_moveit_interface" TYPE PROGRAM FILES "/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/build/interbotix_ros_toolboxes/interbotix_common_toolbox/interbotix_moveit_interface/catkin_generated/installspace/moveit_python_interface")
endif()

