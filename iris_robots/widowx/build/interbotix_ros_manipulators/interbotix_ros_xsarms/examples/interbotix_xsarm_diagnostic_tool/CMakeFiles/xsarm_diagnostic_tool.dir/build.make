# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /iris/u/jyang27/dev/iris_robots/iris_robots/widowx/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /iris/u/jyang27/dev/iris_robots/iris_robots/widowx/build

# Include any dependencies generated for this target.
include interbotix_ros_manipulators/interbotix_ros_xsarms/examples/interbotix_xsarm_diagnostic_tool/CMakeFiles/xsarm_diagnostic_tool.dir/depend.make

# Include the progress variables for this target.
include interbotix_ros_manipulators/interbotix_ros_xsarms/examples/interbotix_xsarm_diagnostic_tool/CMakeFiles/xsarm_diagnostic_tool.dir/progress.make

# Include the compile flags for this target's objects.
include interbotix_ros_manipulators/interbotix_ros_xsarms/examples/interbotix_xsarm_diagnostic_tool/CMakeFiles/xsarm_diagnostic_tool.dir/flags.make

interbotix_ros_manipulators/interbotix_ros_xsarms/examples/interbotix_xsarm_diagnostic_tool/CMakeFiles/xsarm_diagnostic_tool.dir/src/xsarm_diagnostic_tool.cpp.o: interbotix_ros_manipulators/interbotix_ros_xsarms/examples/interbotix_xsarm_diagnostic_tool/CMakeFiles/xsarm_diagnostic_tool.dir/flags.make
interbotix_ros_manipulators/interbotix_ros_xsarms/examples/interbotix_xsarm_diagnostic_tool/CMakeFiles/xsarm_diagnostic_tool.dir/src/xsarm_diagnostic_tool.cpp.o: /iris/u/jyang27/dev/iris_robots/iris_robots/widowx/src/interbotix_ros_manipulators/interbotix_ros_xsarms/examples/interbotix_xsarm_diagnostic_tool/src/xsarm_diagnostic_tool.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object interbotix_ros_manipulators/interbotix_ros_xsarms/examples/interbotix_xsarm_diagnostic_tool/CMakeFiles/xsarm_diagnostic_tool.dir/src/xsarm_diagnostic_tool.cpp.o"
	cd /iris/u/jyang27/dev/iris_robots/iris_robots/widowx/build/interbotix_ros_manipulators/interbotix_ros_xsarms/examples/interbotix_xsarm_diagnostic_tool && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/xsarm_diagnostic_tool.dir/src/xsarm_diagnostic_tool.cpp.o -c /iris/u/jyang27/dev/iris_robots/iris_robots/widowx/src/interbotix_ros_manipulators/interbotix_ros_xsarms/examples/interbotix_xsarm_diagnostic_tool/src/xsarm_diagnostic_tool.cpp

interbotix_ros_manipulators/interbotix_ros_xsarms/examples/interbotix_xsarm_diagnostic_tool/CMakeFiles/xsarm_diagnostic_tool.dir/src/xsarm_diagnostic_tool.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/xsarm_diagnostic_tool.dir/src/xsarm_diagnostic_tool.cpp.i"
	cd /iris/u/jyang27/dev/iris_robots/iris_robots/widowx/build/interbotix_ros_manipulators/interbotix_ros_xsarms/examples/interbotix_xsarm_diagnostic_tool && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /iris/u/jyang27/dev/iris_robots/iris_robots/widowx/src/interbotix_ros_manipulators/interbotix_ros_xsarms/examples/interbotix_xsarm_diagnostic_tool/src/xsarm_diagnostic_tool.cpp > CMakeFiles/xsarm_diagnostic_tool.dir/src/xsarm_diagnostic_tool.cpp.i

interbotix_ros_manipulators/interbotix_ros_xsarms/examples/interbotix_xsarm_diagnostic_tool/CMakeFiles/xsarm_diagnostic_tool.dir/src/xsarm_diagnostic_tool.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/xsarm_diagnostic_tool.dir/src/xsarm_diagnostic_tool.cpp.s"
	cd /iris/u/jyang27/dev/iris_robots/iris_robots/widowx/build/interbotix_ros_manipulators/interbotix_ros_xsarms/examples/interbotix_xsarm_diagnostic_tool && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /iris/u/jyang27/dev/iris_robots/iris_robots/widowx/src/interbotix_ros_manipulators/interbotix_ros_xsarms/examples/interbotix_xsarm_diagnostic_tool/src/xsarm_diagnostic_tool.cpp -o CMakeFiles/xsarm_diagnostic_tool.dir/src/xsarm_diagnostic_tool.cpp.s

# Object files for target xsarm_diagnostic_tool
xsarm_diagnostic_tool_OBJECTS = \
"CMakeFiles/xsarm_diagnostic_tool.dir/src/xsarm_diagnostic_tool.cpp.o"

# External object files for target xsarm_diagnostic_tool
xsarm_diagnostic_tool_EXTERNAL_OBJECTS =

/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: interbotix_ros_manipulators/interbotix_ros_xsarms/examples/interbotix_xsarm_diagnostic_tool/CMakeFiles/xsarm_diagnostic_tool.dir/src/xsarm_diagnostic_tool.cpp.o
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: interbotix_ros_manipulators/interbotix_ros_xsarms/examples/interbotix_xsarm_diagnostic_tool/CMakeFiles/xsarm_diagnostic_tool.dir/build.make
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/libinterbotix_xs_sdk.so
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/libdynamixel_workbench_toolbox.so
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /opt/ros/noetic/lib/libdynamixel_sdk.so
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /opt/ros/noetic/lib/librobot_state_publisher_solver.so
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /opt/ros/noetic/lib/libjoint_state_listener.so
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /opt/ros/noetic/lib/libkdl_parser.so
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /usr/lib/liborocos-kdl.so
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /opt/ros/noetic/lib/librviz.so
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /usr/lib/x86_64-linux-gnu/libOgreOverlay.so
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /usr/lib/x86_64-linux-gnu/libOgreMain.so
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /usr/lib/x86_64-linux-gnu/libOpenGL.so
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /usr/lib/x86_64-linux-gnu/libGLX.so
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /usr/lib/x86_64-linux-gnu/libGLU.so
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /opt/ros/noetic/lib/libimage_transport.so
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /opt/ros/noetic/lib/libinteractive_markers.so
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /opt/ros/noetic/lib/liblaser_geometry.so
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /opt/ros/noetic/lib/libtf.so
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /opt/ros/noetic/lib/libresource_retriever.so
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /opt/ros/noetic/lib/libtf2_ros.so
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /opt/ros/noetic/lib/libactionlib.so
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /opt/ros/noetic/lib/libmessage_filters.so
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /opt/ros/noetic/lib/libtf2.so
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /opt/ros/noetic/lib/liburdf.so
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /usr/lib/x86_64-linux-gnu/liburdfdom_sensor.so
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /usr/lib/x86_64-linux-gnu/liburdfdom_model_state.so
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /usr/lib/x86_64-linux-gnu/liburdfdom_model.so
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /usr/lib/x86_64-linux-gnu/liburdfdom_world.so
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /usr/lib/x86_64-linux-gnu/libtinyxml.so
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /opt/ros/noetic/lib/librosconsole_bridge.so
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /opt/ros/noetic/lib/librosbag.so
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /opt/ros/noetic/lib/librosbag_storage.so
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /opt/ros/noetic/lib/libclass_loader.so
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /usr/lib/x86_64-linux-gnu/libPocoFoundation.so
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /usr/lib/x86_64-linux-gnu/libdl.so
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /opt/ros/noetic/lib/libroslib.so
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /opt/ros/noetic/lib/librospack.so
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /usr/lib/x86_64-linux-gnu/libpython3.8.so
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /usr/lib/x86_64-linux-gnu/libboost_program_options.so.1.71.0
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /usr/lib/x86_64-linux-gnu/libtinyxml2.so
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /opt/ros/noetic/lib/libroslz4.so
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /usr/lib/x86_64-linux-gnu/liblz4.so
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /opt/ros/noetic/lib/libtopic_tools.so
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /opt/ros/noetic/lib/libroscpp.so
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /usr/lib/x86_64-linux-gnu/libpthread.so
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /usr/lib/x86_64-linux-gnu/libboost_chrono.so.1.71.0
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so.1.71.0
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /opt/ros/noetic/lib/librosconsole.so
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /opt/ros/noetic/lib/librosconsole_log4cxx.so
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /opt/ros/noetic/lib/librosconsole_backend_interface.so
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /usr/lib/x86_64-linux-gnu/liblog4cxx.so
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /usr/lib/x86_64-linux-gnu/libboost_regex.so.1.71.0
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /opt/ros/noetic/lib/libxmlrpcpp.so
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /opt/ros/noetic/lib/libroscpp_serialization.so
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /opt/ros/noetic/lib/librostime.so
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /usr/lib/x86_64-linux-gnu/libboost_date_time.so.1.71.0
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /opt/ros/noetic/lib/libcpp_common.so
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /usr/lib/x86_64-linux-gnu/libboost_system.so.1.71.0
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /usr/lib/x86_64-linux-gnu/libboost_thread.so.1.71.0
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: /usr/lib/x86_64-linux-gnu/libconsole_bridge.so.0.4
/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool: interbotix_ros_manipulators/interbotix_ros_xsarms/examples/interbotix_xsarm_diagnostic_tool/CMakeFiles/xsarm_diagnostic_tool.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/iris/u/jyang27/dev/iris_robots/iris_robots/widowx/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable /iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool"
	cd /iris/u/jyang27/dev/iris_robots/iris_robots/widowx/build/interbotix_ros_manipulators/interbotix_ros_xsarms/examples/interbotix_xsarm_diagnostic_tool && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/xsarm_diagnostic_tool.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
interbotix_ros_manipulators/interbotix_ros_xsarms/examples/interbotix_xsarm_diagnostic_tool/CMakeFiles/xsarm_diagnostic_tool.dir/build: /iris/u/jyang27/dev/iris_robots/iris_robots/widowx/devel/lib/interbotix_xsarm_diagnostic_tool/xsarm_diagnostic_tool

.PHONY : interbotix_ros_manipulators/interbotix_ros_xsarms/examples/interbotix_xsarm_diagnostic_tool/CMakeFiles/xsarm_diagnostic_tool.dir/build

interbotix_ros_manipulators/interbotix_ros_xsarms/examples/interbotix_xsarm_diagnostic_tool/CMakeFiles/xsarm_diagnostic_tool.dir/clean:
	cd /iris/u/jyang27/dev/iris_robots/iris_robots/widowx/build/interbotix_ros_manipulators/interbotix_ros_xsarms/examples/interbotix_xsarm_diagnostic_tool && $(CMAKE_COMMAND) -P CMakeFiles/xsarm_diagnostic_tool.dir/cmake_clean.cmake
.PHONY : interbotix_ros_manipulators/interbotix_ros_xsarms/examples/interbotix_xsarm_diagnostic_tool/CMakeFiles/xsarm_diagnostic_tool.dir/clean

interbotix_ros_manipulators/interbotix_ros_xsarms/examples/interbotix_xsarm_diagnostic_tool/CMakeFiles/xsarm_diagnostic_tool.dir/depend:
	cd /iris/u/jyang27/dev/iris_robots/iris_robots/widowx/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /iris/u/jyang27/dev/iris_robots/iris_robots/widowx/src /iris/u/jyang27/dev/iris_robots/iris_robots/widowx/src/interbotix_ros_manipulators/interbotix_ros_xsarms/examples/interbotix_xsarm_diagnostic_tool /iris/u/jyang27/dev/iris_robots/iris_robots/widowx/build /iris/u/jyang27/dev/iris_robots/iris_robots/widowx/build/interbotix_ros_manipulators/interbotix_ros_xsarms/examples/interbotix_xsarm_diagnostic_tool /iris/u/jyang27/dev/iris_robots/iris_robots/widowx/build/interbotix_ros_manipulators/interbotix_ros_xsarms/examples/interbotix_xsarm_diagnostic_tool/CMakeFiles/xsarm_diagnostic_tool.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : interbotix_ros_manipulators/interbotix_ros_xsarms/examples/interbotix_xsarm_diagnostic_tool/CMakeFiles/xsarm_diagnostic_tool.dir/depend

