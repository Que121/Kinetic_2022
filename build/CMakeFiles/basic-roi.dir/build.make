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
CMAKE_SOURCE_DIR = /home/nuc/Desktop/sentry_2022

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/nuc/Desktop/sentry_2022/build

# Include any dependencies generated for this target.
include CMakeFiles/basic-roi.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/basic-roi.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/basic-roi.dir/flags.make

CMakeFiles/basic-roi.dir/module/roi/basic_roi.cpp.o: CMakeFiles/basic-roi.dir/flags.make
CMakeFiles/basic-roi.dir/module/roi/basic_roi.cpp.o: ../module/roi/basic_roi.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nuc/Desktop/sentry_2022/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/basic-roi.dir/module/roi/basic_roi.cpp.o"
	/bin/x86_64-linux-gnu-g++-9  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/basic-roi.dir/module/roi/basic_roi.cpp.o -c /home/nuc/Desktop/sentry_2022/module/roi/basic_roi.cpp

CMakeFiles/basic-roi.dir/module/roi/basic_roi.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/basic-roi.dir/module/roi/basic_roi.cpp.i"
	/bin/x86_64-linux-gnu-g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nuc/Desktop/sentry_2022/module/roi/basic_roi.cpp > CMakeFiles/basic-roi.dir/module/roi/basic_roi.cpp.i

CMakeFiles/basic-roi.dir/module/roi/basic_roi.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/basic-roi.dir/module/roi/basic_roi.cpp.s"
	/bin/x86_64-linux-gnu-g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nuc/Desktop/sentry_2022/module/roi/basic_roi.cpp -o CMakeFiles/basic-roi.dir/module/roi/basic_roi.cpp.s

# Object files for target basic-roi
basic__roi_OBJECTS = \
"CMakeFiles/basic-roi.dir/module/roi/basic_roi.cpp.o"

# External object files for target basic-roi
basic__roi_EXTERNAL_OBJECTS =

libbasic-roi.so: CMakeFiles/basic-roi.dir/module/roi/basic_roi.cpp.o
libbasic-roi.so: CMakeFiles/basic-roi.dir/build.make
libbasic-roi.so: /opt/intel/openvino_2021/opencv/lib/libopencv_gapi.so.4.5.3
libbasic-roi.so: /opt/intel/openvino_2021/opencv/lib/libopencv_highgui.so.4.5.3
libbasic-roi.so: /opt/intel/openvino_2021/opencv/lib/libopencv_ml.so.4.5.3
libbasic-roi.so: /opt/intel/openvino_2021/opencv/lib/libopencv_objdetect.so.4.5.3
libbasic-roi.so: /opt/intel/openvino_2021/opencv/lib/libopencv_photo.so.4.5.3
libbasic-roi.so: /opt/intel/openvino_2021/opencv/lib/libopencv_stitching.so.4.5.3
libbasic-roi.so: /opt/intel/openvino_2021/opencv/lib/libopencv_video.so.4.5.3
libbasic-roi.so: /opt/intel/openvino_2021/opencv/lib/libopencv_videoio.so.4.5.3
libbasic-roi.so: /opt/intel/openvino_2021/opencv/lib/libopencv_dnn.so.4.5.3
libbasic-roi.so: /opt/intel/openvino_2021/opencv/lib/libopencv_imgcodecs.so.4.5.3
libbasic-roi.so: /opt/intel/openvino_2021/opencv/lib/libopencv_calib3d.so.4.5.3
libbasic-roi.so: /opt/intel/openvino_2021/opencv/lib/libopencv_features2d.so.4.5.3
libbasic-roi.so: /opt/intel/openvino_2021/opencv/lib/libopencv_flann.so.4.5.3
libbasic-roi.so: /opt/intel/openvino_2021/opencv/lib/libopencv_imgproc.so.4.5.3
libbasic-roi.so: /opt/intel/openvino_2021/opencv/lib/libopencv_core.so.4.5.3
libbasic-roi.so: CMakeFiles/basic-roi.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/nuc/Desktop/sentry_2022/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libbasic-roi.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/basic-roi.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/basic-roi.dir/build: libbasic-roi.so

.PHONY : CMakeFiles/basic-roi.dir/build

CMakeFiles/basic-roi.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/basic-roi.dir/cmake_clean.cmake
.PHONY : CMakeFiles/basic-roi.dir/clean

CMakeFiles/basic-roi.dir/depend:
	cd /home/nuc/Desktop/sentry_2022/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/nuc/Desktop/sentry_2022 /home/nuc/Desktop/sentry_2022 /home/nuc/Desktop/sentry_2022/build /home/nuc/Desktop/sentry_2022/build /home/nuc/Desktop/sentry_2022/build/CMakeFiles/basic-roi.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/basic-roi.dir/depend

