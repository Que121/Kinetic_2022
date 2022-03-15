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
include CMakeFiles/video-record.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/video-record.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/video-record.dir/flags.make

CMakeFiles/video-record.dir/module/record/record.cpp.o: CMakeFiles/video-record.dir/flags.make
CMakeFiles/video-record.dir/module/record/record.cpp.o: ../module/record/record.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nuc/Desktop/sentry_2022/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/video-record.dir/module/record/record.cpp.o"
	/bin/x86_64-linux-gnu-g++-9  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/video-record.dir/module/record/record.cpp.o -c /home/nuc/Desktop/sentry_2022/module/record/record.cpp

CMakeFiles/video-record.dir/module/record/record.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/video-record.dir/module/record/record.cpp.i"
	/bin/x86_64-linux-gnu-g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nuc/Desktop/sentry_2022/module/record/record.cpp > CMakeFiles/video-record.dir/module/record/record.cpp.i

CMakeFiles/video-record.dir/module/record/record.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/video-record.dir/module/record/record.cpp.s"
	/bin/x86_64-linux-gnu-g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nuc/Desktop/sentry_2022/module/record/record.cpp -o CMakeFiles/video-record.dir/module/record/record.cpp.s

# Object files for target video-record
video__record_OBJECTS = \
"CMakeFiles/video-record.dir/module/record/record.cpp.o"

# External object files for target video-record
video__record_EXTERNAL_OBJECTS =

libvideo-record.so: CMakeFiles/video-record.dir/module/record/record.cpp.o
libvideo-record.so: CMakeFiles/video-record.dir/build.make
libvideo-record.so: /opt/intel/openvino_2021/opencv/lib/libopencv_gapi.so.4.5.3
libvideo-record.so: /opt/intel/openvino_2021/opencv/lib/libopencv_highgui.so.4.5.3
libvideo-record.so: /opt/intel/openvino_2021/opencv/lib/libopencv_ml.so.4.5.3
libvideo-record.so: /opt/intel/openvino_2021/opencv/lib/libopencv_objdetect.so.4.5.3
libvideo-record.so: /opt/intel/openvino_2021/opencv/lib/libopencv_photo.so.4.5.3
libvideo-record.so: /opt/intel/openvino_2021/opencv/lib/libopencv_stitching.so.4.5.3
libvideo-record.so: /opt/intel/openvino_2021/opencv/lib/libopencv_video.so.4.5.3
libvideo-record.so: /opt/intel/openvino_2021/opencv/lib/libopencv_videoio.so.4.5.3
libvideo-record.so: /opt/intel/openvino_2021/opencv/lib/libopencv_dnn.so.4.5.3
libvideo-record.so: /opt/intel/openvino_2021/opencv/lib/libopencv_imgcodecs.so.4.5.3
libvideo-record.so: /opt/intel/openvino_2021/opencv/lib/libopencv_calib3d.so.4.5.3
libvideo-record.so: /opt/intel/openvino_2021/opencv/lib/libopencv_features2d.so.4.5.3
libvideo-record.so: /opt/intel/openvino_2021/opencv/lib/libopencv_flann.so.4.5.3
libvideo-record.so: /opt/intel/openvino_2021/opencv/lib/libopencv_imgproc.so.4.5.3
libvideo-record.so: /opt/intel/openvino_2021/opencv/lib/libopencv_core.so.4.5.3
libvideo-record.so: CMakeFiles/video-record.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/nuc/Desktop/sentry_2022/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libvideo-record.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/video-record.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/video-record.dir/build: libvideo-record.so

.PHONY : CMakeFiles/video-record.dir/build

CMakeFiles/video-record.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/video-record.dir/cmake_clean.cmake
.PHONY : CMakeFiles/video-record.dir/clean

CMakeFiles/video-record.dir/depend:
	cd /home/nuc/Desktop/sentry_2022/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/nuc/Desktop/sentry_2022 /home/nuc/Desktop/sentry_2022 /home/nuc/Desktop/sentry_2022/build /home/nuc/Desktop/sentry_2022/build /home/nuc/Desktop/sentry_2022/build/CMakeFiles/video-record.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/video-record.dir/depend
