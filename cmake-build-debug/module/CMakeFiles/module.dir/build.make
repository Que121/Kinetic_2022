# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.21

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/nuc/clion/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/nuc/clion/bin/cmake/linux/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/nuc/Desktop/Kinetic_2022

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/nuc/Desktop/Kinetic_2022/cmake-build-debug

# Include any dependencies generated for this target.
include module/CMakeFiles/module.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include module/CMakeFiles/module.dir/compiler_depend.make

# Include the progress variables for this target.
include module/CMakeFiles/module.dir/progress.make

# Include the compile flags for this target's objects.
include module/CMakeFiles/module.dir/flags.make

module/CMakeFiles/module.dir/openvinoNanodet/openvinoNanodet_armorDetection.cpp.o: module/CMakeFiles/module.dir/flags.make
module/CMakeFiles/module.dir/openvinoNanodet/openvinoNanodet_armorDetection.cpp.o: ../module/openvinoNanodet/openvinoNanodet_armorDetection.cpp
module/CMakeFiles/module.dir/openvinoNanodet/openvinoNanodet_armorDetection.cpp.o: module/CMakeFiles/module.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nuc/Desktop/Kinetic_2022/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object module/CMakeFiles/module.dir/openvinoNanodet/openvinoNanodet_armorDetection.cpp.o"
	cd /home/nuc/Desktop/Kinetic_2022/cmake-build-debug/module && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT module/CMakeFiles/module.dir/openvinoNanodet/openvinoNanodet_armorDetection.cpp.o -MF CMakeFiles/module.dir/openvinoNanodet/openvinoNanodet_armorDetection.cpp.o.d -o CMakeFiles/module.dir/openvinoNanodet/openvinoNanodet_armorDetection.cpp.o -c /home/nuc/Desktop/Kinetic_2022/module/openvinoNanodet/openvinoNanodet_armorDetection.cpp

module/CMakeFiles/module.dir/openvinoNanodet/openvinoNanodet_armorDetection.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/module.dir/openvinoNanodet/openvinoNanodet_armorDetection.cpp.i"
	cd /home/nuc/Desktop/Kinetic_2022/cmake-build-debug/module && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nuc/Desktop/Kinetic_2022/module/openvinoNanodet/openvinoNanodet_armorDetection.cpp > CMakeFiles/module.dir/openvinoNanodet/openvinoNanodet_armorDetection.cpp.i

module/CMakeFiles/module.dir/openvinoNanodet/openvinoNanodet_armorDetection.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/module.dir/openvinoNanodet/openvinoNanodet_armorDetection.cpp.s"
	cd /home/nuc/Desktop/Kinetic_2022/cmake-build-debug/module && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nuc/Desktop/Kinetic_2022/module/openvinoNanodet/openvinoNanodet_armorDetection.cpp -o CMakeFiles/module.dir/openvinoNanodet/openvinoNanodet_armorDetection.cpp.s

module/CMakeFiles/module.dir/angle_solve/basic_pnp.cpp.o: module/CMakeFiles/module.dir/flags.make
module/CMakeFiles/module.dir/angle_solve/basic_pnp.cpp.o: ../module/angle_solve/basic_pnp.cpp
module/CMakeFiles/module.dir/angle_solve/basic_pnp.cpp.o: module/CMakeFiles/module.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nuc/Desktop/Kinetic_2022/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object module/CMakeFiles/module.dir/angle_solve/basic_pnp.cpp.o"
	cd /home/nuc/Desktop/Kinetic_2022/cmake-build-debug/module && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT module/CMakeFiles/module.dir/angle_solve/basic_pnp.cpp.o -MF CMakeFiles/module.dir/angle_solve/basic_pnp.cpp.o.d -o CMakeFiles/module.dir/angle_solve/basic_pnp.cpp.o -c /home/nuc/Desktop/Kinetic_2022/module/angle_solve/basic_pnp.cpp

module/CMakeFiles/module.dir/angle_solve/basic_pnp.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/module.dir/angle_solve/basic_pnp.cpp.i"
	cd /home/nuc/Desktop/Kinetic_2022/cmake-build-debug/module && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nuc/Desktop/Kinetic_2022/module/angle_solve/basic_pnp.cpp > CMakeFiles/module.dir/angle_solve/basic_pnp.cpp.i

module/CMakeFiles/module.dir/angle_solve/basic_pnp.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/module.dir/angle_solve/basic_pnp.cpp.s"
	cd /home/nuc/Desktop/Kinetic_2022/cmake-build-debug/module && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nuc/Desktop/Kinetic_2022/module/angle_solve/basic_pnp.cpp -o CMakeFiles/module.dir/angle_solve/basic_pnp.cpp.s

module/CMakeFiles/module.dir/armor/basic_armor.cpp.o: module/CMakeFiles/module.dir/flags.make
module/CMakeFiles/module.dir/armor/basic_armor.cpp.o: ../module/armor/basic_armor.cpp
module/CMakeFiles/module.dir/armor/basic_armor.cpp.o: module/CMakeFiles/module.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nuc/Desktop/Kinetic_2022/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object module/CMakeFiles/module.dir/armor/basic_armor.cpp.o"
	cd /home/nuc/Desktop/Kinetic_2022/cmake-build-debug/module && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT module/CMakeFiles/module.dir/armor/basic_armor.cpp.o -MF CMakeFiles/module.dir/armor/basic_armor.cpp.o.d -o CMakeFiles/module.dir/armor/basic_armor.cpp.o -c /home/nuc/Desktop/Kinetic_2022/module/armor/basic_armor.cpp

module/CMakeFiles/module.dir/armor/basic_armor.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/module.dir/armor/basic_armor.cpp.i"
	cd /home/nuc/Desktop/Kinetic_2022/cmake-build-debug/module && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nuc/Desktop/Kinetic_2022/module/armor/basic_armor.cpp > CMakeFiles/module.dir/armor/basic_armor.cpp.i

module/CMakeFiles/module.dir/armor/basic_armor.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/module.dir/armor/basic_armor.cpp.s"
	cd /home/nuc/Desktop/Kinetic_2022/cmake-build-debug/module && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nuc/Desktop/Kinetic_2022/module/armor/basic_armor.cpp -o CMakeFiles/module.dir/armor/basic_armor.cpp.s

module/CMakeFiles/module.dir/filter/basic_kalman.cpp.o: module/CMakeFiles/module.dir/flags.make
module/CMakeFiles/module.dir/filter/basic_kalman.cpp.o: ../module/filter/basic_kalman.cpp
module/CMakeFiles/module.dir/filter/basic_kalman.cpp.o: module/CMakeFiles/module.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nuc/Desktop/Kinetic_2022/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object module/CMakeFiles/module.dir/filter/basic_kalman.cpp.o"
	cd /home/nuc/Desktop/Kinetic_2022/cmake-build-debug/module && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT module/CMakeFiles/module.dir/filter/basic_kalman.cpp.o -MF CMakeFiles/module.dir/filter/basic_kalman.cpp.o.d -o CMakeFiles/module.dir/filter/basic_kalman.cpp.o -c /home/nuc/Desktop/Kinetic_2022/module/filter/basic_kalman.cpp

module/CMakeFiles/module.dir/filter/basic_kalman.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/module.dir/filter/basic_kalman.cpp.i"
	cd /home/nuc/Desktop/Kinetic_2022/cmake-build-debug/module && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nuc/Desktop/Kinetic_2022/module/filter/basic_kalman.cpp > CMakeFiles/module.dir/filter/basic_kalman.cpp.i

module/CMakeFiles/module.dir/filter/basic_kalman.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/module.dir/filter/basic_kalman.cpp.s"
	cd /home/nuc/Desktop/Kinetic_2022/cmake-build-debug/module && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nuc/Desktop/Kinetic_2022/module/filter/basic_kalman.cpp -o CMakeFiles/module.dir/filter/basic_kalman.cpp.s

module/CMakeFiles/module.dir/record/record.cpp.o: module/CMakeFiles/module.dir/flags.make
module/CMakeFiles/module.dir/record/record.cpp.o: ../module/record/record.cpp
module/CMakeFiles/module.dir/record/record.cpp.o: module/CMakeFiles/module.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nuc/Desktop/Kinetic_2022/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object module/CMakeFiles/module.dir/record/record.cpp.o"
	cd /home/nuc/Desktop/Kinetic_2022/cmake-build-debug/module && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT module/CMakeFiles/module.dir/record/record.cpp.o -MF CMakeFiles/module.dir/record/record.cpp.o.d -o CMakeFiles/module.dir/record/record.cpp.o -c /home/nuc/Desktop/Kinetic_2022/module/record/record.cpp

module/CMakeFiles/module.dir/record/record.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/module.dir/record/record.cpp.i"
	cd /home/nuc/Desktop/Kinetic_2022/cmake-build-debug/module && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nuc/Desktop/Kinetic_2022/module/record/record.cpp > CMakeFiles/module.dir/record/record.cpp.i

module/CMakeFiles/module.dir/record/record.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/module.dir/record/record.cpp.s"
	cd /home/nuc/Desktop/Kinetic_2022/cmake-build-debug/module && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nuc/Desktop/Kinetic_2022/module/record/record.cpp -o CMakeFiles/module.dir/record/record.cpp.s

module/CMakeFiles/module.dir/roi/basic_roi.cpp.o: module/CMakeFiles/module.dir/flags.make
module/CMakeFiles/module.dir/roi/basic_roi.cpp.o: ../module/roi/basic_roi.cpp
module/CMakeFiles/module.dir/roi/basic_roi.cpp.o: module/CMakeFiles/module.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nuc/Desktop/Kinetic_2022/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object module/CMakeFiles/module.dir/roi/basic_roi.cpp.o"
	cd /home/nuc/Desktop/Kinetic_2022/cmake-build-debug/module && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT module/CMakeFiles/module.dir/roi/basic_roi.cpp.o -MF CMakeFiles/module.dir/roi/basic_roi.cpp.o.d -o CMakeFiles/module.dir/roi/basic_roi.cpp.o -c /home/nuc/Desktop/Kinetic_2022/module/roi/basic_roi.cpp

module/CMakeFiles/module.dir/roi/basic_roi.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/module.dir/roi/basic_roi.cpp.i"
	cd /home/nuc/Desktop/Kinetic_2022/cmake-build-debug/module && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nuc/Desktop/Kinetic_2022/module/roi/basic_roi.cpp > CMakeFiles/module.dir/roi/basic_roi.cpp.i

module/CMakeFiles/module.dir/roi/basic_roi.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/module.dir/roi/basic_roi.cpp.s"
	cd /home/nuc/Desktop/Kinetic_2022/cmake-build-debug/module && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nuc/Desktop/Kinetic_2022/module/roi/basic_roi.cpp -o CMakeFiles/module.dir/roi/basic_roi.cpp.s

# Object files for target module
module_OBJECTS = \
"CMakeFiles/module.dir/openvinoNanodet/openvinoNanodet_armorDetection.cpp.o" \
"CMakeFiles/module.dir/angle_solve/basic_pnp.cpp.o" \
"CMakeFiles/module.dir/armor/basic_armor.cpp.o" \
"CMakeFiles/module.dir/filter/basic_kalman.cpp.o" \
"CMakeFiles/module.dir/record/record.cpp.o" \
"CMakeFiles/module.dir/roi/basic_roi.cpp.o"

# External object files for target module
module_EXTERNAL_OBJECTS =

module/libmodule.so: module/CMakeFiles/module.dir/openvinoNanodet/openvinoNanodet_armorDetection.cpp.o
module/libmodule.so: module/CMakeFiles/module.dir/angle_solve/basic_pnp.cpp.o
module/libmodule.so: module/CMakeFiles/module.dir/armor/basic_armor.cpp.o
module/libmodule.so: module/CMakeFiles/module.dir/filter/basic_kalman.cpp.o
module/libmodule.so: module/CMakeFiles/module.dir/record/record.cpp.o
module/libmodule.so: module/CMakeFiles/module.dir/roi/basic_roi.cpp.o
module/libmodule.so: module/CMakeFiles/module.dir/build.make
module/libmodule.so: /opt/intel/openvino_2021/opencv/lib/libopencv_gapi.so.4.5.3
module/libmodule.so: /opt/intel/openvino_2021/opencv/lib/libopencv_highgui.so.4.5.3
module/libmodule.so: /opt/intel/openvino_2021/opencv/lib/libopencv_ml.so.4.5.3
module/libmodule.so: /opt/intel/openvino_2021/opencv/lib/libopencv_objdetect.so.4.5.3
module/libmodule.so: /opt/intel/openvino_2021/opencv/lib/libopencv_photo.so.4.5.3
module/libmodule.so: /opt/intel/openvino_2021/opencv/lib/libopencv_stitching.so.4.5.3
module/libmodule.so: /opt/intel/openvino_2021/opencv/lib/libopencv_video.so.4.5.3
module/libmodule.so: /opt/intel/openvino_2021/opencv/lib/libopencv_videoio.so.4.5.3
module/libmodule.so: /usr/local/lib/libceres.a
module/libmodule.so: /usr/local/lib/libfmt.a
module/libmodule.so: /opt/intel/openvino_2021/opencv/lib/libopencv_dnn.so.4.5.3
module/libmodule.so: /opt/intel/openvino_2021/deployment_tools/ngraph/lib/libngraph.so
module/libmodule.so: /opt/intel/openvino_2021/deployment_tools/inference_engine/lib/intel64/libinference_engine.so
module/libmodule.so: /opt/intel/openvino_2021/deployment_tools/inference_engine/lib/intel64/libinference_engine_c_api.so
module/libmodule.so: /opt/intel/openvino_2021/opencv/lib/libopencv_imgcodecs.so.4.5.3
module/libmodule.so: /opt/intel/openvino_2021/opencv/lib/libopencv_calib3d.so.4.5.3
module/libmodule.so: /opt/intel/openvino_2021/opencv/lib/libopencv_features2d.so.4.5.3
module/libmodule.so: /opt/intel/openvino_2021/opencv/lib/libopencv_flann.so.4.5.3
module/libmodule.so: /opt/intel/openvino_2021/opencv/lib/libopencv_imgproc.so.4.5.3
module/libmodule.so: /opt/intel/openvino_2021/opencv/lib/libopencv_core.so.4.5.3
module/libmodule.so: /usr/lib/x86_64-linux-gnu/libglog.so
module/libmodule.so: module/CMakeFiles/module.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/nuc/Desktop/Kinetic_2022/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking CXX shared library libmodule.so"
	cd /home/nuc/Desktop/Kinetic_2022/cmake-build-debug/module && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/module.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
module/CMakeFiles/module.dir/build: module/libmodule.so
.PHONY : module/CMakeFiles/module.dir/build

module/CMakeFiles/module.dir/clean:
	cd /home/nuc/Desktop/Kinetic_2022/cmake-build-debug/module && $(CMAKE_COMMAND) -P CMakeFiles/module.dir/cmake_clean.cmake
.PHONY : module/CMakeFiles/module.dir/clean

module/CMakeFiles/module.dir/depend:
	cd /home/nuc/Desktop/Kinetic_2022/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/nuc/Desktop/Kinetic_2022 /home/nuc/Desktop/Kinetic_2022/module /home/nuc/Desktop/Kinetic_2022/cmake-build-debug /home/nuc/Desktop/Kinetic_2022/cmake-build-debug/module /home/nuc/Desktop/Kinetic_2022/cmake-build-debug/module/CMakeFiles/module.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : module/CMakeFiles/module.dir/depend

