# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 4.0

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
CMAKE_COMMAND = /snap/cmake/1463/bin/cmake

# The command to remove a file.
RM = /snap/cmake/1463/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jhahn/00.Project/02.Quarot/quarot

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jhahn/00.Project/02.Quarot/quarot

# Include any dependencies generated for this target.
include third-party/cutlass/examples/42_ampere_tensorop_group_conv/CMakeFiles/42_ampere_tensorop_group_conv.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include third-party/cutlass/examples/42_ampere_tensorop_group_conv/CMakeFiles/42_ampere_tensorop_group_conv.dir/compiler_depend.make

# Include the progress variables for this target.
include third-party/cutlass/examples/42_ampere_tensorop_group_conv/CMakeFiles/42_ampere_tensorop_group_conv.dir/progress.make

# Include the compile flags for this target's objects.
include third-party/cutlass/examples/42_ampere_tensorop_group_conv/CMakeFiles/42_ampere_tensorop_group_conv.dir/flags.make

third-party/cutlass/examples/42_ampere_tensorop_group_conv/CMakeFiles/42_ampere_tensorop_group_conv.dir/codegen:
.PHONY : third-party/cutlass/examples/42_ampere_tensorop_group_conv/CMakeFiles/42_ampere_tensorop_group_conv.dir/codegen

third-party/cutlass/examples/42_ampere_tensorop_group_conv/CMakeFiles/42_ampere_tensorop_group_conv.dir/ampere_tensorop_group_conv.cu.o: third-party/cutlass/examples/42_ampere_tensorop_group_conv/CMakeFiles/42_ampere_tensorop_group_conv.dir/flags.make
third-party/cutlass/examples/42_ampere_tensorop_group_conv/CMakeFiles/42_ampere_tensorop_group_conv.dir/ampere_tensorop_group_conv.cu.o: third-party/cutlass/examples/42_ampere_tensorop_group_conv/CMakeFiles/42_ampere_tensorop_group_conv.dir/includes_CUDA.rsp
third-party/cutlass/examples/42_ampere_tensorop_group_conv/CMakeFiles/42_ampere_tensorop_group_conv.dir/ampere_tensorop_group_conv.cu.o: third-party/cutlass/examples/42_ampere_tensorop_group_conv/ampere_tensorop_group_conv.cu
third-party/cutlass/examples/42_ampere_tensorop_group_conv/CMakeFiles/42_ampere_tensorop_group_conv.dir/ampere_tensorop_group_conv.cu.o: third-party/cutlass/examples/42_ampere_tensorop_group_conv/CMakeFiles/42_ampere_tensorop_group_conv.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/jhahn/00.Project/02.Quarot/quarot/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object third-party/cutlass/examples/42_ampere_tensorop_group_conv/CMakeFiles/42_ampere_tensorop_group_conv.dir/ampere_tensorop_group_conv.cu.o"
	cd /home/jhahn/00.Project/02.Quarot/quarot/third-party/cutlass/examples/42_ampere_tensorop_group_conv && /usr/local/cuda-12.4/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT third-party/cutlass/examples/42_ampere_tensorop_group_conv/CMakeFiles/42_ampere_tensorop_group_conv.dir/ampere_tensorop_group_conv.cu.o -MF CMakeFiles/42_ampere_tensorop_group_conv.dir/ampere_tensorop_group_conv.cu.o.d -x cu -c /home/jhahn/00.Project/02.Quarot/quarot/third-party/cutlass/examples/42_ampere_tensorop_group_conv/ampere_tensorop_group_conv.cu -o CMakeFiles/42_ampere_tensorop_group_conv.dir/ampere_tensorop_group_conv.cu.o

third-party/cutlass/examples/42_ampere_tensorop_group_conv/CMakeFiles/42_ampere_tensorop_group_conv.dir/ampere_tensorop_group_conv.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/42_ampere_tensorop_group_conv.dir/ampere_tensorop_group_conv.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

third-party/cutlass/examples/42_ampere_tensorop_group_conv/CMakeFiles/42_ampere_tensorop_group_conv.dir/ampere_tensorop_group_conv.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/42_ampere_tensorop_group_conv.dir/ampere_tensorop_group_conv.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target 42_ampere_tensorop_group_conv
42_ampere_tensorop_group_conv_OBJECTS = \
"CMakeFiles/42_ampere_tensorop_group_conv.dir/ampere_tensorop_group_conv.cu.o"

# External object files for target 42_ampere_tensorop_group_conv
42_ampere_tensorop_group_conv_EXTERNAL_OBJECTS =

third-party/cutlass/examples/42_ampere_tensorop_group_conv/42_ampere_tensorop_group_conv: third-party/cutlass/examples/42_ampere_tensorop_group_conv/CMakeFiles/42_ampere_tensorop_group_conv.dir/ampere_tensorop_group_conv.cu.o
third-party/cutlass/examples/42_ampere_tensorop_group_conv/42_ampere_tensorop_group_conv: third-party/cutlass/examples/42_ampere_tensorop_group_conv/CMakeFiles/42_ampere_tensorop_group_conv.dir/build.make
third-party/cutlass/examples/42_ampere_tensorop_group_conv/42_ampere_tensorop_group_conv: third-party/cutlass/examples/42_ampere_tensorop_group_conv/CMakeFiles/42_ampere_tensorop_group_conv.dir/linkLibs.rsp
third-party/cutlass/examples/42_ampere_tensorop_group_conv/42_ampere_tensorop_group_conv: third-party/cutlass/examples/42_ampere_tensorop_group_conv/CMakeFiles/42_ampere_tensorop_group_conv.dir/objects1.rsp
third-party/cutlass/examples/42_ampere_tensorop_group_conv/42_ampere_tensorop_group_conv: third-party/cutlass/examples/42_ampere_tensorop_group_conv/CMakeFiles/42_ampere_tensorop_group_conv.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/jhahn/00.Project/02.Quarot/quarot/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable 42_ampere_tensorop_group_conv"
	cd /home/jhahn/00.Project/02.Quarot/quarot/third-party/cutlass/examples/42_ampere_tensorop_group_conv && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/42_ampere_tensorop_group_conv.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
third-party/cutlass/examples/42_ampere_tensorop_group_conv/CMakeFiles/42_ampere_tensorop_group_conv.dir/build: third-party/cutlass/examples/42_ampere_tensorop_group_conv/42_ampere_tensorop_group_conv
.PHONY : third-party/cutlass/examples/42_ampere_tensorop_group_conv/CMakeFiles/42_ampere_tensorop_group_conv.dir/build

third-party/cutlass/examples/42_ampere_tensorop_group_conv/CMakeFiles/42_ampere_tensorop_group_conv.dir/clean:
	cd /home/jhahn/00.Project/02.Quarot/quarot/third-party/cutlass/examples/42_ampere_tensorop_group_conv && $(CMAKE_COMMAND) -P CMakeFiles/42_ampere_tensorop_group_conv.dir/cmake_clean.cmake
.PHONY : third-party/cutlass/examples/42_ampere_tensorop_group_conv/CMakeFiles/42_ampere_tensorop_group_conv.dir/clean

third-party/cutlass/examples/42_ampere_tensorop_group_conv/CMakeFiles/42_ampere_tensorop_group_conv.dir/depend:
	cd /home/jhahn/00.Project/02.Quarot/quarot && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jhahn/00.Project/02.Quarot/quarot /home/jhahn/00.Project/02.Quarot/quarot/third-party/cutlass/examples/42_ampere_tensorop_group_conv /home/jhahn/00.Project/02.Quarot/quarot /home/jhahn/00.Project/02.Quarot/quarot/third-party/cutlass/examples/42_ampere_tensorop_group_conv /home/jhahn/00.Project/02.Quarot/quarot/third-party/cutlass/examples/42_ampere_tensorop_group_conv/CMakeFiles/42_ampere_tensorop_group_conv.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : third-party/cutlass/examples/42_ampere_tensorop_group_conv/CMakeFiles/42_ampere_tensorop_group_conv.dir/depend

