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
include third-party/cutlass/tools/library/CMakeFiles/cutlass_library_gemm_sm80_u4_i16864gemm_u4.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include third-party/cutlass/tools/library/CMakeFiles/cutlass_library_gemm_sm80_u4_i16864gemm_u4.dir/compiler_depend.make

# Include the progress variables for this target.
include third-party/cutlass/tools/library/CMakeFiles/cutlass_library_gemm_sm80_u4_i16864gemm_u4.dir/progress.make

# Include the compile flags for this target's objects.
include third-party/cutlass/tools/library/CMakeFiles/cutlass_library_gemm_sm80_u4_i16864gemm_u4.dir/flags.make

third-party/cutlass/tools/library/CMakeFiles/cutlass_library_gemm_sm80_u4_i16864gemm_u4.dir/codegen:
.PHONY : third-party/cutlass/tools/library/CMakeFiles/cutlass_library_gemm_sm80_u4_i16864gemm_u4.dir/codegen

# Object files for target cutlass_library_gemm_sm80_u4_i16864gemm_u4
cutlass_library_gemm_sm80_u4_i16864gemm_u4_OBJECTS =

# External object files for target cutlass_library_gemm_sm80_u4_i16864gemm_u4
cutlass_library_gemm_sm80_u4_i16864gemm_u4_EXTERNAL_OBJECTS = \
"/home/jhahn/00.Project/02.Quarot/quarot/third-party/cutlass/tools/library/CMakeFiles/cutlass_library_gemm_sm80_u4_i16864gemm_u4_objs.dir/generated/gemm/80/u4_i16864gemm_u4/all_sm80_u4_i16864gemm_u4_gemm_operations.cu.o" \
"/home/jhahn/00.Project/02.Quarot/quarot/third-party/cutlass/tools/library/CMakeFiles/cutlass_library_gemm_sm80_u4_i16864gemm_u4_objs.dir/generated/gemm/80/u4_i16864gemm_u4/cutlass_tensorop_u4_i16864gemm_u4_256x128_128x3_tn_align32.cu.o" \
"/home/jhahn/00.Project/02.Quarot/quarot/third-party/cutlass/tools/library/CMakeFiles/cutlass_library_gemm_sm80_u4_i16864gemm_u4_objs.dir/generated/gemm/80/u4_i16864gemm_u4/cutlass_tensorop_u4_i16864gemm_u4_256x128_128x3_n64t64_align32.cu.o"

third-party/cutlass/tools/library/libcutlass_gemm_sm80_u4_i16864gemm_u4.so: third-party/cutlass/tools/library/CMakeFiles/cutlass_library_gemm_sm80_u4_i16864gemm_u4_objs.dir/generated/gemm/80/u4_i16864gemm_u4/all_sm80_u4_i16864gemm_u4_gemm_operations.cu.o
third-party/cutlass/tools/library/libcutlass_gemm_sm80_u4_i16864gemm_u4.so: third-party/cutlass/tools/library/CMakeFiles/cutlass_library_gemm_sm80_u4_i16864gemm_u4_objs.dir/generated/gemm/80/u4_i16864gemm_u4/cutlass_tensorop_u4_i16864gemm_u4_256x128_128x3_tn_align32.cu.o
third-party/cutlass/tools/library/libcutlass_gemm_sm80_u4_i16864gemm_u4.so: third-party/cutlass/tools/library/CMakeFiles/cutlass_library_gemm_sm80_u4_i16864gemm_u4_objs.dir/generated/gemm/80/u4_i16864gemm_u4/cutlass_tensorop_u4_i16864gemm_u4_256x128_128x3_n64t64_align32.cu.o
third-party/cutlass/tools/library/libcutlass_gemm_sm80_u4_i16864gemm_u4.so: third-party/cutlass/tools/library/CMakeFiles/cutlass_library_gemm_sm80_u4_i16864gemm_u4.dir/build.make
third-party/cutlass/tools/library/libcutlass_gemm_sm80_u4_i16864gemm_u4.so: /usr/local/cuda-12.4/lib64/stubs/libcuda.so
third-party/cutlass/tools/library/libcutlass_gemm_sm80_u4_i16864gemm_u4.so: third-party/cutlass/tools/library/CMakeFiles/cutlass_library_gemm_sm80_u4_i16864gemm_u4.dir/linkLibs.rsp
third-party/cutlass/tools/library/libcutlass_gemm_sm80_u4_i16864gemm_u4.so: third-party/cutlass/tools/library/CMakeFiles/cutlass_library_gemm_sm80_u4_i16864gemm_u4.dir/objects1.rsp
third-party/cutlass/tools/library/libcutlass_gemm_sm80_u4_i16864gemm_u4.so: third-party/cutlass/tools/library/CMakeFiles/cutlass_library_gemm_sm80_u4_i16864gemm_u4.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/jhahn/00.Project/02.Quarot/quarot/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Linking CUDA shared library libcutlass_gemm_sm80_u4_i16864gemm_u4.so"
	cd /home/jhahn/00.Project/02.Quarot/quarot/third-party/cutlass/tools/library && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cutlass_library_gemm_sm80_u4_i16864gemm_u4.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
third-party/cutlass/tools/library/CMakeFiles/cutlass_library_gemm_sm80_u4_i16864gemm_u4.dir/build: third-party/cutlass/tools/library/libcutlass_gemm_sm80_u4_i16864gemm_u4.so
.PHONY : third-party/cutlass/tools/library/CMakeFiles/cutlass_library_gemm_sm80_u4_i16864gemm_u4.dir/build

third-party/cutlass/tools/library/CMakeFiles/cutlass_library_gemm_sm80_u4_i16864gemm_u4.dir/clean:
	cd /home/jhahn/00.Project/02.Quarot/quarot/third-party/cutlass/tools/library && $(CMAKE_COMMAND) -P CMakeFiles/cutlass_library_gemm_sm80_u4_i16864gemm_u4.dir/cmake_clean.cmake
.PHONY : third-party/cutlass/tools/library/CMakeFiles/cutlass_library_gemm_sm80_u4_i16864gemm_u4.dir/clean

third-party/cutlass/tools/library/CMakeFiles/cutlass_library_gemm_sm80_u4_i16864gemm_u4.dir/depend:
	cd /home/jhahn/00.Project/02.Quarot/quarot && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jhahn/00.Project/02.Quarot/quarot /home/jhahn/00.Project/02.Quarot/quarot/third-party/cutlass/tools/library /home/jhahn/00.Project/02.Quarot/quarot /home/jhahn/00.Project/02.Quarot/quarot/third-party/cutlass/tools/library /home/jhahn/00.Project/02.Quarot/quarot/third-party/cutlass/tools/library/CMakeFiles/cutlass_library_gemm_sm80_u4_i16864gemm_u4.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : third-party/cutlass/tools/library/CMakeFiles/cutlass_library_gemm_sm80_u4_i16864gemm_u4.dir/depend

