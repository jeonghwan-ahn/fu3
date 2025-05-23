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

# Utility rule file for test_profiler_trmm.

# Include any custom commands dependencies for this target.
include third-party/cutlass/tools/profiler/CMakeFiles/test_profiler_trmm.dir/compiler_depend.make

# Include the progress variables for this target.
include third-party/cutlass/tools/profiler/CMakeFiles/test_profiler_trmm.dir/progress.make

third-party/cutlass/tools/profiler/CMakeFiles/test_profiler_trmm: third-party/cutlass/tools/profiler/cutlass_profiler
	cd /home/jhahn/00.Project/02.Quarot/quarot/third-party/cutlass/tools/profiler && ./cutlass_profiler --operation=Trmm --providers=cutlass --verification-providers=device,host --junit-output=test_cutlass_profiler_trmm --print-kernel-before-running=true

third-party/cutlass/tools/profiler/CMakeFiles/test_profiler_trmm.dir/codegen:
.PHONY : third-party/cutlass/tools/profiler/CMakeFiles/test_profiler_trmm.dir/codegen

test_profiler_trmm: third-party/cutlass/tools/profiler/CMakeFiles/test_profiler_trmm
test_profiler_trmm: third-party/cutlass/tools/profiler/CMakeFiles/test_profiler_trmm.dir/build.make
.PHONY : test_profiler_trmm

# Rule to build all files generated by this target.
third-party/cutlass/tools/profiler/CMakeFiles/test_profiler_trmm.dir/build: test_profiler_trmm
.PHONY : third-party/cutlass/tools/profiler/CMakeFiles/test_profiler_trmm.dir/build

third-party/cutlass/tools/profiler/CMakeFiles/test_profiler_trmm.dir/clean:
	cd /home/jhahn/00.Project/02.Quarot/quarot/third-party/cutlass/tools/profiler && $(CMAKE_COMMAND) -P CMakeFiles/test_profiler_trmm.dir/cmake_clean.cmake
.PHONY : third-party/cutlass/tools/profiler/CMakeFiles/test_profiler_trmm.dir/clean

third-party/cutlass/tools/profiler/CMakeFiles/test_profiler_trmm.dir/depend:
	cd /home/jhahn/00.Project/02.Quarot/quarot && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jhahn/00.Project/02.Quarot/quarot /home/jhahn/00.Project/02.Quarot/quarot/third-party/cutlass/tools/profiler /home/jhahn/00.Project/02.Quarot/quarot /home/jhahn/00.Project/02.Quarot/quarot/third-party/cutlass/tools/profiler /home/jhahn/00.Project/02.Quarot/quarot/third-party/cutlass/tools/profiler/CMakeFiles/test_profiler_trmm.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : third-party/cutlass/tools/profiler/CMakeFiles/test_profiler_trmm.dir/depend

