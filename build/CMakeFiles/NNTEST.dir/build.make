# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

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

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/kevin/Development/NN

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/kevin/Development/NN/build

# Include any dependencies generated for this target.
include CMakeFiles/NNTEST.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/NNTEST.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/NNTEST.dir/flags.make

CMakeFiles/NNTEST.dir/testNN.cpp.o: CMakeFiles/NNTEST.dir/flags.make
CMakeFiles/NNTEST.dir/testNN.cpp.o: ../testNN.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/kevin/Development/NN/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/NNTEST.dir/testNN.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/NNTEST.dir/testNN.cpp.o -c /home/kevin/Development/NN/testNN.cpp

CMakeFiles/NNTEST.dir/testNN.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/NNTEST.dir/testNN.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/kevin/Development/NN/testNN.cpp > CMakeFiles/NNTEST.dir/testNN.cpp.i

CMakeFiles/NNTEST.dir/testNN.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/NNTEST.dir/testNN.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/kevin/Development/NN/testNN.cpp -o CMakeFiles/NNTEST.dir/testNN.cpp.s

CMakeFiles/NNTEST.dir/testNN.cpp.o.requires:
.PHONY : CMakeFiles/NNTEST.dir/testNN.cpp.o.requires

CMakeFiles/NNTEST.dir/testNN.cpp.o.provides: CMakeFiles/NNTEST.dir/testNN.cpp.o.requires
	$(MAKE) -f CMakeFiles/NNTEST.dir/build.make CMakeFiles/NNTEST.dir/testNN.cpp.o.provides.build
.PHONY : CMakeFiles/NNTEST.dir/testNN.cpp.o.provides

CMakeFiles/NNTEST.dir/testNN.cpp.o.provides.build: CMakeFiles/NNTEST.dir/testNN.cpp.o

# Object files for target NNTEST
NNTEST_OBJECTS = \
"CMakeFiles/NNTEST.dir/testNN.cpp.o"

# External object files for target NNTEST
NNTEST_EXTERNAL_OBJECTS =

../bin/NNTEST: CMakeFiles/NNTEST.dir/testNN.cpp.o
../bin/NNTEST: CMakeFiles/NNTEST.dir/build.make
../bin/NNTEST: CMakeFiles/NNTEST.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable ../bin/NNTEST"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/NNTEST.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/NNTEST.dir/build: ../bin/NNTEST
.PHONY : CMakeFiles/NNTEST.dir/build

CMakeFiles/NNTEST.dir/requires: CMakeFiles/NNTEST.dir/testNN.cpp.o.requires
.PHONY : CMakeFiles/NNTEST.dir/requires

CMakeFiles/NNTEST.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/NNTEST.dir/cmake_clean.cmake
.PHONY : CMakeFiles/NNTEST.dir/clean

CMakeFiles/NNTEST.dir/depend:
	cd /home/kevin/Development/NN/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/kevin/Development/NN /home/kevin/Development/NN /home/kevin/Development/NN/build /home/kevin/Development/NN/build /home/kevin/Development/NN/build/CMakeFiles/NNTEST.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/NNTEST.dir/depend

