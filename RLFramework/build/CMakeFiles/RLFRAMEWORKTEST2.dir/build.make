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
CMAKE_SOURCE_DIR = /home/kevin/Development/NN/RLFramework

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/kevin/Development/NN/RLFramework/build

# Include any dependencies generated for this target.
include CMakeFiles/RLFRAMEWORKTEST2.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/RLFRAMEWORKTEST2.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/RLFRAMEWORKTEST2.dir/flags.make

CMakeFiles/RLFRAMEWORKTEST2.dir/testRLFramework.cpp.o: CMakeFiles/RLFRAMEWORKTEST2.dir/flags.make
CMakeFiles/RLFRAMEWORKTEST2.dir/testRLFramework.cpp.o: ../testRLFramework.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/kevin/Development/NN/RLFramework/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/RLFRAMEWORKTEST2.dir/testRLFramework.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/RLFRAMEWORKTEST2.dir/testRLFramework.cpp.o -c /home/kevin/Development/NN/RLFramework/testRLFramework.cpp

CMakeFiles/RLFRAMEWORKTEST2.dir/testRLFramework.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/RLFRAMEWORKTEST2.dir/testRLFramework.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/kevin/Development/NN/RLFramework/testRLFramework.cpp > CMakeFiles/RLFRAMEWORKTEST2.dir/testRLFramework.cpp.i

CMakeFiles/RLFRAMEWORKTEST2.dir/testRLFramework.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/RLFRAMEWORKTEST2.dir/testRLFramework.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/kevin/Development/NN/RLFramework/testRLFramework.cpp -o CMakeFiles/RLFRAMEWORKTEST2.dir/testRLFramework.cpp.s

CMakeFiles/RLFRAMEWORKTEST2.dir/testRLFramework.cpp.o.requires:
.PHONY : CMakeFiles/RLFRAMEWORKTEST2.dir/testRLFramework.cpp.o.requires

CMakeFiles/RLFRAMEWORKTEST2.dir/testRLFramework.cpp.o.provides: CMakeFiles/RLFRAMEWORKTEST2.dir/testRLFramework.cpp.o.requires
	$(MAKE) -f CMakeFiles/RLFRAMEWORKTEST2.dir/build.make CMakeFiles/RLFRAMEWORKTEST2.dir/testRLFramework.cpp.o.provides.build
.PHONY : CMakeFiles/RLFRAMEWORKTEST2.dir/testRLFramework.cpp.o.provides

CMakeFiles/RLFRAMEWORKTEST2.dir/testRLFramework.cpp.o.provides.build: CMakeFiles/RLFRAMEWORKTEST2.dir/testRLFramework.cpp.o

# Object files for target RLFRAMEWORKTEST2
RLFRAMEWORKTEST2_OBJECTS = \
"CMakeFiles/RLFRAMEWORKTEST2.dir/testRLFramework.cpp.o"

# External object files for target RLFRAMEWORKTEST2
RLFRAMEWORKTEST2_EXTERNAL_OBJECTS =

../bin/RLFRAMEWORKTEST2: CMakeFiles/RLFRAMEWORKTEST2.dir/testRLFramework.cpp.o
../bin/RLFRAMEWORKTEST2: CMakeFiles/RLFRAMEWORKTEST2.dir/build.make
../bin/RLFRAMEWORKTEST2: CMakeFiles/RLFRAMEWORKTEST2.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable ../bin/RLFRAMEWORKTEST2"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/RLFRAMEWORKTEST2.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/RLFRAMEWORKTEST2.dir/build: ../bin/RLFRAMEWORKTEST2
.PHONY : CMakeFiles/RLFRAMEWORKTEST2.dir/build

CMakeFiles/RLFRAMEWORKTEST2.dir/requires: CMakeFiles/RLFRAMEWORKTEST2.dir/testRLFramework.cpp.o.requires
.PHONY : CMakeFiles/RLFRAMEWORKTEST2.dir/requires

CMakeFiles/RLFRAMEWORKTEST2.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/RLFRAMEWORKTEST2.dir/cmake_clean.cmake
.PHONY : CMakeFiles/RLFRAMEWORKTEST2.dir/clean

CMakeFiles/RLFRAMEWORKTEST2.dir/depend:
	cd /home/kevin/Development/NN/RLFramework/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/kevin/Development/NN/RLFramework /home/kevin/Development/NN/RLFramework /home/kevin/Development/NN/RLFramework/build /home/kevin/Development/NN/RLFramework/build /home/kevin/Development/NN/RLFramework/build/CMakeFiles/RLFRAMEWORKTEST2.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/RLFRAMEWORKTEST2.dir/depend

