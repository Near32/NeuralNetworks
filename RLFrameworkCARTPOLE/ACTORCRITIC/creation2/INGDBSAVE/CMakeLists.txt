cmake_minimum_required (VERSION 2.6)
project (VALIDPOLICY)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../bin)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -pthread")


#########################################################
# Include Files
#########################################################

add_executable(VALIDPOLICY validationPOLICY.cpp )

########################################################
# Linking & stuff
#########################################################

# create the program "Gunvarrel"
#target_link_libraries(Gunvarrel ${OPENGL_LIBRARIES} ${SDL_LIBRARY} )
