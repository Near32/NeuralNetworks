cmake_minimum_required (VERSION 2.6)
project (RLFRAMEWORKCARTPOLEActorCriticUse)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../bin)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -g")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -pthread -L/usr/lib/python2.7/config-x86_64-linux-gnu -L/usr/lib -lpthread -ldl -lutil -lm  -lpython2.7 -Xlinker -export-dynamic -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,--no-as-needed")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -pthread -L/usr/lib/python2.7/config-x86_64-linux-gnu -L/usr/lib -lpthread -ldl  -lutil -lm  -lpython2.7 -Xlinker -export-dynamic -Wl,-O1 -Wl,-Bsymbolic-functions -I/usr/include/python2.7 -I/usr/include/x86_64-linux-gnu/python2.7  -fno-strict-aliasing -D_FORTIFY_SOURCE=2 -g -fstack-protector --param=ssp-buffer-size=4 -Wformat  -DNDEBUG -g -fwrapv -O2")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -pthread `python-config --ldflags --cflags`")

#########################################################
# Include Files
#########################################################

add_executable(RLFRAMEWORKCARTPOLEActorCriticUse actorcriticuse.cpp )

########################################################
# Linking & stuff
#########################################################

# create the program "Gunvarrel"
#target_link_libraries(Gunvarrel ${OPENGL_LIBRARIES} ${SDL_LIBRARY} )
