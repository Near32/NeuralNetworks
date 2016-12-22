cmake_minimum_required (VERSION 2.6)
project (DDPGA3C_OPENAIGYM)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../bin)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -g")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -pthread -L/usr/lib/python2.7/config-x86_64-linux-gnu -L/usr/lib -lpthread -ldl -lutil -lm  -lpython2.7 -Xlinker -export-dynamic -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,--no-as-needed")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -pthread -I/usr/include/python2.7 -L/usr/lib/python2.7/config-x86_64-linux-gnu -L/usr/lib -lpthread -ldl -lutil -lm  -lpython2.7 -Xlinker -export-dynamic -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,--no-as-needed")


#########################################################
# Include Files
#########################################################

add_executable(DDPGA3C_OPENAIGYM DDPGA3C_OpenAIGym.cpp )

########################################################
# Linking & stuff
#########################################################
