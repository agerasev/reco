cmake_minimum_required(VERSION 2.6)
project(reco)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11 -Wall")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -Wall")
file(STRINGS "reco.files" SOURCES REGEX ^source/.*.c$|^source/.*.cpp$)
add_definitions(-DCL_PROFILING)
include_directories(libnn/include)
include_directories(libnn/libcl4u)

add_executable(reco ${SOURCES})
target_link_libraries(${PROJECT_NAME} -lOpenCL)
