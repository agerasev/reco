cmake_minimum_required(VERSION 2.8)
project(reco)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11 -Wall")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -Wall")

set(SOURCES 
	source/main.cpp
)

include_directories(libnn/include)
include_directories(libnn/libcl4u)

add_subdirectory(libnn)
add_executable(reco ${SOURCES})
target_link_libraries(${PROJECT_NAME} OpenCL libnn)
