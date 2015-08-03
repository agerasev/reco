cmake_minimum_required(VERSION 2.8)
project(reco)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11 -Wall")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -Wall")

set(SOURCES 
	source/main.cpp
)

add_executable(reco ${SOURCES})

add_subdirectory(libnn)
include_directories(libnn/include)

find_package(OpenCL)
if(OPENCL_FOUND)
	add_definitions(-DNN_OPENCL)
	include_directories(libnn/libcl4u)
	target_link_libraries (${PROJECT_NAME} ${OPENCL_LIBRARIES})
endif(OPENCL_FOUND)

target_link_libraries(${PROJECT_NAME} libnn)
