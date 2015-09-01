cmake_minimum_required(VERSION 2.8)
project(reco)

if (CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11 -Wall")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -Wall")
elseif (MSVC)
	
endif()


set(SOURCES 
	source/main.cpp
)

add_executable(reco ${SOURCES})

add_subdirectory(libnn)
include_directories(libnn/include)

find_package(OpenCL)
if(OPENCL_FOUND)
	message(STATUS "OPENCL_INCLUDE_DIRS: " ${OPENCL_INCLUDE_DIRS})
	message(STATUS "OPENCL_LIBRARIES: " ${OPENCL_LIBRARIES})
	include_directories(libnn/libcl4u/include ${OPENCL_INCLUDE_DIRS})
	target_link_libraries (${PROJECT_NAME} ${OPENCL_LIBRARIES} cl4u)
endif(OPENCL_FOUND)

target_link_libraries(${PROJECT_NAME} nn)
