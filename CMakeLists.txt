cmake_minimum_required(VERSION 2.8.10)
project(MScThesisCUDA)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

include_directories(include)

# dependencies
find_package(Boost 1.34.0 REQUIRED COMPONENTS program_options)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

if (UNIX AND NOT APPLE)
	find_library(RT_LIBRARY rt)
endif ()

include_directories(${ARMADILLO_INCLUDE_DIRS})

add_subdirectory(src)
