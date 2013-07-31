cmake_minimum_required(VERSION 2.8.10)
project(MScThesisCUDA)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

include_directories(include)

# dependencies (begin)
find_package(Boost 1.34.0 REQUIRED COMPONENTS program_options)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

if (UNIX AND NOT APPLE)
	find_library(RT_LIBRARY rt)
endif ()
# dependencies (end)

include_directories(${ARMADILLO_INCLUDE_DIRS})

add_subdirectory(src)
add_subdirectory(doc)

# tests (begin)
enable_testing()

# gtest
add_subdirectory(vendor/gtest-1.6.0 EXCLUDE_FROM_ALL)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

add_custom_target(tests)

add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND})
add_dependencies(check tests)

add_subdirectory(test)

# tests (end)
