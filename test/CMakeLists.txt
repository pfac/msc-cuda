function (add_gtest test source)
	set(executable ${test}-exec)
	add_executable(${executable} EXCLUDE_FROM_ALL ${source})
	target_link_libraries(${executable} gtest gtest_main ${ARGN})
	add_test(${test} ${EXECUTABLE_OUTPUT_PATH}/${executable})

	add_dependencies(tests ${executable})
endfunction ()



function (add_gtest_cuda test source)
	set(executable ${test}-exec)
	CUDA_ADD_EXECUTABLE(${executable} EXCLUDE_FROM_ALL ${source})
	target_link_libraries(${executable} gtest gtest_main ${CUDA_LIBRARIES} ${ARGN})
	add_test(${test} ${EXECUTABLE_OUTPUT_PATH}/${executable})

	add_dependencies(tests ${executable})
endfunction ()


add_gtest (gtest.working gtest_working)


# add tests as dependencies of "check" and "tests"


add_subdirectory(nvidia)
add_subdirectory(linear_algebra)
add_subdirectory(cuda)
add_subdirectory(core)

