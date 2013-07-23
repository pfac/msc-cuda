function (add_gtest test source)
	list(REMOVE_ITEM ARGV ${test} ${source})
	set(executable ${test}-exec)
	add_executable(${executable} EXCLUDE_FROM_ALL ${source})
	target_link_libraries(${executable} gtest gtest_main ${ARGV})
	add_test(${test} ${EXECUTABLE_OUTPUT_PATH}/${executable})

	add_dependencies(tests ${executable})
endfunction (add_gtest)


add_gtest (gtest.working gtest_working)


# add tests as dependencies of "check" and "tests"


add_subdirectory(nvidia)

