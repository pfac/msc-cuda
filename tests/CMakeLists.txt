add_executable(test-gtest-working gtest_working)
target_link_libraries(test-gtest-working gtest gtest_main)

add_test(gtest.working ${EXECUTABLE_OUTPUT_PATH}/test-gtest-working)

# # add tests as dependencies of "check" and "tests"
# add_dependencies(tests gtest-working-test)
# add_dependencies(check tests)

add_subdirectory(nvidia)

