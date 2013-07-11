add_executable(gtest-working-test gtest_test)
target_link_libraries(gtest-working-test gtest gtest_main)

add_test(gtest.working ${EXECUTABLE_OUTPUT_PATH}/gtest-working-test)

# add tests as dependencies of "check" and "tests"
add_dependencies(tests gtest-working-test)
add_dependencies(check tests)

