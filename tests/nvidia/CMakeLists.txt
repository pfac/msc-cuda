add_executable(test-device-query device_query)
target_link_libraries(test-device-query ${CUDA_LIBRARIES} gtest gtest_main)

add_test(NVIDIA.deviceQuery ${EXECUTABLE_OUTPUT_PATH}/test-device-query)
