add_executable(device_query main ${CMAKE_SOURCE_DIR}/src/cuda/cuda)
target_link_libraries(device_query ${CUDA_LIBRARIES})
