add_executable(cuda main main_options cuda/cuda)
target_link_libraries(cuda ${Boost_LIBRARIES} ${CUDA_LIBRARIES})
