CUDA_ADD_EXECUTABLE(cuda main.cu main_options cuda/cuda)
target_link_libraries(cuda ${Boost_LIBRARIES} ${CUDA_LIBRARIES})

add_subdirectory(arma EXCLUDE_FROM_ALL)
