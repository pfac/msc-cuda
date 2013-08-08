add_executable(square main main_options)
target_link_libraries(square ${Boost_LIBRARIES} ${BLAS_LIBRARIES})
