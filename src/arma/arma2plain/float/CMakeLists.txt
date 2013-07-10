add_definitions(-DTYPE=float)
add_executable(arma2plain-float ../main ../main_options)
target_link_libraries(arma2plain-float ${Boost_LIBRARIES})
