find_package(Doxygen)
if(DOXYGEN_FOUND)
	configure_file(Doxyfile.in Doxyfile @ONLY)
	add_custom_target(doc 
		${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
		COMMENT "Generating API documentation with Doxygen"
		VERBATIM
	)
endif(DOXYGEN_FOUND)
