#define MAIN

// project headers
#include "main_options.h"
#include <msc/cuda>

// stc C++ headers
#include <iostream>


// names
using std::clog;


// macros
#define endl '\n'


int main (int argc, char * argv[]) {
	parse_arguments(argc, argv);

	#ifndef NDEBUG
	clog << "Options parsed:" << endl
	     << "filename: \"" << filename << '"' << endl
	     ;
	#endif

	CUDA::query_devices(true);
	if (CUDA::get_device(0).supports_double_precision()) {
		clog << "Running with double precision" << endl;
	} else
		clog << "Running with single precision" << endl;

	return 0;
}
