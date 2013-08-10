#define MAIN

// project headers
#include "main_options.h"
#include <msc/cuda/core>
#include <msc/matrix.hpp>
#include <msc/core/gpu/block>

// stc C++ headers
#include <iostream>


// names
using std::clog;
using std::cout;


// macros
#define endl '\n'


template<typename T>
int _main () {
	double nanoseconds;
	matrix<T> t(filename);

	sqrtm(t.data_ptr(), t.rows(), block_size, nanoseconds);

	if (!ignore_output)
		cout << t << endl;

	if (print_time)
		cout << nanoseconds << endl;

	return 0;
}


int main (int argc, char * argv[]) {
	parse_arguments(argc, argv);
	int retval;

	#ifndef NDEBUG
	clog << "Options parsed:" << endl
	     << "filename: \"" << filename << '"' << endl
	     ;
	#endif

	CUDA::query_devices(true);
	if (CUDA::get_device(0).supports_double_precision())
		retval = _main<double>();
	else
		retval = _main<float>();

	return retval;
}
