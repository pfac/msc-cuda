#define MAIN

// project headers
#include "main_options.h"
#include <msc/cuda/core>
#include <msc/matrix.hpp>
#include <msc/core/gpu/block>

#include "../vendor/nvidia/cuda/cuPrintf.cu"

// stc C++ headers
#include <iostream>


// names
using std::clog;
using std::cout;


// macros
#define endl '\n'


template<typename T>
int _main () {
	matrix<T> t(filename);

	sqrtm(t.data_ptr(), t.rows(), block_size);

	if (print_sqrtm)
		cout << t << endl; 

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

	cudaPrintfInit();

	CUDA::query_devices(true);
	if (CUDA::get_device(0).supports_double_precision())
		retval = _main<double>();
	else
		retval = _main<float>();

	cudaDeviceSynchronize();
	cudaPrintfDisplay(stdout, true);
	cudaPrintfEnd();
	cudaDeviceReset();

	return retval;
}
