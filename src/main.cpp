#define MAIN

// project headers
#include "main_options.h"
#include <msc/cuda>
#include <msc/matrix/print>

// stc C++ headers
#include <iostream>

// Armadillo headers
#include <armadillo>


// names
using std::clog;

using arma::Mat;


// macros
#define endl '\n'


template<typename T>
int _main () {
	Mat<T> arma_matrix;
	if (!arma_matrix.load(filename))
		return 1;

	const ulong m = arma_matrix.n_rows;

	if (print_sqrtm)
		print(arma_matrix);

	return 0;
}


int main (int argc, char * argv[]) {
	parse_arguments(argc, argv);

	#ifndef NDEBUG
	clog << "Options parsed:" << endl
	     << "filename: \"" << filename << '"' << endl
	     ;
	#endif

	CUDA::query_devices(true);
	if (CUDA::get_device(0).supports_double_precision())
		return _main<double>();
	else
		return _main<float>();
}
