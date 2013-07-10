#define MAIN

// project headers
#include "main_options.h"
#include <msc/matrix.hpp>

// std C++ headers
#include <iostream>

// Armadillo headers
#include <armadillo>


// names
using std::clog;
using std::cout;
using arma::Mat;


// types
typedef unsigned long ulong;


// macros
#define endl '\n'


int main(int argc, char *argv[]) {
	parse_arguments(argc, argv);

	Mat<TYPE> arma_m;
	if (!arma_m.load(filename))
		return 1;

	if (verbose)
		clog << arma_m << endl;

	const ulong r = arma_m.n_rows;
	const ulong c = arma_m.n_cols;

	matrix<TYPE> m(r, c);

	for (ulong j = 0; j < c; ++j)
		for (ulong i = 0; i < r; ++i)
			m(i,j) = arma_m(i,j);

	cout << m << endl;

	return 0;
}
