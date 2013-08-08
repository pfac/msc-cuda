#define MAIN

// std C++ headers
#include <fstream>
#include <iostream>

// Armadillo headers
#include <armadillo>

// extra headers
#include "main_options.h"


// names
using std::cout;
using std::ostream;
using std::ofstream;

using arma::arma_ascii;
using arma::Mat;


// types
typedef unsigned long ulong;


// macros
#define ATOI(s) strtol(s, NULL, 0)
#define endl '\n'


template<typename T>
void upper_triangular_control (Mat<T>& matrix) {
	const ulong n = matrix.n_rows;

	for (ulong j = 0; j < n; ++j) {
		T x = T(j) * (T(j) + T(1)) / T(2);
		
		#pragma omp parallel for
		for (ulong i = 0; i < n; ++i)
			matrix(i,j) = (j < i) ? T(0) : ++x;
	}
}


template<typename T>
void upper_triangular (Mat<T>& matrix, const bool control) {
	if (control)
		upper_triangular_control<T>(matrix);
	else {
		matrix = trimatu(matrix.randu());
	}
}

template<typename T>
void save (ostream& out, const Mat<T>& matrix) {
	matrix.save(out, arma_ascii);
}


int main (int argc, char * argv[]) {
	parse_arguments(argc, argv);

	Mat<double> matrix(dimension, dimension);
	upper_triangular(matrix, control);

	if (output_filename.empty())
		matrix.save(cout, arma_ascii);
	else
		matrix.save(output_filename, arma_ascii);

	return 0;
}
