#define MAIN

// std C++ headers
#include <iostream>

// Armadillo headers
#include <armadillo>

// extra headers
#include "main_options.h"


// names
using std::cin;
using std::cout;

using arma::arma_ascii;
using arma::Mat;


int main (int argc, char * argv[]) {
	parse_arguments(argc, argv);

	Mat<double> matrix;

	if (input_filename.empty())
		matrix.load(cin);
	else
		matrix.load(input_filename);

	matrix *= matrix;

	if (output_filename.empty())
		matrix.save(cout, arma_ascii);
	else
		matrix.save(output_filename, arma_ascii);

	return 0;
}
