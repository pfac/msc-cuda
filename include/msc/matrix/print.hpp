#ifndef ___MSC__MATRIX__PRINT_HPP___
#define ___MSC__MATRIX__PRINT_HPP___


// project headers
#include <msc/matrix.hpp>

// std C++ headers
#include <algorithm>
#include <iostream>

// Armadillo headers
#include <armadillo>


// names
using std::cout;
using std::min;
using std::ostream;
using arma::Mat;
using arma::span;


// types
typedef unsigned long ulong;


// macros
#define endl '\n'



template<typename T>
void print (const Mat<T> m, ostream& out = cout, unsigned max_rows = 5, unsigned max_cols = 5) {
	const unsigned i1 = min<unsigned>(m.n_rows, max_rows) - 1;
	const unsigned j1 = min<unsigned>(m.n_cols, max_cols) - 1;

	const span i = span(0, i1);
	const span j = span(0, j1);

	const Mat<T> p = m(i,j);

	out << p << endl;
}



template<typename T>
void print (const T * const m, const ulong rows, const ulong cols, ostream& out = cout, ulong max_rows = 5, ulong max_cols = 5) {
	Mat<T> arma_m(rows, cols);

	array2arma(m, rows, cols, arma_m);
	print(arma_m, out, max_rows, max_cols);
}



#endif//___MSC__MATRIX__PRINT_HPP___
