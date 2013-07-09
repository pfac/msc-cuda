#ifndef ___MSC__MATRIX__ARMA_HPP___
#define ___MSC__MATRIX__ARMA_HPP___

// Armadillo headers
#include <armadillo>


// names
using arma::Mat;
using arma::span;


// types
typedef unsigned long ulong;



template<typename T>
void arma2array (const Mat<T>& arma_matrix, T * const arr) {
	const ulong m = arma_matrix.n_rows;
	const ulong n = arma_matrix.n_cols;

	#pragma omp parallel for collapse(2)
	for (ulong j = 0; j < n; ++j)
		for (ulong i = 0; i < m; ++i)
			arr[j * m + i] = arma_matrix(i,j);
}



template<typename T>
void array2arma (const T * const arr, const ulong m, const ulong n, Mat<T>& arma_matrix) {
	arma_matrix.set_size(m, n);

	#pragma omp parallel for collapse(2)
	for (ulong j = 0; j < n; ++j)
		for (ulong i = 0; i < m; ++i)
			arma_matrix(i,j) = arr[j * m + i];
}



#endif//___MSC__MATRIX__ARMA_HPP___
