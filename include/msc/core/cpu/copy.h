#ifndef ___COPY_H___
#define ___COPY_H___

// // MKL headers
// #include <mkl.h>


typedef unsigned long ulong;


template<typename T>
void copy (const T * const source, T * const destiny, const ulong n) {
	memcpy(destiny, source, n * sizeof(T));
}


// template<>
// void copy<double> (const double * const source, double * const destiny, const ulong n) {
// 	cblas_dcopy(n, source, 1, destiny, 1);
// }

#endif//___COPY_H___
