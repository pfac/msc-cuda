#ifndef ___MSC__CUDA__ERROR_HPP___
#define ___MSC__CUDA__ERROR_HPP___

// std C++ headers
#include <iostream>


// types
typedef unsigned long ulong;


// macros
#define HANDLE_ERROR(e) handle_error(e, __FILE__, __LINE__)
#define endl '\n'


namespace CUDA {


	inline
	void handle_error (cudaError_t error, const char * file, const ulong line) throw(int) {
		if (error != cudaSuccess) {
			std::cerr << cudaGetErrorString(error) << endl;
			throw error;
		}
	}



}


#endif//___MSC__CUDA__ERROR_HPP___
